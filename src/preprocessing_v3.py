"""
preprocessing_v3.py — Hybrid Physics-ML Pipeline (V_proxy + I_unscaled)
=========================================================================
Replaces raw V_t with V_proxy = V_t - I * R_int(T) to remove Ohmic Drop
Illusion. Also saves unscaled current for the Hard Constraint output layer.

Changes from v2:
  1. V_proxy replaces V_t at feature index 0
  2. dV_proxy/dt replaces dV/dt at feature index 3
  3. Saves I_unscaled_{split}.npy for Hard Constraint layer
  4. Updated physics scaling bounds (V_proxy range is narrower)

Created : 2026-05-16
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import warnings
import gc
from tqdm import tqdm
from scipy.interpolate import PchipInterpolator

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import R_INT_PER_TEMP, PHYS_MIN_V3, PHYS_MAX_V3

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW  = os.path.join(BASE_DIR, "data", "raw", "LG Dataset", "LG_HG2_Original_Dataset")
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(DATA_PROC, exist_ok=True)

Q_NOMINAL = 3.0

Q_ACTUAL_PER_TEMP = {
    '25degC':  2.7744, '10degC':  2.6102, '0degC':   2.5764,
    '40degC':  2.7180, 'n10degC': 2.4919, 'n20degC': 2.3304,
}

FEATURE_COLS_V3 = ['V_proxy', 'Current', 'Temperature', 'dV_proxy_dt', 'dI_dt']


# =====================================================================
# 1. CSV Reading (reused from v2)
# =====================================================================
def find_header_row(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if 'Voltage' in line and 'Current' in line:
                return i
    raise ValueError(f"Header not found: {filepath}")


def read_csv(filepath):
    hrow = find_header_row(filepath)
    df = pd.read_csv(filepath, skiprows=hrow, encoding='utf-8',
                     encoding_errors='replace', low_memory=False)
    df.columns = df.columns.str.strip()

    if 'Time Stamp' in df.columns:
        df['Time Stamp'] = pd.to_datetime(
            df['Time Stamp'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
        if df['Time Stamp'].isna().all():
            df['Time Stamp'] = pd.to_datetime(
                df['Time Stamp'], errors='coerce', format='mixed')
        t0 = df['Time Stamp'].dropna().iloc[0]
        df['time_sec'] = (df['Time Stamp'] - t0).dt.total_seconds()
    else:
        df['time_sec'] = np.arange(len(df)) * 0.1

    for col in ['Voltage', 'Current', 'Temperature', 'Capacity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df.dropna(subset=['Voltage', 'Current', 'time_sec']).reset_index(drop=True)


# =====================================================================
# 2. OCV-SOC Lookup (reused from v2)
# =====================================================================
def build_ocv_soc_lookup(temp_name):
    temp_dir = os.path.join(DATA_RAW, temp_name)
    hppc_files = glob.glob(os.path.join(temp_dir, '*HPPC*.csv'))

    if not hppc_files:
        print(f"    [WARNING] No HPPC file for {temp_name}, using fallback")
        return None, Q_ACTUAL_PER_TEMP.get(temp_name, Q_NOMINAL)

    fpath = hppc_files[0]
    hrow = find_header_row(fpath)
    df = pd.read_csv(fpath, skiprows=hrow, encoding='utf-8',
                     encoding_errors='replace', low_memory=False)
    df.columns = df.columns.str.strip()

    for col in ['Voltage', 'Current', 'Capacity']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Time Stamp'] = pd.to_datetime(
        df['Time Stamp'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    if df['Time Stamp'].isna().all():
        df['Time Stamp'] = pd.to_datetime(
            df['Time Stamp'], errors='coerce', format='mixed')
    t0 = df['Time Stamp'].dropna().iloc[0]
    df['time_sec'] = (df['Time Stamp'] - t0).dt.total_seconds()

    q_actual = df['Capacity'].abs().max()

    is_rest = df['Current'].abs() < 0.01
    rest_groups = (is_rest != is_rest.shift()).cumsum()
    rest_segments = df[is_rest].groupby(rest_groups)

    ocv_list, soc_list = [], []
    for _, seg in rest_segments:
        duration = seg['time_sec'].max() - seg['time_sec'].min()
        if duration >= 600:
            ocv = seg['Voltage'].iloc[-1]
            cap_used = abs(seg['Capacity'].iloc[-1])
            soc = np.clip(1.0 - cap_used / q_actual, 0.0, 1.0)
            ocv_list.append(ocv)
            soc_list.append(soc)

    if len(ocv_list) < 5:
        return None, q_actual

    pairs = sorted(zip(ocv_list, soc_list), key=lambda x: x[0])
    clean_ocv, clean_soc = [pairs[0][0]], [pairs[0][1]]
    for ocv, soc in pairs[1:]:
        if ocv > clean_ocv[-1]:
            clean_ocv.append(ocv)
            clean_soc.append(soc)

    if len(clean_ocv) < 4:
        return None, q_actual

    interp_func = PchipInterpolator(clean_ocv, clean_soc, extrapolate=True)
    return interp_func, q_actual


# =====================================================================
# 3. Feature Engineering v3 — V_proxy + dV_proxy/dt
# =====================================================================
def engineer_features_v3(df, q_actual, r_int, ocv_lookup=None):
    """
    Compute V_proxy and calibrated SOC ground truth.

    V_proxy = V_t - I * R_int
    (During discharge I<0, so V_proxy > V_t, recovering OCV approximation)
    """
    df = df.copy()

    # SOC_initial from OCV lookup
    v_initial = df['Voltage'].iloc[0]
    if ocv_lookup is not None:
        soc_initial = float(np.clip(ocv_lookup(v_initial), 0.0, 1.0))
    else:
        soc_initial = 1.0

    # SOC from Capacity column
    if 'Capacity' in df.columns and df['Capacity'].notna().sum() > 0:
        cap = df['Capacity'].fillna(0.0)
        cap_offset = cap.iloc[0]
        cap_used = (cap - cap_offset).abs()
        df['SOC_cc'] = (soc_initial - cap_used / q_actual).clip(0.0, 1.0)
    else:
        dt = df['time_sec'].diff().fillna(0).abs().clip(upper=10.0)
        delta_ah = (df['Current'] * dt) / 3600.0
        df['SOC_cc'] = (soc_initial + delta_ah.cumsum() / q_actual).clip(0.0, 1.0)

    # V_proxy: remove Ohmic drop
    df['V_proxy'] = df['Voltage'] - df['Current'] * r_int

    # Kinetic derivatives (using V_proxy, not raw V_t)
    dt = df['time_sec'].diff().fillna(0).abs().clip(upper=10.0)
    safe_dt = dt.replace(0, np.nan)
    df['dV_proxy_dt'] = (df['V_proxy'].diff() / safe_dt).fillna(0).clip(-2.0, 2.0)
    df['dI_dt'] = (df['Current'].diff() / safe_dt).fillna(0).clip(-20.0, 20.0)

    if 'Temperature' not in df.columns:
        df['Temperature'] = 25.0

    return df, soc_initial


# =====================================================================
# 4. Sequence Building (with unscaled current extraction)
# =====================================================================
def build_sequences_v3(df, window=100, stride=10):
    """
    Build sequences with V_proxy features + unscaled current.

    Returns
    -------
    X         : (N, window, 5)  — [V_proxy, I, T, dV_proxy/dt, dI/dt]
    y         : (N, window)     — SOC trajectory
    I_unscaled: (N, window)     — raw current in Amperes for Hard Constraint
    """
    required = FEATURE_COLS_V3 + ['SOC_cc']
    if not all(c in df.columns for c in required):
        return None, None, None

    data   = df[FEATURE_COLS_V3].values.astype(np.float32)
    labels = df['SOC_cc'].values.astype(np.float32)
    current_raw = df['Current'].values.astype(np.float32)

    X, y, I_raw = [], [], []
    for s in range(0, len(data) - window + 1, stride):
        X.append(data[s:s+window])
        y.append(labels[s:s+window])
        I_raw.append(current_raw[s:s+window])

    if not X:
        return None, None, None
    return np.array(X), np.array(y), np.array(I_raw)


# =====================================================================
# 5. Main Pipeline v3
# =====================================================================
def run_pipeline_v3(scenario='A', window=100, stride=10):
    """Process raw CSVs into v3 datasets with V_proxy + I_unscaled."""
    print(f"{'='*65}")
    print(f"  Pipeline v3: Hybrid Physics-ML — Scenario {scenario}")
    print(f"  V_proxy = V_t - I*R_int | Hard Constraint I_unscaled")
    print(f"{'='*65}")

    all_temps = ['0degC', '10degC', '25degC', '40degC', 'n10degC', 'n20degC']
    temp_data = {t: {'X': [], 'y': [], 'I': [], 'soc_initials': []} for t in all_temps}

    metadata = {
        'version': 'v3',
        'method': 'V_proxy (Ohmic-corrected) + Hard Constraint',
        'feature_cols': FEATURE_COLS_V3,
        'r_int_per_temp': {k: round(v, 5) for k, v in R_INT_PER_TEMP.items()},
        'q_actual': {},
        'ocv_points': {},
        'soc_initial_stats': {},
    }

    for temp in tqdm(all_temps, desc=f"Processing Temps (v3, Scenario {scenario})"):
        temp_dir = os.path.join(DATA_RAW, temp)
        if not os.path.exists(temp_dir):
            print(f"  [{temp}] Missing directory, skipping.")
            continue

        ocv_lookup, q_actual = build_ocv_soc_lookup(temp)
        r_int = R_INT_PER_TEMP.get(temp, 0.03)  # fallback 30 mOhm

        metadata['q_actual'][temp] = round(q_actual, 4)
        metadata['ocv_points'][temp] = 'PCHIP' if ocv_lookup is not None else 'fallback'

        print(f"  [{temp}] Q={q_actual:.4f} Ah, R_int={r_int*1000:.2f} mOhm, "
              f"OCV={'PCHIP' if ocv_lookup else 'None'}")

        csvs = glob.glob(os.path.join(temp_dir, '*.csv'))
        soc_initials = []

        for c in tqdm(sorted(csvs), desc=f"CSVs for {temp}", leave=False):
            fname = os.path.basename(c).lower()
            if not any(k in fname for k in ['udds', 'la92', 'hwfet', 'us06', 'mixed']):
                continue
            try:
                df = read_csv(c)
                df, soc_init = engineer_features_v3(df, q_actual, r_int, ocv_lookup)
                soc_initials.append(soc_init)

                X, y, I_raw = build_sequences_v3(df, window, stride)
                if X is not None:
                    temp_data[temp]['X'].append(X)
                    temp_data[temp]['y'].append(y)
                    temp_data[temp]['I'].append(I_raw)
            except Exception:
                pass

        if temp_data[temp]['X']:
            temp_data[temp]['X'] = np.concatenate(temp_data[temp]['X'], axis=0)
            temp_data[temp]['y'] = np.concatenate(temp_data[temp]['y'], axis=0)
            temp_data[temp]['I'] = np.concatenate(temp_data[temp]['I'], axis=0)
            n_seq = len(temp_data[temp]['X'])
        else:
            temp_data[temp]['X'] = np.array([])
            temp_data[temp]['y'] = np.array([])
            temp_data[temp]['I'] = np.array([])
            n_seq = 0

        if soc_initials:
            metadata['soc_initial_stats'][temp] = {
                'mean': round(float(np.mean(soc_initials)), 4),
                'min': round(float(np.min(soc_initials)), 4),
                'max': round(float(np.max(soc_initials)), 4),
                'count': len(soc_initials),
            }
        print(f"    -> {n_seq} sequences")

    # ── Splitting Logic ──────────────────────────────────────────────
    print(f"\n--- Splitting Logic (Scenario {scenario}) ---")

    if scenario == 'A':
        train_temps = ['25degC', '10degC']
        test_temps  = ['40degC', 'n10degC', 'n20degC']

        train_X, train_y, train_I = [], [], []
        val_X, val_y, val_I = [], [], []

        for t in train_temps:
            X_t = temp_data[t]['X']
            y_t = temp_data[t]['y']
            I_t = temp_data[t]['I']
            if len(X_t) == 0:
                continue
            n_val_t = max(1, int(0.10 * len(X_t)))
            train_X.append(X_t[:-n_val_t]); train_y.append(y_t[:-n_val_t]); train_I.append(I_t[:-n_val_t])
            val_X.append(X_t[-n_val_t:]);   val_y.append(y_t[-n_val_t:]);   val_I.append(I_t[-n_val_t:])
            print(f"  [{t}] train={len(X_t)-n_val_t:,}  val={n_val_t:,}")

        # Augment val with 0degC
        for key, lst in [('X', val_X), ('y', val_y), ('I', val_I)]:
            arr = temp_data['0degC'][key]
            if len(arr) > 0:
                lst.append(arr)
        if len(temp_data['0degC']['X']) > 0:
            print(f"  [0degC]  val={len(temp_data['0degC']['X']):,}")

        X_train = np.concatenate(train_X) if train_X else np.empty((0, window, 5), dtype=np.float32)
        y_train = np.concatenate(train_y) if train_y else np.empty((0, window), dtype=np.float32)
        I_train = np.concatenate(train_I) if train_I else np.empty((0, window), dtype=np.float32)
        del train_X, train_y, train_I

        X_val = np.concatenate(val_X) if val_X else np.empty((0, window, 5), dtype=np.float32)
        y_val = np.concatenate(val_y) if val_y else np.empty((0, window), dtype=np.float32)
        I_val = np.concatenate(val_I) if val_I else np.empty((0, window), dtype=np.float32)
        del val_X, val_y, val_I

        X_test = _concat_temps(temp_data, test_temps, 'X', window)
        y_test = _concat_temps(temp_data, test_temps, 'y')
        I_test = _concat_temps(temp_data, test_temps, 'I')
        temp_labels_test = _build_temp_labels(temp_data, test_temps)

        # Shuffle train only
        rng = np.random.default_rng(42)
        if len(X_train) > 0:
            idx = rng.permutation(len(X_train))
            X_train, y_train, I_train = X_train[idx], y_train[idx], I_train[idx]

    elif scenario == 'B':
        X_tr, y_tr, I_tr = [], [], []
        X_vl, y_vl, I_vl = [], [], []
        X_te, y_te, I_te = [], [], []
        test_label_list = []

        for temp in all_temps:
            X_tmp = temp_data[temp]['X']
            y_tmp = temp_data[temp]['y']
            I_tmp = temp_data[temp]['I']
            if len(X_tmp) == 0:
                continue
            n_tr = int(0.7 * len(X_tmp))
            n_vl = int(0.1 * len(X_tmp))

            X_tr.append(X_tmp[:n_tr]);           y_tr.append(y_tmp[:n_tr]);           I_tr.append(I_tmp[:n_tr])
            X_vl.append(X_tmp[n_tr:n_tr+n_vl]);  y_vl.append(y_tmp[n_tr:n_tr+n_vl]);  I_vl.append(I_tmp[n_tr:n_tr+n_vl])
            X_te.append(X_tmp[n_tr+n_vl:]);       y_te.append(y_tmp[n_tr+n_vl:]);       I_te.append(I_tmp[n_tr+n_vl:])
            test_label_list.extend([temp] * len(X_tmp[n_tr+n_vl:]))

        X_train = np.concatenate(X_tr) if X_tr else np.empty((0, window, 5), dtype=np.float32)
        y_train = np.concatenate(y_tr) if y_tr else np.empty((0, window), dtype=np.float32)
        I_train = np.concatenate(I_tr) if I_tr else np.empty((0, window), dtype=np.float32)
        X_val   = np.concatenate(X_vl) if X_vl else np.empty((0, window, 5), dtype=np.float32)
        y_val   = np.concatenate(y_vl) if y_vl else np.empty((0, window), dtype=np.float32)
        I_val   = np.concatenate(I_vl) if I_vl else np.empty((0, window), dtype=np.float32)
        X_test  = np.concatenate(X_te) if X_te else np.empty((0, window, 5), dtype=np.float32)
        y_test  = np.concatenate(y_te) if y_te else np.empty((0, window), dtype=np.float32)
        I_test  = np.concatenate(I_te) if I_te else np.empty((0, window), dtype=np.float32)
        temp_labels_test = np.array(test_label_list)
        del X_tr, y_tr, I_tr, X_vl, y_vl, I_vl, X_te, y_te, I_te

        rng = np.random.default_rng(42)
        if len(X_train) > 0:
            idx = rng.permutation(len(X_train))
            X_train, y_train, I_train = X_train[idx], y_train[idx], I_train[idx]
    else:
        raise ValueError("Invalid scenario")

    del temp_data
    gc.collect()

    # ── Shape Guards ─────────────────────────────────────────────────
    assert X_train.ndim == 3 and X_train.shape[2] == 5, f"Bad X: {X_train.shape}"
    assert y_train.ndim == 2 and y_train.shape[1] == window, f"Bad y: {y_train.shape}"
    assert I_train.ndim == 2 and I_train.shape[1] == window, f"Bad I: {I_train.shape}"

    # ── Physics-Informed Scaling (v3 bounds) ─────────────────────────
    PHYS_MIN = np.array(PHYS_MIN_V3, dtype=np.float32)
    PHYS_MAX = np.array(PHYS_MAX_V3, dtype=np.float32)
    PHYS_RANGE = PHYS_MAX - PHYS_MIN

    p_min = PHYS_MIN.reshape(1, 1, 5)
    p_rng = PHYS_RANGE.reshape(1, 1, 5)

    X_train -= p_min; X_train /= p_rng
    if len(X_val)  > 0: X_val  -= p_min; X_val  /= p_rng
    if len(X_test) > 0: X_test -= p_min; X_test /= p_rng

    print(f"\n  v3 Physics Scaling applied:")
    print(f"    Bounds: V_proxy[2.5,4.25] I[-20,20] T[-20,50] dVp[-2,2] dI[-20,20]")
    print(f"    Train range: [{X_train.min():.4f}, {X_train.max():.4f}]")

    # ── V_proxy sanity check ─────────────────────────────────────────
    v_proxy_col = X_train[:, :, 0]
    print(f"    V_proxy (scaled) — mean={v_proxy_col.mean():.4f}, "
          f"std={v_proxy_col.std():.4f}, "
          f"[{v_proxy_col.min():.4f}, {v_proxy_col.max():.4f}]")

    # ── SOC Distribution ─────────────────────────────────────────────
    print(f"\n  SOC Distribution:")
    print(f"    Train: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
    if len(y_val)  > 0: print(f"    Val  : mean={y_val.mean():.4f}, std={y_val.std():.4f}")
    if len(y_test) > 0: print(f"    Test : mean={y_test.mean():.4f}, std={y_test.std():.4f}")

    # ── Save to disk ────────────────────────────────────────────────
    out_dir = os.path.join(DATA_PROC, f'v3_scenario_{scenario}')
    os.makedirs(out_dir, exist_ok=True)

    for name, arrs in [
        ('train', (X_train, y_train, I_train)),
        ('val',   (X_val,   y_val,   I_val)),
        ('test',  (X_test,  y_test,  I_test)),
    ]:
        np.save(os.path.join(out_dir, f"X_{name}.npy"), arrs[0])
        np.save(os.path.join(out_dir, f"y_{name}.npy"), arrs[1])
        np.save(os.path.join(out_dir, f"I_unscaled_{name}.npy"), arrs[2])

    np.save(os.path.join(out_dir, "X_min.npy"), p_min)
    np.save(os.path.join(out_dir, "X_max.npy"), p_min + p_rng)
    np.save(os.path.join(out_dir, "temp_labels_test.npy"), temp_labels_test)

    metadata['shapes'] = {
        'X_train': list(X_train.shape), 'y_train': list(y_train.shape),
        'I_train': list(I_train.shape),
        'X_val':   list(X_val.shape),   'y_val':   list(y_val.shape),
        'X_test':  list(X_test.shape),  'y_test':  list(y_test.shape),
    }
    with open(os.path.join(out_dir, "metadata_v3.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  X_train : {X_train.shape}")
    print(f"  y_train : {y_train.shape}")
    print(f"  I_train : {I_train.shape}")
    print(f"  X_val   : {X_val.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  Saved to: {out_dir}\n")


# ── Helpers ──────────────────────────────────────────────────────────
def _concat_temps(temp_data, temps, key, window=100):
    arrays = [temp_data[t][key] for t in temps if len(temp_data[t][key]) > 0]
    if arrays:
        return np.concatenate(arrays, axis=0)
    if key == 'X':
        return np.empty((0, window, 5), dtype=np.float32)
    return np.empty((0, window), dtype=np.float32)


def _build_temp_labels(temp_data, temps):
    labels = []
    for t in temps:
        if len(temp_data[t]['X']) > 0:
            labels.extend([t] * len(temp_data[t]['X']))
    return np.array(labels)


if __name__ == "__main__":
    run_pipeline_v3('A')
    run_pipeline_v3('B')
