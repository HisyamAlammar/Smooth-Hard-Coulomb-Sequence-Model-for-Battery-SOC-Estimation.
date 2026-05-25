"""
preprocessing.py — Sprint 4.2: Physics-Calibrated SOC Ground Truth
====================================================================
Temperature-dependent Q_actual + OCV-SOC anchor calibration using HPPC
data. Generates v2 datasets with per-temperature metadata.

Key improvements over v1:
  1. Q_actual extracted per-temperature from HPPC files (not global Q_NOMINAL)
  2. SOC_initial calibrated via OCV-SOC lookup (PCHIP interpolation)
  3. SOC computed from hardware Capacity column (more accurate than manual CC)
  4. Temperature labels saved for per-temp RMSE breakdown

Created : 2026-04-07
Updated : 2026-04-08 — Sprint 4.2: OCV-anchor + temp-dependent capacity
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import warnings
import gc
from tqdm import tqdm
from scipy.interpolate import PchipInterpolator

warnings.filterwarnings('ignore')

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW  = os.path.join(BASE_DIR, "data", "raw", "LG Dataset", "LG_HG2_Original_Dataset")
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(DATA_PROC, exist_ok=True)

Q_NOMINAL = 3.0  # Fallback only

# ── Temperature-specific Q_actual from HPPC (verified 2026-04-08) ────
Q_ACTUAL_PER_TEMP = {
    '25degC':  2.7744,
    '10degC':  2.6102,
    '0degC':   2.5764,
    '40degC':  2.7180,
    'n10degC': 2.4919,
    'n20degC': 2.3304,
}


# =====================================================================
# 1. CSV Reading Utilities
# =====================================================================
def find_header_row(filepath):
    """Find the row containing column headers (Voltage, Current)."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if 'Voltage' in line and 'Current' in line:
                return i
    raise ValueError(f"Header not found: {filepath}")


def read_csv(filepath):
    """Read a battery CSV file with automatic header detection."""
    hrow = find_header_row(filepath)
    df = pd.read_csv(filepath, skiprows=hrow, encoding='utf-8',
                     encoding_errors='replace', low_memory=False)
    df.columns = df.columns.str.strip()

    # Parse timestamps
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

    # Coerce numeric columns
    for col in ['Voltage', 'Current', 'Temperature', 'Capacity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df.dropna(subset=['Voltage', 'Current', 'time_sec']).reset_index(drop=True)


# =====================================================================
# 2. OCV-SOC Lookup Table Construction (from HPPC)
# =====================================================================
def build_ocv_soc_lookup(temp_name):
    """
    Build a PCHIP interpolator: voltage -> SOC from HPPC rest segments.

    Uses rest periods >= 600s where voltage has stabilized to approximate
    Open Circuit Voltage (OCV) at known SOC levels.

    Returns
    -------
    interp_func : PchipInterpolator  (input: voltage, output: SOC)
    q_actual    : float              (total extractable capacity in Ah)
    """
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

    # Parse timestamps for duration calculation
    df['Time Stamp'] = pd.to_datetime(
        df['Time Stamp'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    if df['Time Stamp'].isna().all():
        df['Time Stamp'] = pd.to_datetime(
            df['Time Stamp'], errors='coerce', format='mixed')
    t0 = df['Time Stamp'].dropna().iloc[0]
    df['time_sec'] = (df['Time Stamp'] - t0).dt.total_seconds()

    # Q_actual = total capacity throughput from HPPC
    q_actual = df['Capacity'].abs().max()

    # Extract OCV-SOC pairs from long rest segments (>= 600s)
    is_rest = df['Current'].abs() < 0.01
    rest_groups = (is_rest != is_rest.shift()).cumsum()
    rest_segments = df[is_rest].groupby(rest_groups)

    ocv_list = []
    soc_list = []

    for _, seg in rest_segments:
        duration = seg['time_sec'].max() - seg['time_sec'].min()
        if duration >= 600:
            ocv = seg['Voltage'].iloc[-1]
            cap_used = abs(seg['Capacity'].iloc[-1])
            soc = np.clip(1.0 - cap_used / q_actual, 0.0, 1.0)
            ocv_list.append(ocv)
            soc_list.append(soc)

    if len(ocv_list) < 5:
        print(f"    [WARNING] Only {len(ocv_list)} OCV points for {temp_name}")
        return None, q_actual

    # Sort by OCV ascending for monotonic interpolation
    pairs = sorted(zip(ocv_list, soc_list), key=lambda x: x[0])

    # Remove duplicate OCV values (keep first)
    clean_ocv = [pairs[0][0]]
    clean_soc = [pairs[0][1]]
    for ocv, soc in pairs[1:]:
        if ocv > clean_ocv[-1]:  # strictly increasing OCV
            clean_ocv.append(ocv)
            clean_soc.append(soc)

    if len(clean_ocv) < 4:
        print(f"    [WARNING] Too few unique OCV points for {temp_name}")
        return None, q_actual

    # Build PCHIP interpolator (monotone cubic, no overshoot)
    interp_func = PchipInterpolator(clean_ocv, clean_soc, extrapolate=True)

    return interp_func, q_actual


# =====================================================================
# 3. Feature Engineering with Calibrated SOC
# =====================================================================
def engineer_features(df, q_actual, ocv_lookup=None):
    """
    Compute features + calibrated SOC ground truth.

    SOC is reconstructed as:
      SOC(t) = SOC_initial - abs(Capacity(t) - Capacity(0)) / Q_actual

    Where SOC_initial is determined by looking up V(0) in OCV-SOC table.
    If no lookup is available, falls back to SOC_initial = 1.0.

    Parameters
    ----------
    df         : DataFrame with Voltage, Current, Temperature, Capacity
    q_actual   : float, temperature-specific extractable capacity (Ah)
    ocv_lookup : PchipInterpolator or None
    """
    df = df.copy()

    # --- Determine SOC_initial from OCV lookup ---
    v_initial = df['Voltage'].iloc[0]
    if ocv_lookup is not None:
        soc_initial = float(np.clip(ocv_lookup(v_initial), 0.0, 1.0))
    else:
        soc_initial = 1.0  # Fallback: assume fully charged

    # --- Compute SOC from Capacity column (preferred) ---
    if 'Capacity' in df.columns and df['Capacity'].notna().sum() > 0:
        cap = df['Capacity'].fillna(0.0)
        cap_offset = cap.iloc[0]  # Handle non-zero starting capacity
        cap_used = (cap - cap_offset).abs()
        df['SOC_cc'] = (soc_initial - cap_used / q_actual).clip(0.0, 1.0)
    else:
        # Fallback: manual Coulomb Counting
        dt = df['time_sec'].diff().fillna(0).abs().clip(upper=10.0)
        delta_ah = (df['Current'] * dt) / 3600.0
        df['SOC_cc'] = (soc_initial + delta_ah.cumsum() / q_actual).clip(0.0, 1.0)

    # --- Kinetic derivatives ---
    dt = df['time_sec'].diff().fillna(0).abs().clip(upper=10.0)
    safe_dt = dt.replace(0, np.nan)
    df['dV_dt'] = (df['Voltage'].diff() / safe_dt).fillna(0).clip(-2.0, 2.0)
    df['dI_dt'] = (df['Current'].diff() / safe_dt).fillna(0).clip(-20.0, 20.0)

    if 'Temperature' not in df.columns:
        df['Temperature'] = 25.0

    return df, soc_initial


# =====================================================================
# 4. Sequence Building
# =====================================================================
FEATURE_COLS = ['Voltage', 'Current', 'Temperature', 'dV_dt', 'dI_dt']


def build_sequences(df, window=100, stride=10):
    """Build sliding-window sequences from a DataFrame.

    Returns
    -------
    X : ndarray (N, window, 5)  — input feature windows
    y : ndarray (N, window)     — full SOC trajectory per window (Seq2Seq)

    Note: range(0, len-window+1, stride) ensures the terminal window is
    included, so the final low-SOC region is never dropped.
    """
    if not all(c in df.columns for c in FEATURE_COLS + ['SOC_cc']):
        return None, None

    data   = df[FEATURE_COLS].values.astype(np.float32)
    labels = df['SOC_cc'].values.astype(np.float32)

    X, y = [], []
    # Fix R4: +1 ensures the last valid window (ending at index len-1) is included
    for s in range(0, len(data) - window + 1, stride):
        X.append(data[s:s+window])
        y.append(labels[s:s+window])        # full sequence label (Seq2Seq)

    if not X:
        return None, None
    return np.array(X), np.array(y)


# =====================================================================
# 5. Main Pipeline v2 — Temperature-Calibrated
# =====================================================================
def run_pipeline_v2(scenario='A', window=100, stride=10):
    """
    Process raw CSVs into v2 datasets with temperature-calibrated SOC.

    Saves:
      - X_train.npy, y_train.npy, etc.  (scaled features + SOC targets)
      - temp_labels_test.npy             (temperature label per test sequence)
      - metadata_v2.json                 (Q_actual per temp, OCV stats)
    """
    print(f"{'='*65}")
    print(f"  PI-TCN Pipeline v2: Scenario {scenario}")
    print(f"  Temperature-Calibrated SOC + OCV-Anchor")
    print(f"{'='*65}")

    all_temps = ['0degC', '10degC', '25degC', '40degC', 'n10degC', 'n20degC']
    temp_data = {t: {'X': [], 'y': [], 'soc_initials': []} for t in all_temps}

    # --- Metadata for reproducibility ---
    metadata = {
        'version': 'v2',
        'method': 'OCV-anchor + Capacity-based SOC',
        'interpolation': 'PchipInterpolator (monotone cubic)',
        'q_actual': {},
        'ocv_points': {},
        'soc_initial_stats': {},
    }

    for temp in tqdm(all_temps, desc=f"Processing Temps (Scenario {scenario})"):
        temp_dir = os.path.join(DATA_RAW, temp)
        if not os.path.exists(temp_dir):
            print(f"  [{temp}] Missing directory, skipping.")
            continue

        # Build OCV-SOC lookup for this temperature
        ocv_lookup, q_actual = build_ocv_soc_lookup(temp)
        metadata['q_actual'][temp] = round(q_actual, 4)
        metadata['ocv_points'][temp] = 'PCHIP' if ocv_lookup is not None else 'fallback'

        print(f"  [{temp}] Q_actual={q_actual:.4f} Ah, "
              f"OCV={'PCHIP' if ocv_lookup else 'None'}...")

        csvs = glob.glob(os.path.join(temp_dir, '*.csv'))
        soc_initials = []

        for c in tqdm(sorted(csvs), desc=f"CSVs for {temp}", leave=False):
            fname = os.path.basename(c).lower()
            if not any(k in fname for k in ['udds', 'la92', 'hwfet', 'us06', 'mixed']):
                continue

            try:
                df = read_csv(c)
                df, soc_init = engineer_features(df, q_actual, ocv_lookup)
                soc_initials.append(soc_init)

                X, y = build_sequences(df, window, stride)
                if X is not None:
                    temp_data[temp]['X'].append(X)
                    temp_data[temp]['y'].append(y)
            except Exception:
                pass

        # Stack sequences for this temperature
        if temp_data[temp]['X']:
            temp_data[temp]['X'] = np.concatenate(temp_data[temp]['X'], axis=0)
            temp_data[temp]['y'] = np.concatenate(temp_data[temp]['y'], axis=0)
            n_seq = len(temp_data[temp]['X'])
        else:
            temp_data[temp]['X'] = np.array([])
            temp_data[temp]['y'] = np.array([])
            n_seq = 0

        if soc_initials:
            metadata['soc_initial_stats'][temp] = {
                'mean': round(float(np.mean(soc_initials)), 4),
                'min': round(float(np.min(soc_initials)), 4),
                'max': round(float(np.max(soc_initials)), 4),
                'count': len(soc_initials),
            }
        print(f"    -> {n_seq} sequences, SOC_init: {soc_initials[:3]}")

    # ── Splitting Logic ──────────────────────────────────────────────
    print(f"\n--- Splitting Logic (Scenario {scenario}) ---")

    if scenario == 'A':
        # Train: {25°C, 10°C}  — chronological 90/10 split per temp
        # Val  : last 10% of {25°C, 10°C} sequences (chronological holdout)
        #         + ALL of {0°C} sequences (different drive cycle, no overlap)
        # Test : {40°C, -10°C, -20°C}  — fully held-out OOD temperatures
        #
        # Fix F1: Split BEFORE shuffling, using the last 10% of each temp's
        # time-ordered sequences. This prevents overlapping windows from
        # appearing in both train and val.
        train_temps = ['25degC', '10degC']
        test_temps  = ['40degC', 'n10degC', 'n20degC']

        train_X_list, train_y_list = [], []
        val_X_list,   val_y_list   = [], []

        for t in train_temps:
            X_t = temp_data[t]['X']   # (N_t, window, 5) — already in time order
            y_t = temp_data[t]['y']   # (N_t, window)
            if len(X_t) == 0:
                continue
            n_val_t = max(1, int(0.10 * len(X_t)))   # last 10% → val
            train_X_list.append(X_t[:-n_val_t])
            train_y_list.append(y_t[:-n_val_t])
            val_X_list.append(X_t[-n_val_t:])
            val_y_list.append(y_t[-n_val_t:])
            print(f"  [{t}] train={len(X_t)-n_val_t:,}  val={n_val_t:,} (chron holdout)")

        # Augment val with the entire 0°C split (no overlap with train temps)
        X_val_0 = temp_data['0degC']['X']
        y_val_0 = temp_data['0degC']['y']
        if len(X_val_0) > 0:
            val_X_list.append(X_val_0)
            val_y_list.append(y_val_0)
            print(f"  [0degC]  val={len(X_val_0):,} (dedicated val temperature)")

        X_train = np.concatenate(train_X_list) if train_X_list else np.empty((0, window, 5), dtype=np.float32)
        y_train = np.concatenate(train_y_list) if train_y_list else np.empty((0, window), dtype=np.float32)
        del train_X_list, train_y_list
        
        X_val   = np.concatenate(val_X_list)   if val_X_list   else np.empty((0, window, 5), dtype=np.float32)
        y_val   = np.concatenate(val_y_list)   if val_y_list   else np.empty((0, window), dtype=np.float32)
        del val_X_list, val_y_list
        
        X_test  = _concat_temps(temp_data, test_temps, 'X', window)
        y_test  = _concat_temps(temp_data, test_temps, 'y')
        gc.collect()

        # Build temperature labels for test set (per-temp RMSE breakdown)
        temp_labels_test = _build_temp_labels(temp_data, test_temps)

        # Shuffle ONLY train (AFTER the chronological split)
        rng = np.random.default_rng(42)
        if len(X_train) > 0:
            idx = rng.permutation(len(X_train))
            X_train, y_train = X_train[idx], y_train[idx]

    elif scenario == 'B':
        # Chronological per-temp split: first 70% train | next 10% val | last 20% test
        # Fix F1 already satisfied here: split is strictly by time index, not shuffle
        X_train_list, y_train_list = [], []
        X_val_list, y_val_list = [], []
        X_test_list, y_test_list = [], []
        test_label_list = []

        for temp in all_temps:
            X_tmp = temp_data[temp]['X']
            y_tmp = temp_data[temp]['y']
            if len(X_tmp) == 0:
                continue

            n_tr = int(0.7 * len(X_tmp))
            n_vl = int(0.1 * len(X_tmp))

            X_train_list.append(X_tmp[:n_tr])
            y_train_list.append(y_tmp[:n_tr])
            X_val_list.append(X_tmp[n_tr:n_tr+n_vl])
            y_val_list.append(y_tmp[n_tr:n_tr+n_vl])
            X_test_list.append(X_tmp[n_tr+n_vl:])
            y_test_list.append(y_tmp[n_tr+n_vl:])
            test_label_list.extend([temp] * len(X_tmp[n_tr+n_vl:]))

        X_train = np.concatenate(X_train_list) if X_train_list else np.empty((0, window, 5), dtype=np.float32)
        y_train = np.concatenate(y_train_list) if y_train_list else np.empty((0, window), dtype=np.float32)
        del X_train_list, y_train_list
        X_val   = np.concatenate(X_val_list)   if X_val_list   else np.empty((0, window, 5), dtype=np.float32)
        y_val   = np.concatenate(y_val_list)   if y_val_list   else np.empty((0, window), dtype=np.float32)
        del X_val_list, y_val_list
        X_test  = np.concatenate(X_test_list)  if X_test_list  else np.empty((0, window, 5), dtype=np.float32)
        y_test  = np.concatenate(y_test_list)  if y_test_list  else np.empty((0, window), dtype=np.float32)
        del X_test_list, y_test_list
        temp_labels_test = np.array(test_label_list)
        gc.collect()

        # Shuffle ONLY train (AFTER the chronological split)
        rng = np.random.default_rng(42)
        if len(X_train) > 0:
            idx = rng.permutation(len(X_train))
            X_train, y_train = X_train[idx], y_train[idx]
    else:
        raise ValueError("Invalid scenario")

    # ── Shape Guards ─────────────────────────────────────────────────
    assert X_train.ndim == 3 and X_train.shape[2] == 5, f"Bad X: {X_train.shape}"
    assert y_train.ndim == 2 and y_train.shape[1] == window, f"Bad y: {y_train.shape} (expected seq2seq width={window})"

    # ── Physics-Informed Scaling ─────────────────────────────────────
    PHYS_MIN = np.array([2.0, -20.0, -20.0, -2.0, -20.0], dtype=np.float32)
    PHYS_MAX = np.array([4.25, 20.0,  50.0,  2.0,  20.0], dtype=np.float32)
    PHYS_RANGE = PHYS_MAX - PHYS_MIN

    p_min = PHYS_MIN.reshape(1, 1, 5)
    p_rng = PHYS_RANGE.reshape(1, 1, 5)

    # Clear temp_data to save memory since we don't need it anymore
    del temp_data
    gc.collect()

    X_train -= p_min
    X_train /= p_rng
    if len(X_val)  > 0:
        X_val  -= p_min
        X_val  /= p_rng
    if len(X_test) > 0:
        X_test -= p_min
        X_test /= p_rng

    print(f"\n  Physics-Informed Scaling applied:")
    print(f"    Bounds: V[2.0,4.25] I[-20,20] T[-20,50] dV[-2,2] dI[-20,20]")
    print(f"    Train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    if len(X_val)  > 0: print(f"    Val   range: [{X_val.min():.4f}, {X_val.max():.4f}]")
    if len(X_test) > 0: print(f"    Test  range: [{X_test.min():.4f}, {X_test.max():.4f}]")

    # ── SOC Distribution Summary ────────────────────────────────────
    print(f"\n  SOC Distribution:")
    print(f"    Train: mean={y_train.mean():.4f}, std={y_train.std():.4f}, "
          f"[{y_train.min():.4f}, {y_train.max():.4f}]")
    if len(y_val)  > 0:
        print(f"    Val  : mean={y_val.mean():.4f}, std={y_val.std():.4f}, "
              f"[{y_val.min():.4f}, {y_val.max():.4f}]")
    if len(y_test) > 0:
        print(f"    Test : mean={y_test.mean():.4f}, std={y_test.std():.4f}, "
              f"[{y_test.min():.4f}, {y_test.max():.4f}]")

    # ── Save to disk ────────────────────────────────────────────────
    out_dir = os.path.join(DATA_PROC, f'scenario_{scenario}')
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "X_val.npy"),   X_val)
    np.save(os.path.join(out_dir, "y_val.npy"),   y_val)
    np.save(os.path.join(out_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(out_dir, "y_test.npy"),  y_test)
    np.save(os.path.join(out_dir, "X_min.npy"),   p_min)
    np.save(os.path.join(out_dir, "X_max.npy"),   p_min + p_rng)

    # Save temperature labels for per-temp evaluation
    np.save(os.path.join(out_dir, "temp_labels_test.npy"), temp_labels_test)

    # Save metadata
    metadata['shapes'] = {
        'X_train': list(X_train.shape),
        'y_train': list(y_train.shape),
        'X_val':   list(X_val.shape),
        'y_val':   list(y_val.shape),
        'X_test':  list(X_test.shape),
        'y_test':  list(y_test.shape),
    }
    with open(os.path.join(out_dir, "metadata_v2.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  X_train : {X_train.shape}")
    print(f"  y_train : {y_train.shape}")
    print(f"  X_val   : {X_val.shape}")
    print(f"  y_val   : {y_val.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_test  : {y_test.shape}")
    print(f"  temp_labels_test: {len(temp_labels_test)} labels")
    print(f"  Metadata: {os.path.join(out_dir, 'metadata_v2.json')}")
    print(f"  Saved to: {out_dir}\n")


# ── Helper functions ─────────────────────────────────────────────────
def _concat_temps(temp_data, temps, key, window=100):
    """Concatenate data from multiple temperature groups."""
    arrays = [temp_data[t][key] for t in temps if len(temp_data[t][key]) > 0]
    if arrays:
        return np.concatenate(arrays, axis=0)
    if key == 'X':
        return np.empty((0, window, 5), dtype=np.float32)
    return np.empty((0, window), dtype=np.float32)


def _build_temp_labels(temp_data, temps):
    """Build array of temperature labels for each sequence."""
    labels = []
    for t in temps:
        if len(temp_data[t]['X']) > 0:
            labels.extend([t] * len(temp_data[t]['X']))
    return np.array(labels)


# =====================================================================
# Entry point
# =====================================================================
if __name__ == "__main__":
    run_pipeline_v2('A')
    run_pipeline_v2('B')
