"""
preprocessing_v4.py -- Leakage-safe 1 Hz Hybrid Physics-ML pipeline
===================================================================

Phase 1 refactor of preprocessing_v3.py.

What changes:
  1. Raw profiles are collapsed to strict 1.0 Hz before feature engineering,
     splitting, or windowing.
  2. Train/Val/Test splits happen on continuous 1 Hz dataframes first.
  3. Sliding windows are created separately inside each split.
  4. Timestamp-key tensors are saved and checked for split leakage.

What does NOT change:
  - V_proxy = Voltage - Current * R_int
  - Temperature R_int mapping
  - SOC_cc, dV_proxy_dt, dI_dt feature formulas
  - Fixed physics scaling bounds
"""

import gc
import glob
import json
import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm

warnings.filterwarnings("ignore")
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LABEL_MODE, PHYS_MAX_V3, PHYS_MIN_V3, R_INT_PER_TEMP  # noqa: E402


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw", "LG Dataset", "LG_HG2_Original_Dataset")
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(DATA_PROC, exist_ok=True)

Q_NOMINAL = 3.0
WINDOW = 100
STRIDE = 10
PROFILE_KEY_SCALE = 10_000_000_000

Q_ACTUAL_PER_TEMP = {
    "25degC": 2.7744,
    "10degC": 2.6102,
    "0degC": 2.5764,
    "40degC": 2.7180,
    "n10degC": 2.4919,
    "n20degC": 2.3304,
}

FEATURE_COLS_V4 = ["V_proxy", "Current", "Temperature", "dV_proxy_dt", "dI_dt"]


# =====================================================================
# 1. CSV reading and unchanged feature engineering helpers
# =====================================================================
def find_header_row(filepath: str) -> int:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if "Voltage" in line and "Current" in line:
                return i
    raise ValueError(f"Header not found: {filepath}")


def read_csv(filepath: str) -> pd.DataFrame:
    hrow = find_header_row(filepath)
    df = pd.read_csv(
        filepath,
        skiprows=hrow,
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=False,
    )
    df.columns = df.columns.str.strip()

    for col in ["Voltage", "Current", "Temperature", "Capacity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Time Stamp" in df.columns:
        timestamp = pd.to_datetime(
            df["Time Stamp"],
            format="%m/%d/%Y %I:%M:%S %p",
            errors="coerce",
        )
        if timestamp.isna().all():
            timestamp = pd.to_datetime(df["Time Stamp"], errors="coerce", format="mixed")
        df["_timestamp"] = timestamp
        valid_ts = df["_timestamp"].notna()
        if valid_ts.any():
            t0 = df.loc[valid_ts, "_timestamp"].iloc[0]
            df["time_sec"] = (df["_timestamp"] - t0).dt.total_seconds()
            df["timestamp_ns"] = df["_timestamp"].astype("int64")
        else:
            df["time_sec"] = np.arange(len(df), dtype=np.float64) * 0.1
            df["timestamp_ns"] = (df["time_sec"].to_numpy() * 1_000_000_000).astype(np.int64)
    else:
        df["time_sec"] = np.arange(len(df), dtype=np.float64) * 0.1
        df["timestamp_ns"] = (df["time_sec"].to_numpy() * 1_000_000_000).astype(np.int64)

    return df.dropna(subset=["Voltage", "Current", "time_sec"]).reset_index(drop=True)


def build_ocv_soc_lookup(temp_name: str):
    temp_dir = os.path.join(DATA_RAW, temp_name)
    hppc_files = glob.glob(os.path.join(temp_dir, "*HPPC*.csv"))

    if not hppc_files:
        print(f"    [WARNING] No HPPC file for {temp_name}, using fallback")
        return None, Q_ACTUAL_PER_TEMP.get(temp_name, Q_NOMINAL)

    fpath = hppc_files[0]
    hrow = find_header_row(fpath)
    df = pd.read_csv(
        fpath,
        skiprows=hrow,
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=False,
    )
    df.columns = df.columns.str.strip()

    for col in ["Voltage", "Current", "Capacity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Time Stamp"] = pd.to_datetime(
        df["Time Stamp"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce",
    )
    if df["Time Stamp"].isna().all():
        df["Time Stamp"] = pd.to_datetime(df["Time Stamp"], errors="coerce", format="mixed")
    t0 = df["Time Stamp"].dropna().iloc[0]
    df["time_sec"] = (df["Time Stamp"] - t0).dt.total_seconds()

    q_actual = df["Capacity"].abs().max()

    is_rest = df["Current"].abs() < 0.01
    rest_groups = (is_rest != is_rest.shift()).cumsum()
    rest_segments = df[is_rest].groupby(rest_groups)

    ocv_list, soc_list = [], []
    for _, seg in rest_segments:
        duration = seg["time_sec"].max() - seg["time_sec"].min()
        if duration >= 600:
            ocv = seg["Voltage"].iloc[-1]
            cap_used = abs(seg["Capacity"].iloc[-1])
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


def engineer_features_v4(df: pd.DataFrame, q_actual: float, r_int: float, ocv_lookup=None,
                         label_mode: str | None = None):
    """
    Compute V_proxy and calibrated SOC ground truth.

    This is intentionally identical to v3 feature math. The only upstream
    difference is that df is already strict 1.0 Hz.

    label_mode (Phase 5 audit): "legacy" reproduces the published labels;
    "ohmic_corrected" removes the I*R_int drop from the segment-start voltage
    before the OCV lookup, reducing the loaded-start soc_initial bias.
    Defaults to config.LABEL_MODE.
    """
    if label_mode is None:
        label_mode = LABEL_MODE
    if label_mode not in ("legacy", "ohmic_corrected"):
        raise ValueError(f"Unknown label_mode: {label_mode}")
    df = df.copy()

    v_initial = df["Voltage"].iloc[0]
    if label_mode == "ohmic_corrected":
        v_initial = v_initial - df["Current"].iloc[0] * r_int
    if ocv_lookup is not None:
        soc_initial = float(np.clip(ocv_lookup(v_initial), 0.0, 1.0))
    else:
        soc_initial = 1.0

    if "Capacity" in df.columns and df["Capacity"].notna().sum() > 0:
        cap = df["Capacity"].fillna(0.0)
        cap_offset = cap.iloc[0]
        cap_used = (cap - cap_offset).abs()
        df["SOC_cc"] = (soc_initial - cap_used / q_actual).clip(0.0, 1.0)
    else:
        dt = df["time_sec"].diff().fillna(0).abs().clip(upper=10.0)
        delta_ah = (df["Current"] * dt) / 3600.0
        df["SOC_cc"] = (soc_initial + delta_ah.cumsum() / q_actual).clip(0.0, 1.0)

    df["V_proxy"] = df["Voltage"] - df["Current"] * r_int

    dt = df["time_sec"].diff().fillna(0).abs().clip(upper=10.0)
    safe_dt = dt.replace(0, np.nan)
    df["dV_proxy_dt"] = (df["V_proxy"].diff() / safe_dt).fillna(0).clip(-2.0, 2.0)
    df["dI_dt"] = (df["Current"].diff() / safe_dt).fillna(0).clip(-20.0, 20.0)

    if "Temperature" not in df.columns:
        df["Temperature"] = 25.0

    return df, soc_initial


# =====================================================================
# 2. Strict 1 Hz conversion before any split/window operation
# =====================================================================
def assert_strict_1hz(df: pd.DataFrame, context: str) -> None:
    if len(df) <= 1:
        return
    deltas = np.diff(df["time_sec"].to_numpy(dtype=np.int64))
    bad = deltas[deltas != 1]
    assert bad.size == 0, (
        f"{context}: non-1Hz deltas found after resampling. "
        f"Bad examples={bad[:10].tolist()}"
    )


def make_timestamp_keys(df: pd.DataFrame, profile_code: int) -> pd.Series:
    seconds = df["time_sec"].to_numpy(dtype=np.int64)
    return pd.Series(profile_code * PROFILE_KEY_SCALE + seconds, index=df.index, dtype="int64")


def to_strict_1hz_segments(
    df: pd.DataFrame,
    source_id: str,
    profile_code_start: int,
    min_len: int = WINDOW,
    decimation_mode: str = "first_sample",
) -> Tuple[List[pd.DataFrame], int, Dict[str, int]]:
    """
    Collapse each raw second to one row, then split at any remaining time
    gaps. Every returned segment is guaranteed strict 1 Hz.

    decimation_mode (v5 campaign):
      "first_sample"                 -- legacy: first raw sample per second.
      "mean_per_second"              -- V/I/T replaced by intra-second means
                                        (anti-aliased), Capacity by the last
                                        sample (it is an integral state).
      "integrated_current_per_second"-- only Current replaced by the intra-
                                        second mean (charge-preserving);
                                        V/T stay first-sample.
    All modes keep the first row's remaining columns, so downstream feature
    math and window logic are untouched.
    """
    if df.empty:
        return [], profile_code_start, {"raw_rows": 0, "dedup_rows": 0, "segments": 0, "short_segments": 0}
    if decimation_mode not in ("first_sample", "mean_per_second", "integrated_current_per_second"):
        raise ValueError(f"Unknown decimation_mode: {decimation_mode}")

    raw_rows = len(df)
    work = df.copy().sort_values("time_sec").reset_index(drop=True)
    work["_second"] = np.floor(work["time_sec"].to_numpy(dtype=np.float64) + 1e-9).astype(np.int64)

    if decimation_mode != "first_sample":
        grouped = work.groupby("_second", sort=True)
        agg_current = grouped["Current"].mean()
        agg_voltage = grouped["Voltage"].mean() if "Voltage" in work.columns else None
        agg_temp = grouped["Temperature"].mean() if "Temperature" in work.columns else None
        agg_cap = grouped["Capacity"].last() if "Capacity" in work.columns else None

    work = work.drop_duplicates("_second", keep="first").sort_values("_second").reset_index(drop=True)

    if decimation_mode != "first_sample":
        sec_index = work["_second"].to_numpy()
        work["Current"] = agg_current.loc[sec_index].to_numpy()
        if agg_cap is not None:
            work["Capacity"] = agg_cap.loc[sec_index].to_numpy()
        if decimation_mode == "mean_per_second":
            if agg_voltage is not None:
                work["Voltage"] = agg_voltage.loc[sec_index].to_numpy()
            if agg_temp is not None:
                work["Temperature"] = agg_temp.loc[sec_index].to_numpy()

    work["time_sec"] = work["_second"].astype(np.float64)

    gaps = work["_second"].diff().fillna(1).ne(1)
    segment_id = gaps.cumsum()

    segments: List[pd.DataFrame] = []
    profile_code = profile_code_start
    short_segments = 0
    for seg_idx, seg in work.groupby(segment_id, sort=True):
        seg = seg.copy().reset_index(drop=True)
        if len(seg) < min_len:
            short_segments += 1
            continue
        context = f"{source_id}#seg{int(seg_idx)}"
        assert_strict_1hz(seg, context)
        seg["profile_id"] = context
        seg["timestamp_key"] = make_timestamp_keys(seg, profile_code)
        profile_code += 1
        segments.append(seg)

    stats = {
        "raw_rows": int(raw_rows),
        "dedup_rows": int(len(work)),
        "segments": int(len(segments)),
        "short_segments": int(short_segments),
    }
    return segments, profile_code, stats


# =====================================================================
# 3. Split before windowing
# =====================================================================
def split_train_val(df: pd.DataFrame, train_ratio: float, window: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = int(train_ratio * n)
    if n >= 2 * window and n - cut < window:
        cut = n - window
    if n >= 2 * window and cut < window:
        cut = window
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def split_train_val_test(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    window: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut_train = int(train_ratio * n)
    cut_val = int((train_ratio + val_ratio) * n)

    if n >= 3 * window:
        if n - cut_val < window:
            cut_val = n - window
        if cut_val - cut_train < window:
            cut_train = max(window, cut_val - window)
        if cut_train < window:
            cut_train = window

    return (
        df.iloc[:cut_train].copy(),
        df.iloc[cut_train:cut_val].copy(),
        df.iloc[cut_val:].copy(),
    )


def build_windows_v4(df: pd.DataFrame, window: int = WINDOW, stride: int = STRIDE):
    required = FEATURE_COLS_V4 + ["SOC_cc", "timestamp_ns", "timestamp_key"]
    if len(df) < window or not all(c in df.columns for c in required):
        return None, None, None, None, None

    assert_strict_1hz(df, str(df["profile_id"].iloc[0]) if "profile_id" in df.columns else "window_source")

    data = df[FEATURE_COLS_V4].values.astype(np.float32)
    labels = df["SOC_cc"].values.astype(np.float32)
    current_raw = df["Current"].values.astype(np.float32)
    timestamps = df["timestamp_ns"].values.astype(np.int64)
    timestamp_keys = df["timestamp_key"].values.astype(np.int64)

    X, y, I_raw, T_raw, T_key = [], [], [], [], []
    for start in range(0, len(data) - window + 1, stride):
        end = start + window
        X.append(data[start:end])
        y.append(labels[start:end])
        I_raw.append(current_raw[start:end])
        T_raw.append(timestamps[start:end])
        T_key.append(timestamp_keys[start:end])

    if not X:
        return None, None, None, None, None
    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(I_raw, dtype=np.float32),
        np.array(T_raw, dtype=np.int64),
        np.array(T_key, dtype=np.int64),
    )


def empty_split_arrays(window: int = WINDOW):
    return {
        "X": np.empty((0, window, len(FEATURE_COLS_V4)), dtype=np.float32),
        "y": np.empty((0, window), dtype=np.float32),
        "I": np.empty((0, window), dtype=np.float32),
        "T": np.empty((0, window), dtype=np.int64),
        "K": np.empty((0, window), dtype=np.int64),
    }


def append_windowed(
    buckets: Dict[str, Dict[str, list]],
    split_name: str,
    df: pd.DataFrame,
    temp: str,
    test_labels: list,
    window: int,
    stride: int,
) -> int:
    X, y, I_raw, timestamps, timestamp_keys = build_windows_v4(df, window, stride)
    if X is None:
        return 0

    buckets[split_name]["X"].append(X)
    buckets[split_name]["y"].append(y)
    buckets[split_name]["I"].append(I_raw)
    buckets[split_name]["T"].append(timestamps)
    buckets[split_name]["K"].append(timestamp_keys)
    if split_name == "test":
        test_labels.extend([temp] * len(X))
    return int(len(X))


def concat_bucket(bucket: Dict[str, list], window: int = WINDOW) -> Dict[str, np.ndarray]:
    if not bucket["X"]:
        return empty_split_arrays(window)
    return {
        "X": np.concatenate(bucket["X"], axis=0),
        "y": np.concatenate(bucket["y"], axis=0),
        "I": np.concatenate(bucket["I"], axis=0),
        "T": np.concatenate(bucket["T"], axis=0),
        "K": np.concatenate(bucket["K"], axis=0),
    }


# =====================================================================
# 4. Verification: zero timestamp overlap between splits
# =====================================================================
def overlap_count(a: np.ndarray, b: np.ndarray) -> int:
    if a.size == 0 or b.size == 0:
        return 0
    a_unique = np.unique(a.reshape(-1))
    b_unique = np.unique(b.reshape(-1))
    return int(np.intersect1d(a_unique, b_unique, assume_unique=True).size)


def verify_no_split_leakage(splits: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, int]:
    overlaps = {
        "train_val": overlap_count(splits["train"]["T"], splits["val"]["T"]),
        "train_test": overlap_count(splits["train"]["T"], splits["test"]["T"]),
        "val_test": overlap_count(splits["val"]["T"], splits["test"]["T"]),
        "train_val_key": overlap_count(splits["train"]["K"], splits["val"]["K"]),
        "train_test_key": overlap_count(splits["train"]["K"], splits["test"]["K"]),
        "val_test_key": overlap_count(splits["val"]["K"], splits["test"]["K"]),
    }

    print("\n--- Timestamp Leakage Verification ---")
    print(f"  Train/Val overlap : {overlaps['train_val']:,} timesteps")
    print(f"  Train/Test overlap: {overlaps['train_test']:,} timesteps")
    print(f"  Val/Test overlap  : {overlaps['val_test']:,} timesteps")

    assert overlaps["train_test"] == 0, (
        "Temporal leakage detected: Train/Test timestamp intersection is not zero."
    )
    assert overlaps["train_val"] == 0, (
        "Temporal leakage detected: Train/Val timestamp intersection is not zero."
    )
    assert overlaps["val_test"] == 0, (
        "Temporal leakage detected: Val/Test timestamp intersection is not zero."
    )
    assert overlaps["train_test_key"] == 0, (
        "Temporal leakage detected: Train/Test timestamp-key intersection is not zero."
    )
    assert overlaps["train_val_key"] == 0, (
        "Temporal leakage detected: Train/Val timestamp-key intersection is not zero."
    )
    assert overlaps["val_test_key"] == 0, (
        "Temporal leakage detected: Val/Test timestamp-key intersection is not zero."
    )

    print("VERIFICATION PASSED: 0 Overlapping Timesteps between Splits.")
    return overlaps


# =====================================================================
# 5. Main pipeline v4
# =====================================================================
def run_pipeline_v4(
    scenario: str = "A",
    window: int = WINDOW,
    stride: int = STRIDE,
    label_mode: str | None = None,
    decimation_mode: str | None = None,
    variant_name: str = "v4",
) -> None:
    """
    variant_name controls the output directory (data/processed/<variant>_scenario_X).
    "v4" with default modes reproduces the legacy pipeline exactly; v5 variants
    change ONLY label_mode / decimation_mode (window, stride, features, splits
    unchanged).
    """
    if label_mode is None:
        label_mode = LABEL_MODE
    if decimation_mode is None:
        decimation_mode = "first_sample"
    print(f"{'=' * 72}")
    print(f"  Pipeline {variant_name}: 1Hz + Split-Before-Windowing -- Scenario {scenario}")
    print(f"  label_mode={label_mode} | decimation_mode={decimation_mode}")
    print("  Feature math locked: V_proxy = V_t - I*R_int")
    print(f"{'=' * 72}")

    all_temps = ["0degC", "10degC", "25degC", "40degC", "n10degC", "n20degC"]
    train_temps_a = {"25degC", "10degC"}
    val_temps_a = {"0degC"}
    test_temps_a = {"40degC", "n10degC", "n20degC"}

    buckets = {
        split: {"X": [], "y": [], "I": [], "T": [], "K": []}
        for split in ["train", "val", "test"]
    }
    temp_labels_test: List[str] = []
    profile_code = 1

    metadata = {
        "version": variant_name,
        "dataset_version": variant_name,
        "label_mode": label_mode,
        "decimation_mode": decimation_mode,
        "method": "V_proxy + strict 1Hz + split-before-windowing",
        "feature_cols": FEATURE_COLS_V4,
        "window": window,
        "stride": stride,
        "sampling_hz": 1.0,
        "split_before_windowing": True,
        "r_int_per_temp": {k: round(v, 5) for k, v in R_INT_PER_TEMP.items()},
        "q_actual": {},
        "ocv_points": {},
        "soc_initial_stats": {},
        "profile_stats": {},
        "split_window_counts": {
            "train": {},
            "val": {},
            "test": {},
        },
    }

    for temp in tqdm(all_temps, desc=f"Processing Temps (v4, Scenario {scenario})"):
        temp_dir = os.path.join(DATA_RAW, temp)
        if not os.path.exists(temp_dir):
            print(f"  [{temp}] Missing directory, skipping.")
            continue

        ocv_lookup, q_actual = build_ocv_soc_lookup(temp)
        r_int = R_INT_PER_TEMP.get(temp, 0.03)

        metadata["q_actual"][temp] = round(float(q_actual), 4)
        metadata["ocv_points"][temp] = "PCHIP" if ocv_lookup is not None else "fallback"
        metadata["profile_stats"][temp] = {
            "raw_rows": 0,
            "dedup_rows": 0,
            "segments": 0,
            "short_segments": 0,
            "files_processed": 0,
        }

        print(
            f"  [{temp}] Q={q_actual:.4f} Ah, R_int={r_int * 1000:.2f} mOhm, "
            f"OCV={'PCHIP' if ocv_lookup else 'None'}"
        )

        csvs = sorted(glob.glob(os.path.join(temp_dir, "*.csv")))
        soc_initials: List[float] = []
        temp_counts = {"train": 0, "val": 0, "test": 0}

        for csv_path in tqdm(csvs, desc=f"CSVs for {temp}", leave=False):
            fname = os.path.basename(csv_path).lower()
            if not any(k in fname for k in ["udds", "la92", "hwfet", "us06", "mixed"]):
                continue

            source_id = f"{temp}/{os.path.basename(csv_path)}"
            try:
                raw_df = read_csv(csv_path)
                segments, profile_code, stats = to_strict_1hz_segments(
                    raw_df,
                    source_id=source_id,
                    profile_code_start=profile_code,
                    min_len=window,
                    decimation_mode=decimation_mode,
                )

                metadata["profile_stats"][temp]["files_processed"] += 1
                for key, value in stats.items():
                    metadata["profile_stats"][temp][key] += int(value)

                for segment in segments:
                    engineered, soc_init = engineer_features_v4(
                        segment,
                        q_actual=q_actual,
                        r_int=r_int,
                        ocv_lookup=ocv_lookup,
                        label_mode=label_mode,
                    )
                    soc_initials.append(soc_init)

                    if scenario == "A":
                        if temp in train_temps_a:
                            train_df, val_df = split_train_val(engineered, train_ratio=0.90, window=window)
                            temp_counts["train"] += append_windowed(
                                buckets, "train", train_df, temp, temp_labels_test, window, stride
                            )
                            temp_counts["val"] += append_windowed(
                                buckets, "val", val_df, temp, temp_labels_test, window, stride
                            )
                        elif temp in val_temps_a:
                            temp_counts["val"] += append_windowed(
                                buckets, "val", engineered, temp, temp_labels_test, window, stride
                            )
                        elif temp in test_temps_a:
                            temp_counts["test"] += append_windowed(
                                buckets, "test", engineered, temp, temp_labels_test, window, stride
                            )
                    elif scenario == "B":
                        train_df, val_df, test_df = split_train_val_test(
                            engineered,
                            train_ratio=0.70,
                            val_ratio=0.10,
                            window=window,
                        )
                        temp_counts["train"] += append_windowed(
                            buckets, "train", train_df, temp, temp_labels_test, window, stride
                        )
                        temp_counts["val"] += append_windowed(
                            buckets, "val", val_df, temp, temp_labels_test, window, stride
                        )
                        temp_counts["test"] += append_windowed(
                            buckets, "test", test_df, temp, temp_labels_test, window, stride
                        )
                    else:
                        raise ValueError("Invalid scenario. Use 'A' or 'B'.")
            except Exception as exc:
                print(f"    [WARNING] Skipped {source_id}: {exc}")

        if soc_initials:
            metadata["soc_initial_stats"][temp] = {
                "mean": round(float(np.mean(soc_initials)), 4),
                "min": round(float(np.min(soc_initials)), 4),
                "max": round(float(np.max(soc_initials)), 4),
                "count": len(soc_initials),
            }

        for split_name, count in temp_counts.items():
            metadata["split_window_counts"][split_name][temp] = int(count)
        print(
            f"    -> windows train={temp_counts['train']:,} "
            f"val={temp_counts['val']:,} test={temp_counts['test']:,}"
        )

    splits = {
        "train": concat_bucket(buckets["train"], window),
        "val": concat_bucket(buckets["val"], window),
        "test": concat_bucket(buckets["test"], window),
    }
    del buckets
    gc.collect()

    X_train, y_train, I_train = splits["train"]["X"], splits["train"]["y"], splits["train"]["I"]
    X_val, y_val, I_val = splits["val"]["X"], splits["val"]["y"], splits["val"]["I"]
    X_test, y_test, I_test = splits["test"]["X"], splits["test"]["y"], splits["test"]["I"]

    assert X_train.ndim == 3 and X_train.shape[2] == len(FEATURE_COLS_V4), f"Bad X_train: {X_train.shape}"
    assert y_train.ndim == 2 and y_train.shape[1] == window, f"Bad y_train: {y_train.shape}"
    assert I_train.ndim == 2 and I_train.shape[1] == window, f"Bad I_train: {I_train.shape}"

    verification = verify_no_split_leakage(splits)

    rng = np.random.default_rng(42)
    if len(X_train) > 0:
        idx = rng.permutation(len(X_train))
        for key in ["X", "y", "I", "T", "K"]:
            splits["train"][key] = splits["train"][key][idx]
        X_train, y_train, I_train = splits["train"]["X"], splits["train"]["y"], splits["train"]["I"]

    phys_min = np.array(PHYS_MIN_V3, dtype=np.float32)
    phys_max = np.array(PHYS_MAX_V3, dtype=np.float32)
    phys_range = phys_max - phys_min
    p_min = phys_min.reshape(1, 1, len(FEATURE_COLS_V4))
    p_rng = phys_range.reshape(1, 1, len(FEATURE_COLS_V4))

    X_train -= p_min
    X_train /= p_rng
    if len(X_val) > 0:
        X_val -= p_min
        X_val /= p_rng
    if len(X_test) > 0:
        X_test -= p_min
        X_test /= p_rng

    print("\n  v4 Physics Scaling applied:")
    print("    Bounds: V_proxy[2.5,4.25] I[-20,20] T[-20,50] dVp[-2,2] dI[-20,20]")
    if len(X_train) > 0:
        print(f"    Train range: [{X_train.min():.4f}, {X_train.max():.4f}]")

    print("\n  SOC Distribution:")
    if len(y_train) > 0:
        print(f"    Train: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
    if len(y_val) > 0:
        print(f"    Val  : mean={y_val.mean():.4f}, std={y_val.std():.4f}")
    if len(y_test) > 0:
        print(f"    Test : mean={y_test.mean():.4f}, std={y_test.std():.4f}")

    out_dir = os.path.join(DATA_PROC, f"{variant_name}_scenario_{scenario}")
    os.makedirs(out_dir, exist_ok=True)

    save_map = {
        "train": (X_train, y_train, I_train, splits["train"]["T"], splits["train"]["K"]),
        "val": (X_val, y_val, I_val, splits["val"]["T"], splits["val"]["K"]),
        "test": (X_test, y_test, I_test, splits["test"]["T"], splits["test"]["K"]),
    }
    for split_name, arrs in save_map.items():
        np.save(os.path.join(out_dir, f"X_{split_name}.npy"), arrs[0])
        np.save(os.path.join(out_dir, f"y_{split_name}.npy"), arrs[1])
        np.save(os.path.join(out_dir, f"I_unscaled_{split_name}.npy"), arrs[2])
        np.save(os.path.join(out_dir, f"timestamp_ns_{split_name}.npy"), arrs[3])
        np.save(os.path.join(out_dir, f"timestamp_key_{split_name}.npy"), arrs[4])

    np.save(os.path.join(out_dir, "X_min.npy"), p_min)
    np.save(os.path.join(out_dir, "X_max.npy"), p_min + p_rng)
    np.save(os.path.join(out_dir, "temp_labels_test.npy"), np.array(temp_labels_test))

    metadata["shapes"] = {
        "X_train": list(X_train.shape),
        "y_train": list(y_train.shape),
        "I_train": list(I_train.shape),
        "timestamp_train": list(splits["train"]["T"].shape),
        "X_val": list(X_val.shape),
        "y_val": list(y_val.shape),
        "I_val": list(I_val.shape),
        "timestamp_val": list(splits["val"]["T"].shape),
        "X_test": list(X_test.shape),
        "y_test": list(y_test.shape),
        "I_test": list(I_test.shape),
        "timestamp_test": list(splits["test"]["T"].shape),
    }
    metadata["verification"] = verification
    with open(os.path.join(out_dir, "metadata_v4.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  X_train : {X_train.shape}")
    print(f"  y_train : {y_train.shape}")
    print(f"  I_train : {I_train.shape}")
    print(f"  X_val   : {X_val.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  Saved to: {out_dir}\n")


if __name__ == "__main__":
    run_pipeline_v4("A")
    run_pipeline_v4("B")
