"""
preprocessing_v5_contextual.py -- Sprint 50 Contextual Anchor pipeline
=====================================================================

Creates a new, isolated v5_contextual branch without modifying v4/v5/v7
artifacts. Feature math for the sequence path remains locked to v4:

    X_seq = [V_proxy, Current, Temperature, dV_proxy_dt, dI_dt]

The only new information is a causal static anchor context A_anchor computed
inside each already-split continuous dataframe before windowing:

    - OCV/rest evidence strictly before t=0
    - 60-second rolling history strictly before t=0

Leakage rule:
    raw -> strict 1 Hz -> feature engineering -> split continuous dataframe
    -> contextual features computed separately inside train/val/test
    -> windowing separately inside train/val/test
"""

from __future__ import annotations

import gc
import glob
import json
import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PHYS_MAX_V3, PHYS_MIN_V3, R_INT_PER_TEMP  # noqa: E402
from preprocessing_v4 import (  # noqa: E402
    DATA_PROC,
    DATA_RAW,
    FEATURE_COLS_V4,
    PROFILE_KEY_SCALE,
    Q_ACTUAL_PER_TEMP,
    Q_NOMINAL,
    STRIDE,
    WINDOW,
    assert_strict_1hz,
    build_ocv_soc_lookup,
    engineer_features_v4,
    read_csv,
    split_train_val,
    split_train_val_test,
    to_strict_1hz_segments,
)


REST_I_THRESH = 0.05
REST_MIN_SEC = 30
REST_MAX_AGE_SEC = 3600
DV_STABLE_THRESH = 0.001
LOOKBACK_SEC = 60
MAX_ABS_CHARGE_THROUGHPUT_60S = 20.0 * LOOKBACK_SEC / (Q_NOMINAL * 3600.0)

ANCHOR_CTX_COLS = [
    "ctx_ocv_rest_valid",
    "ctx_ocv_rest_voltage",
    "ctx_ocv_rest_age_sec",
    "ctx_ocv_rest_delta_v",
    "ctx_hist_valid_60s",
    "ctx_I_mean_60s",
    "ctx_I_std_60s",
    "ctx_I_min_60s",
    "ctx_I_max_60s",
    "ctx_abs_I_mean_60s",
    "ctx_V_mean_60s",
    "ctx_V_std_60s",
    "ctx_T_mean_60s",
    "ctx_charge_throughput_60s",
]

OCV_CTX_INDICES = list(range(0, 4))
HISTORY_CTX_INDICES = list(range(4, len(ANCHOR_CTX_COLS)))
ANCHOR_CTX_INDEX = {name: idx for idx, name in enumerate(ANCHOR_CTX_COLS)}


def add_ocv_rest_features(split_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add strictly causal OCV/rest evidence inside one already-split dataframe.

    The shift(1) is mandatory: a rest sample at t=0 cannot be used to predict
    SOC at t=0. Only rest evidence from t < t0 is propagated.
    """
    df = split_df.sort_values("time_sec").copy().reset_index(drop=True)
    assert_strict_1hz(df, "v5_contextual_ocv_rest")

    is_rest = df["Current"].abs() <= REST_I_THRESH
    rest_group = is_rest.ne(is_rest.shift(fill_value=False)).cumsum()
    rest_duration = is_rest.groupby(rest_group).cumcount() + 1
    rest_duration = rest_duration.where(is_rest, 0)

    d_voltage = df["Voltage"].diff().fillna(0.0)
    stable_rest = is_rest & (rest_duration >= REST_MIN_SEC) & (d_voltage.abs() <= DV_STABLE_THRESH)

    last_rest_voltage = df["Voltage"].where(stable_rest).shift(1).ffill()
    last_rest_time = df["time_sec"].where(stable_rest).shift(1).ffill()
    last_rest_key = df["timestamp_key"].where(stable_rest).shift(1).ffill()

    rest_age = df["time_sec"] - last_rest_time
    rest_valid = last_rest_voltage.notna() & (rest_age <= REST_MAX_AGE_SEC)

    df["ctx_ocv_rest_valid"] = rest_valid.astype(np.float32)
    df["ctx_ocv_rest_voltage"] = last_rest_voltage.where(rest_valid, 3.7).astype(np.float32)
    df["ctx_ocv_rest_age_sec"] = rest_age.where(rest_valid, REST_MAX_AGE_SEC).astype(np.float32)
    df["ctx_ocv_rest_delta_v"] = (
        df["Voltage"] - df["ctx_ocv_rest_voltage"]
    ).where(rest_valid, 0.0).astype(np.float32)
    df["ctx_ocv_rest_key"] = last_rest_key.where(rest_valid, -1).fillna(-1).astype(np.int64)

    return df


def add_history_features(split_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling 60 s history from [t0-60, t0-1] only.

    The shifted dataframe excludes the current row. Because this function is
    called after splitting, rolling history cannot cross train/val/test cuts.
    """
    df = split_df.sort_values("time_sec").copy().reset_index(drop=True)
    assert_strict_1hz(df, "v5_contextual_history")

    past = df.shift(1)
    current_roll = past["Current"].rolling(window=LOOKBACK_SEC, min_periods=LOOKBACK_SEC)
    voltage_roll = past["Voltage"].rolling(window=LOOKBACK_SEC, min_periods=LOOKBACK_SEC)
    temp_roll = past["Temperature"].rolling(window=LOOKBACK_SEC, min_periods=LOOKBACK_SEC)

    df["ctx_hist_valid_60s"] = (current_roll.count() == LOOKBACK_SEC).astype(np.float32)
    df["ctx_I_mean_60s"] = current_roll.mean().fillna(0.0).astype(np.float32)
    df["ctx_I_std_60s"] = current_roll.std().fillna(0.0).astype(np.float32)
    df["ctx_I_min_60s"] = current_roll.min().fillna(0.0).astype(np.float32)
    df["ctx_I_max_60s"] = current_roll.max().fillna(0.0).astype(np.float32)
    df["ctx_abs_I_mean_60s"] = past["Current"].abs().rolling(
        window=LOOKBACK_SEC,
        min_periods=LOOKBACK_SEC,
    ).mean().fillna(0.0).astype(np.float32)

    df["ctx_V_mean_60s"] = voltage_roll.mean().fillna(df["Voltage"]).astype(np.float32)
    df["ctx_V_std_60s"] = voltage_roll.std().fillna(0.0).astype(np.float32)
    df["ctx_T_mean_60s"] = temp_roll.mean().fillna(df["Temperature"]).astype(np.float32)
    df["ctx_charge_throughput_60s"] = (
        current_roll.sum().fillna(0.0) / (Q_NOMINAL * 3600.0)
    ).astype(np.float32)

    return df


def scale_anchor_context(anchor_raw: np.ndarray) -> np.ndarray:
    anchor_scaled = anchor_raw.astype(np.float32, copy=True)

    def col(name: str) -> int:
        return ANCHOR_CTX_INDEX[name]

    anchor_scaled[:, col("ctx_ocv_rest_valid")] = np.clip(anchor_scaled[:, col("ctx_ocv_rest_valid")], 0.0, 1.0)
    anchor_scaled[:, col("ctx_ocv_rest_voltage")] = np.clip(
        (anchor_scaled[:, col("ctx_ocv_rest_voltage")] - 2.5) / (4.25 - 2.5),
        0.0,
        1.0,
    )
    anchor_scaled[:, col("ctx_ocv_rest_age_sec")] = np.clip(
        np.log1p(anchor_scaled[:, col("ctx_ocv_rest_age_sec")]) / np.log1p(REST_MAX_AGE_SEC),
        0.0,
        1.0,
    )
    anchor_scaled[:, col("ctx_ocv_rest_delta_v")] = np.clip(
        (anchor_scaled[:, col("ctx_ocv_rest_delta_v")] + 1.0) / 2.0,
        0.0,
        1.0,
    )

    anchor_scaled[:, col("ctx_hist_valid_60s")] = np.clip(anchor_scaled[:, col("ctx_hist_valid_60s")], 0.0, 1.0)
    for name in ["ctx_I_mean_60s", "ctx_I_min_60s", "ctx_I_max_60s"]:
        anchor_scaled[:, col(name)] = np.clip((anchor_scaled[:, col(name)] + 20.0) / 40.0, 0.0, 1.0)
    anchor_scaled[:, col("ctx_I_std_60s")] = np.clip(anchor_scaled[:, col("ctx_I_std_60s")] / 20.0, 0.0, 1.0)
    anchor_scaled[:, col("ctx_abs_I_mean_60s")] = np.clip(
        anchor_scaled[:, col("ctx_abs_I_mean_60s")] / 20.0,
        0.0,
        1.0,
    )
    anchor_scaled[:, col("ctx_V_mean_60s")] = np.clip(
        (anchor_scaled[:, col("ctx_V_mean_60s")] - 2.5) / (4.25 - 2.5),
        0.0,
        1.0,
    )
    anchor_scaled[:, col("ctx_V_std_60s")] = np.clip(anchor_scaled[:, col("ctx_V_std_60s")] / 1.75, 0.0, 1.0)
    anchor_scaled[:, col("ctx_T_mean_60s")] = np.clip(
        (anchor_scaled[:, col("ctx_T_mean_60s")] + 20.0) / 70.0,
        0.0,
        1.0,
    )
    anchor_scaled[:, col("ctx_charge_throughput_60s")] = np.clip(
        (anchor_scaled[:, col("ctx_charge_throughput_60s")] + MAX_ABS_CHARGE_THROUGHPUT_60S)
        / (2.0 * MAX_ABS_CHARGE_THROUGHPUT_60S),
        0.0,
        1.0,
    )

    return anchor_scaled.astype(np.float32)


def build_contextual_windows(
    split_df: pd.DataFrame,
    window: int = WINDOW,
    stride: int = STRIDE,
    require_full_history: bool = True,
):
    required = FEATURE_COLS_V4 + ["SOC_cc", "timestamp_ns", "timestamp_key", "Voltage", "Current", "Temperature"]
    if len(split_df) < window or not all(column in split_df.columns for column in required):
        return None, None, None, None, None, None, None, None

    df = add_ocv_rest_features(split_df)
    df = add_history_features(df)
    assert_strict_1hz(df, str(df["profile_id"].iloc[0]) if "profile_id" in df.columns else "v5_window_source")

    sequence_data = df[FEATURE_COLS_V4].values.astype(np.float32)
    labels = df["SOC_cc"].values.astype(np.float32)
    current_raw = df["Current"].values.astype(np.float32)
    timestamps = df["timestamp_ns"].values.astype(np.int64)
    timestamp_keys = df["timestamp_key"].values.astype(np.int64)
    anchor_raw = df[ANCHOR_CTX_COLS].values.astype(np.float32)
    rest_keys = df["ctx_ocv_rest_key"].values.astype(np.int64)
    history_valid = df["ctx_hist_valid_60s"].values.astype(np.float32)

    X_seq, y_seq, I_raw, T_raw, T_key = [], [], [], [], []
    A_raw, C_key, R_key = [], [], []

    for start in range(0, len(sequence_data) - window + 1, stride):
        end = start + window
        if require_full_history and (start < LOOKBACK_SEC or history_valid[start] < 1.0):
            continue

        history_start = start - LOOKBACK_SEC
        context_keys = timestamp_keys[history_start:start]
        if require_full_history and len(context_keys) != LOOKBACK_SEC:
            continue

        X_seq.append(sequence_data[start:end])
        y_seq.append(labels[start:end])
        I_raw.append(current_raw[start:end])
        T_raw.append(timestamps[start:end])
        T_key.append(timestamp_keys[start:end])
        A_raw.append(anchor_raw[start])
        C_key.append(context_keys.astype(np.int64))
        R_key.append(np.array([rest_keys[start]], dtype=np.int64))

    if not X_seq:
        return None, None, None, None, None, None, None, None

    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y_seq, dtype=np.float32),
        np.array(I_raw, dtype=np.float32),
        np.array(A_raw, dtype=np.float32),
        np.array(T_raw, dtype=np.int64),
        np.array(T_key, dtype=np.int64),
        np.array(C_key, dtype=np.int64),
        np.array(R_key, dtype=np.int64),
    )


def empty_contextual_split(window: int = WINDOW) -> Dict[str, np.ndarray]:
    return {
        "X": np.empty((0, window, len(FEATURE_COLS_V4)), dtype=np.float32),
        "y": np.empty((0, window), dtype=np.float32),
        "I": np.empty((0, window), dtype=np.float32),
        "A": np.empty((0, len(ANCHOR_CTX_COLS)), dtype=np.float32),
        "T": np.empty((0, window), dtype=np.int64),
        "K": np.empty((0, window), dtype=np.int64),
        "C": np.empty((0, LOOKBACK_SEC), dtype=np.int64),
        "R": np.empty((0, 1), dtype=np.int64),
    }


def append_contextual_windowed(
    buckets: Dict[str, Dict[str, list]],
    split_name: str,
    df: pd.DataFrame,
    temp: str,
    test_labels: list,
    window: int,
    stride: int,
    require_full_history: bool,
) -> int:
    X_seq, y_seq, I_raw, A_raw, timestamps, sequence_keys, context_keys, rest_keys = build_contextual_windows(
        df,
        window=window,
        stride=stride,
        require_full_history=require_full_history,
    )
    if X_seq is None:
        return 0

    buckets[split_name]["X"].append(X_seq)
    buckets[split_name]["y"].append(y_seq)
    buckets[split_name]["I"].append(I_raw)
    buckets[split_name]["A"].append(A_raw)
    buckets[split_name]["T"].append(timestamps)
    buckets[split_name]["K"].append(sequence_keys)
    buckets[split_name]["C"].append(context_keys)
    buckets[split_name]["R"].append(rest_keys)
    if split_name == "test":
        test_labels.extend([temp] * len(X_seq))
    return int(len(X_seq))


def concat_contextual_bucket(bucket: Dict[str, list], window: int = WINDOW) -> Dict[str, np.ndarray]:
    if not bucket["X"]:
        return empty_contextual_split(window)
    return {
        "X": np.concatenate(bucket["X"], axis=0),
        "y": np.concatenate(bucket["y"], axis=0),
        "I": np.concatenate(bucket["I"], axis=0),
        "A": np.concatenate(bucket["A"], axis=0),
        "T": np.concatenate(bucket["T"], axis=0),
        "K": np.concatenate(bucket["K"], axis=0),
        "C": np.concatenate(bucket["C"], axis=0),
        "R": np.concatenate(bucket["R"], axis=0),
    }


def used_timestamp_keys(split: Dict[str, np.ndarray]) -> np.ndarray:
    arrays = []
    for key in ["K", "C", "R"]:
        if split[key].size:
            values = split[key].reshape(-1)
            arrays.append(values[values >= 0])
    if not arrays:
        return np.empty((0,), dtype=np.int64)
    return np.unique(np.concatenate(arrays).astype(np.int64))


def overlap_count(a: np.ndarray, b: np.ndarray) -> int:
    if a.size == 0 or b.size == 0:
        return 0
    return int(np.intersect1d(a, b, assume_unique=True).size)


def verify_contextual_no_leakage(splits: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, int]:
    used = {split_name: used_timestamp_keys(split) for split_name, split in splits.items()}
    overlaps = {
        "train_val_used_key": overlap_count(used["train"], used["val"]),
        "train_test_used_key": overlap_count(used["train"], used["test"]),
        "val_test_used_key": overlap_count(used["val"], used["test"]),
    }

    print("\n--- Contextual Timestamp Leakage Verification ---")
    print(f"  Train/Val used-key overlap : {overlaps['train_val_used_key']:,}")
    print(f"  Train/Test used-key overlap: {overlaps['train_test_used_key']:,}")
    print(f"  Val/Test used-key overlap  : {overlaps['val_test_used_key']:,}")

    assert overlaps["train_val_used_key"] == 0, "Train/Val contextual timestamp leakage detected."
    assert overlaps["train_test_used_key"] == 0, "Train/Test contextual timestamp leakage detected."
    assert overlaps["val_test_used_key"] == 0, "Val/Test contextual timestamp leakage detected."
    print("VERIFICATION PASSED: 0 Overlapping Sequence/Context Timesteps between Splits.")
    return overlaps


def run_pipeline_v5_contextual(
    scenario: str = "A",
    window: int = WINDOW,
    stride: int = STRIDE,
    require_full_history: bool = True,
) -> None:
    print(f"{'=' * 84}")
    print(f"  Pipeline v5_contextual -- Scenario {scenario}")
    print("  Sequence math locked to v4; anchor context is causal and split-local.")
    print(f"{'=' * 84}")

    all_temps = ["0degC", "10degC", "25degC", "40degC", "n10degC", "n20degC"]
    train_temps_a = {"25degC", "10degC"}
    val_temps_a = {"0degC"}
    test_temps_a = {"40degC", "n10degC", "n20degC"}

    buckets = {
        split_name: {"X": [], "y": [], "I": [], "A": [], "T": [], "K": [], "C": [], "R": []}
        for split_name in ["train", "val", "test"]
    }
    temp_labels_test: List[str] = []
    profile_code = 1

    metadata = {
        "version": "v5_contextual",
        "method": "v4 sequence features + causal OCV-rest/history anchor context",
        "feature_cols": FEATURE_COLS_V4,
        "anchor_ctx_cols": ANCHOR_CTX_COLS,
        "ocv_ctx_indices": OCV_CTX_INDICES,
        "history_ctx_indices": HISTORY_CTX_INDICES,
        "window": window,
        "stride": stride,
        "lookback_sec": LOOKBACK_SEC,
        "require_full_history": require_full_history,
        "sampling_hz": 1.0,
        "split_before_windowing": True,
        "context_computed_after_split": True,
        "r_int_per_temp": {key: round(value, 5) for key, value in R_INT_PER_TEMP.items()},
        "q_actual": {},
        "profile_stats": {},
        "split_window_counts": {"train": {}, "val": {}, "test": {}},
    }

    for temp in tqdm(all_temps, desc=f"Processing Temps (v5_contextual, Scenario {scenario})"):
        temp_dir = os.path.join(DATA_RAW, temp)
        if not os.path.exists(temp_dir):
            print(f"  [{temp}] Missing directory, skipping.")
            continue

        ocv_lookup, q_actual = build_ocv_soc_lookup(temp)
        r_int = R_INT_PER_TEMP.get(temp, 0.03)
        metadata["q_actual"][temp] = round(float(q_actual), 4)
        metadata["profile_stats"][temp] = {
            "raw_rows": 0,
            "dedup_rows": 0,
            "segments": 0,
            "short_segments": 0,
            "files_processed": 0,
        }

        print(f"  [{temp}] Q={q_actual:.4f} Ah, R_int={r_int * 1000:.2f} mOhm")
        csv_paths = sorted(glob.glob(os.path.join(temp_dir, "*.csv")))
        temp_counts = {"train": 0, "val": 0, "test": 0}

        for csv_path in tqdm(csv_paths, desc=f"CSVs for {temp}", leave=False):
            file_name = os.path.basename(csv_path).lower()
            if not any(kind in file_name for kind in ["udds", "la92", "hwfet", "us06", "mixed"]):
                continue

            source_id = f"{temp}/{os.path.basename(csv_path)}"
            try:
                raw_df = read_csv(csv_path)
                segments, profile_code, stats = to_strict_1hz_segments(
                    raw_df,
                    source_id=source_id,
                    profile_code_start=profile_code,
                    min_len=window + LOOKBACK_SEC if require_full_history else window,
                )

                metadata["profile_stats"][temp]["files_processed"] += 1
                for key, value in stats.items():
                    metadata["profile_stats"][temp][key] += int(value)

                for segment in segments:
                    engineered, _soc_init = engineer_features_v4(
                        segment,
                        q_actual=q_actual,
                        r_int=r_int,
                        ocv_lookup=ocv_lookup,
                    )

                    if scenario == "A":
                        if temp in train_temps_a:
                            train_df, val_df = split_train_val(engineered, train_ratio=0.90, window=window + LOOKBACK_SEC)
                            temp_counts["train"] += append_contextual_windowed(
                                buckets, "train", train_df, temp, temp_labels_test, window, stride, require_full_history
                            )
                            temp_counts["val"] += append_contextual_windowed(
                                buckets, "val", val_df, temp, temp_labels_test, window, stride, require_full_history
                            )
                        elif temp in val_temps_a:
                            temp_counts["val"] += append_contextual_windowed(
                                buckets, "val", engineered, temp, temp_labels_test, window, stride, require_full_history
                            )
                        elif temp in test_temps_a:
                            temp_counts["test"] += append_contextual_windowed(
                                buckets, "test", engineered, temp, temp_labels_test, window, stride, require_full_history
                            )
                    elif scenario == "B":
                        train_df, val_df, test_df = split_train_val_test(
                            engineered,
                            train_ratio=0.70,
                            val_ratio=0.10,
                            window=window + LOOKBACK_SEC,
                        )
                        temp_counts["train"] += append_contextual_windowed(
                            buckets, "train", train_df, temp, temp_labels_test, window, stride, require_full_history
                        )
                        temp_counts["val"] += append_contextual_windowed(
                            buckets, "val", val_df, temp, temp_labels_test, window, stride, require_full_history
                        )
                        temp_counts["test"] += append_contextual_windowed(
                            buckets, "test", test_df, temp, temp_labels_test, window, stride, require_full_history
                        )
                    else:
                        raise ValueError("Invalid scenario. Use 'A' or 'B'.")
            except Exception as exc:
                print(f"    [WARNING] Skipped {source_id}: {exc}")

        for split_name, count in temp_counts.items():
            metadata["split_window_counts"][split_name][temp] = int(count)
        print(
            f"    -> windows train={temp_counts['train']:,} "
            f"val={temp_counts['val']:,} test={temp_counts['test']:,}"
        )

    splits = {
        "train": concat_contextual_bucket(buckets["train"], window),
        "val": concat_contextual_bucket(buckets["val"], window),
        "test": concat_contextual_bucket(buckets["test"], window),
    }
    del buckets
    gc.collect()

    verification = verify_contextual_no_leakage(splits)

    rng = np.random.default_rng(42)
    if len(splits["train"]["X"]) > 0:
        permutation = rng.permutation(len(splits["train"]["X"]))
        for key in ["X", "y", "I", "A", "T", "K", "C", "R"]:
            splits["train"][key] = splits["train"][key][permutation]

    phys_min = np.array(PHYS_MIN_V3, dtype=np.float32)
    phys_max = np.array(PHYS_MAX_V3, dtype=np.float32)
    phys_range = phys_max - phys_min
    sequence_min = phys_min.reshape(1, 1, len(FEATURE_COLS_V4))
    sequence_range = phys_range.reshape(1, 1, len(FEATURE_COLS_V4))

    for split_name in ["train", "val", "test"]:
        if len(splits[split_name]["X"]) > 0:
            splits[split_name]["A_raw"] = splits[split_name]["A"].copy()
            splits[split_name]["A"] = scale_anchor_context(splits[split_name]["A"])
            splits[split_name]["X"] = (splits[split_name]["X"] - sequence_min) / sequence_range

    out_dir = os.path.join(DATA_PROC, "v5_contextual", f"scenario_{scenario}")
    os.makedirs(out_dir, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        np.save(os.path.join(out_dir, f"X_{split_name}.npy"), splits[split_name]["X"].astype(np.float32))
        np.save(os.path.join(out_dir, f"y_{split_name}.npy"), splits[split_name]["y"].astype(np.float32))
        np.save(os.path.join(out_dir, f"I_unscaled_{split_name}.npy"), splits[split_name]["I"].astype(np.float32))
        np.save(os.path.join(out_dir, f"A_anchor_{split_name}.npy"), splits[split_name]["A"].astype(np.float32))
        np.save(os.path.join(out_dir, f"A_anchor_raw_{split_name}.npy"), splits[split_name]["A_raw"].astype(np.float32))
        np.save(os.path.join(out_dir, f"timestamp_ns_{split_name}.npy"), splits[split_name]["T"].astype(np.int64))
        np.save(os.path.join(out_dir, f"timestamp_key_{split_name}.npy"), splits[split_name]["K"].astype(np.int64))
        np.save(os.path.join(out_dir, f"context_key_{split_name}.npy"), splits[split_name]["C"].astype(np.int64))
        np.save(os.path.join(out_dir, f"rest_key_{split_name}.npy"), splits[split_name]["R"].astype(np.int64))

    np.save(os.path.join(out_dir, "X_min.npy"), sequence_min)
    np.save(os.path.join(out_dir, "X_max.npy"), sequence_min + sequence_range)
    np.save(os.path.join(out_dir, "temp_labels_test.npy"), np.array(temp_labels_test))

    metadata["shapes"] = {
        f"{array_name}_{split_name}": list(splits[split_name][array_name].shape)
        for split_name in ["train", "val", "test"]
        for array_name in ["X", "y", "I", "A", "T", "K", "C", "R"]
    }
    metadata["verification"] = verification

    with open(os.path.join(out_dir, "metadata_v5_contextual.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\n  v5_contextual outputs:")
    for split_name in ["train", "val", "test"]:
        print(
            f"    {split_name}: X={splits[split_name]['X'].shape}, "
            f"A={splits[split_name]['A'].shape}, y={splits[split_name]['y'].shape}"
        )
    print(f"  Saved to: {out_dir}\n")


if __name__ == "__main__":
    run_pipeline_v5_contextual("A")
    run_pipeline_v5_contextual("B")
