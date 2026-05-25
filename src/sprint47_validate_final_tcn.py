"""
sprint47_validate_final_tcn.py -- Final validation for V6 Hard-Coulomb TCN
===========================================================================

Reviewer 3 dual-validation for the selected worst-case-safety checkpoint:
    outputs/models/best_model_v6_tcn_alpha_10.0.pt

This script is evaluation/forensic only. It does not train models and does not
write any artifacts.

Tasks
-----
1. Evaluate the alpha=10.0 V6 TCN checkpoint on Scenario B chronological holdout.
2. Run RCA on Scenario A Full-OOD MaxE and report the worst sequence details.

Usage
-----
python src/sprint47_validate_final_tcn.py
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (  # noqa: E402
    BATCH_SIZE,
    CURRENT_THRESHOLD,
    DATA_PROC,
    DILATION_RATES,
    DROPOUT,
    KERNEL_SIZE,
    NUM_FILTERS,
    NUM_INPUTS,
    OUTPUT_MOD,
    PHYS_MAX_V3,
    PHYS_MIN_V3,
)
from model_v5_coulomb_tcn import HardCoulombTCN  # noqa: E402


CHECKPOINT_NAME = "best_model_v6_tcn_alpha_10.0.pt"
ALPHA = 10.0
SAFETY_FACTOR = 1.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PHYS_MIN = np.asarray(PHYS_MIN_V3, dtype=np.float32)
PHYS_MAX = np.asarray(PHYS_MAX_V3, dtype=np.float32)
PHYS_RNG = PHYS_MAX - PHYS_MIN


def log(msg: str = "") -> None:
    print(msg, flush=True)


def pct(value: float) -> float:
    return float(value) * 100.0


def safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def iter_batches(n_rows: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n_rows, batch_size):
        yield start, min(start + batch_size, n_rows)


def r2_from_sums(sse: float, sum_y: float, sum_y2: float, n_points: int) -> float:
    if n_points <= 0:
        return 0.0
    sst = sum_y2 - (sum_y * sum_y / n_points)
    if sst <= 0.0:
        return 0.0
    return float(1.0 - sse / sst)


def unscale_features(x_scaled_seq: np.ndarray) -> np.ndarray:
    return x_scaled_seq * PHYS_RNG.reshape(1, 5) + PHYS_MIN.reshape(1, 5)


def load_model(checkpoint_path: str, device: torch.device) -> HardCoulombTCN:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = safe_torch_load(checkpoint_path, device)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    model = HardCoulombTCN(
        num_inputs=cfg.get("num_inputs", NUM_INPUTS),
        num_filters=cfg.get("num_filters", NUM_FILTERS),
        kernel_size=cfg.get("kernel_size", KERNEL_SIZE),
        dropout=cfg.get("dropout", DROPOUT),
        dilation_rates=cfg.get("dilation_rates", DILATION_RATES),
        safety_factor=cfg.get("safety_factor", SAFETY_FACTOR),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    log(f"  Loaded checkpoint: {checkpoint_path}")
    if isinstance(ckpt, dict):
        log(f"  Checkpoint epoch: {ckpt.get('epoch', '?')} | alpha={ckpt.get('alpha', ALPHA)}")
        log(f"  Config: {cfg}")
    log(f"  Model gamma={model.hard_constraint.gamma:.6e}, "
        f"safety_factor={model.hard_constraint.safety_factor}")
    return model


def load_test_arrays(scenario_name: str):
    data_dir = os.path.join(DATA_PROC, f"v3_{scenario_name}")
    x_path = os.path.join(data_dir, "X_test.npy")
    y_path = os.path.join(data_dir, "y_test.npy")
    i_path = os.path.join(data_dir, "I_unscaled_test.npy")
    temp_path = os.path.join(data_dir, "temp_labels_test.npy")

    missing = [p for p in [x_path, y_path, i_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing test arrays:\n  " + "\n  ".join(missing))

    x_test = np.load(x_path, mmap_mode="r")
    y_test = np.load(y_path, mmap_mode="r")
    i_test = np.load(i_path, mmap_mode="r")
    temp_labels = np.load(temp_path, allow_pickle=True) if os.path.exists(temp_path) else None
    return data_dir, x_test, y_test, i_test, temp_labels


def model_outputs(model: HardCoulombTCN, x_b: torch.Tensor, i_b: torch.Tensor):
    """Return y_pred, anchor_pred, and delta_raw from the V5 TCN internals."""
    h = model.tcn(x_b.transpose(1, 2)).transpose(1, 2)
    delta_raw = model.delta_head(h)
    anchor_pred = model.anchor_head(h[:, 0, :])
    y_pred = model.hard_constraint(delta_raw, i_b, anchor_pred)
    return y_pred, anchor_pred, delta_raw


def predict_numpy(model: HardCoulombTCN, x_np: np.ndarray, i_np: np.ndarray, device: torch.device):
    x_b = torch.from_numpy(np.asarray(x_np, dtype=np.float32).copy()).to(device)
    i_b = torch.from_numpy(np.asarray(i_np, dtype=np.float32).copy()).to(device)
    with torch.inference_mode():
        y_pred, anchor_pred, delta_raw = model_outputs(model, x_b, i_b)
    return (
        y_pred.detach().cpu().numpy().squeeze(-1),
        anchor_pred.detach().cpu().numpy().squeeze(-1),
        delta_raw.detach().cpu().numpy().squeeze(-1),
    )


def evaluate_scenario_b(model: HardCoulombTCN, batch_size: int, device: torch.device) -> Dict:
    data_dir, x_test, y_test, i_test, _ = load_test_arrays("scenario_B")
    n_rows, seq_len = y_test.shape

    log(f"\n{'=' * 84}")
    log("  TASK 1: SCENARIO B EVALUATION (Chronological Hold-Out)")
    log(f"{'=' * 84}")
    log(f"  Data: {data_dir}")
    log(f"  Test set: N={n_rows:,}, T={seq_len}")

    abs_sum = 0.0
    sse = 0.0
    maxe = 0.0
    sum_y = 0.0
    sum_y2 = 0.0
    n_points = 0
    pvr_violations = 0
    pvr_discharge = 0

    n_batches = (n_rows + batch_size - 1) // batch_size
    for start, end in tqdm(
        iter_batches(n_rows, batch_size),
        total=n_batches,
        desc="  Scenario B Eval",
        dynamic_ncols=True,
    ):
        y_pred, _, _ = predict_numpy(model, x_test[start:end], i_test[start:end], device)
        y_true = np.asarray(y_test[start:end], dtype=np.float32)
        currents = np.asarray(i_test[start:end], dtype=np.float32)
        err = y_pred - y_true
        abs_err = np.abs(err)

        abs_sum += float(abs_err.sum(dtype=np.float64))
        sse += float(np.square(err, dtype=np.float32).sum(dtype=np.float64))
        maxe = max(maxe, float(abs_err.max()))
        sum_y += float(y_true.sum(dtype=np.float64))
        sum_y2 += float(np.square(y_true, dtype=np.float32).sum(dtype=np.float64))
        n_points += y_true.size

        delta_soc = y_pred[:, 1:] - y_pred[:, :-1]
        discharge_mask = currents[:, 1:] < -CURRENT_THRESHOLD
        pvr_discharge += int(discharge_mask.sum())
        pvr_violations += int(((delta_soc > 0.0) & discharge_mask).sum())

    rmse = float(np.sqrt(sse / n_points))
    mae = float(abs_sum / n_points)
    r2 = r2_from_sums(sse, sum_y, sum_y2, n_points)
    pvr = (pvr_violations / pvr_discharge * 100.0) if pvr_discharge else 0.0

    metrics = {
        "scenario": "scenario_B",
        "rmse_pct": pct(rmse),
        "mae_pct": pct(mae),
        "maxe_pct": pct(maxe),
        "r2": r2,
        "pvr_pct": pvr,
        "pvr_violations": pvr_violations,
        "pvr_discharge_steps": pvr_discharge,
        "n_sequences": int(n_rows),
        "n_points": int(n_points),
    }

    log("\n  Scenario B Standard Metrics:")
    log(f"    RMSE : {metrics['rmse_pct']:.4f}%")
    log(f"    MAE  : {metrics['mae_pct']:.4f}%")
    log(f"    MaxE : {metrics['maxe_pct']:.4f}%")
    log(f"    R2   : {metrics['r2']:.6f}")
    log(f"    PVR  : {metrics['pvr_pct']:.4f}% "
        f"({pvr_violations:,} / {pvr_discharge:,} discharge steps)")
    return metrics


def find_scenario_a_worst(model: HardCoulombTCN, batch_size: int, device: torch.device):
    data_dir, x_test, y_test, i_test, temp_labels = load_test_arrays("scenario_A")
    n_rows, seq_len = y_test.shape

    if temp_labels is None:
        temp_labels = np.array(["UNKNOWN"] * n_rows)

    log(f"\n{'=' * 84}")
    log("  TASK 2: RCA ON SCENARIO A MAXE (Full OOD)")
    log(f"{'=' * 84}")
    log(f"  Data: {data_dir}")
    log(f"  Test set: N={n_rows:,}, T={seq_len}")

    per_seq_maxe = np.empty(n_rows, dtype=np.float32)
    per_seq_worst_step = np.empty(n_rows, dtype=np.int16)
    n_batches = (n_rows + batch_size - 1) // batch_size

    for start, end in tqdm(
        iter_batches(n_rows, batch_size),
        total=n_batches,
        desc="  Scenario A RCA Scan",
        dynamic_ncols=True,
    ):
        y_pred, _, _ = predict_numpy(model, x_test[start:end], i_test[start:end], device)
        y_true = np.asarray(y_test[start:end], dtype=np.float32)
        abs_err = np.abs(y_pred - y_true)
        per_seq_maxe[start:end] = abs_err.max(axis=1)
        per_seq_worst_step[start:end] = abs_err.argmax(axis=1).astype(np.int16)

    top50_count = min(50, n_rows)
    top50_idx = np.argpartition(per_seq_maxe, -top50_count)[-top50_count:]
    top50_idx = top50_idx[np.argsort(per_seq_maxe[top50_idx])[::-1]]
    worst_idx = int(top50_idx[0])
    worst_step = int(per_seq_worst_step[worst_idx])

    y_pred_worst, anchor_worst, _ = predict_numpy(
        model,
        np.asarray(x_test[worst_idx:worst_idx + 1], dtype=np.float32),
        np.asarray(i_test[worst_idx:worst_idx + 1], dtype=np.float32),
        device,
    )
    y_pred_seq = y_pred_worst[0]
    anchor_pred = float(anchor_worst[0])
    y_true_seq = np.asarray(y_test[worst_idx], dtype=np.float32)
    i_seq = np.asarray(i_test[worst_idx], dtype=np.float32)
    x_unscaled = unscale_features(np.asarray(x_test[worst_idx], dtype=np.float32))
    v_proxy = x_unscaled[:, 0]

    temp_label = str(temp_labels[worst_idx])
    anchor_error = abs(anchor_pred - float(y_true_seq[0]))
    maxe = float(per_seq_maxe[worst_idx])

    log("\n  Worst-Case Scenario A Forensic Report:")
    log(f"    Sequence Index                 : {worst_idx:,}")
    log(f"    Temperature label              : {temp_label}")
    log(f"    Worst timestep                 : {worst_step}")
    log(f"    Max absolute error             : {pct(maxe):.4f}%")
    log("\n    SOC endpoints:")
    log(f"      True SOC at t=0              : {pct(y_true_seq[0]):.4f}%")
    log(f"      True SOC at t=99             : {pct(y_true_seq[-1]):.4f}%")
    log(f"      Pred SOC at t=0              : {pct(y_pred_seq[0]):.4f}%")
    log(f"      Pred SOC at t=99             : {pct(y_pred_seq[-1]):.4f}%")
    log("\n    Anchor diagnostic:")
    log(f"      Anchor prediction            : {pct(anchor_pred):.4f}%")
    log(f"      Anchor Error                 : {pct(anchor_error):.4f}%")
    log("\n    Physical features at t=0:")
    log(f"      V_proxy at t=0               : {float(v_proxy[0]):.6f} V")
    log(f"      I_unscaled at t=0            : {float(i_seq[0]):+.6f} A")

    top50_temps = np.asarray(temp_labels[top50_idx]).astype(str)
    counts = Counter(top50_temps.tolist())
    log("\n  Temperature distribution of top 50 worst Scenario A sequences:")
    for temp, count in counts.most_common():
        log(f"    {temp:<10}: {count:>2}/50 ({count / 50.0 * 100:5.1f}%)")

    return {
        "scenario": "scenario_A",
        "worst_sequence_index": worst_idx,
        "temperature_label": temp_label,
        "worst_timestep": worst_step,
        "maxe_pct": pct(maxe),
        "anchor_error_pct": pct(anchor_error),
        "top50_temperature_distribution": dict(counts),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final dual-validation for best_model_v6_tcn_alpha_10.0.pt."
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(OUTPUT_MOD, CHECKPOINT_NAME),
        help=f"Checkpoint path. Default: outputs/models/{CHECKPOINT_NAME}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Inference batch size. Default: {BATCH_SIZE}",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force device, e.g. cpu or cuda. Default: cuda if available else cpu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else DEVICE)
    checkpoint_path = os.path.abspath(args.checkpoint)

    log("=" * 84)
    log("  Sprint 47 Final Validation: V6 Hard-Coulomb TCN alpha=10.0")
    log("=" * 84)
    log("  SAFETY: evaluation/forensic only; no training and no files are written.")
    log(f"  Device: {device}")
    log(f"  Checkpoint: {checkpoint_path}")

    model = load_model(checkpoint_path, device)
    evaluate_scenario_b(model, args.batch_size, device)
    find_scenario_a_worst(model, args.batch_size, device)

    log(f"\n{'=' * 84}")
    log("  FINAL TCN VALIDATION COMPLETE")
    log(f"{'=' * 84}")


if __name__ == "__main__":
    main()
