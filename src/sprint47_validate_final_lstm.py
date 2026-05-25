"""
sprint47_validate_final_lstm.py -- Final Scenario B validation for V6 LSTM
===========================================================================

Final decision script for Reviewer 3:
  - Candidate: outputs/models/best_model_v6_alpha_1.0.pt
  - Architecture: HardCoulombLSTM
  - Data: Scenario B chronological holdout test set

Evaluation only. No training and no files are written.

Usage:
    python src/sprint47_validate_final_lstm.py
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (  # noqa: E402
    BATCH_SIZE,
    CURRENT_THRESHOLD,
    DATA_PROC,
    NUM_INPUTS,
    OUTPUT_MOD,
)
from model_v5_coulomb import HardCoulombLSTM  # noqa: E402


CHECKPOINT_NAME = "best_model_v6_alpha_1.0.pt"
SCENARIO = "scenario_B"
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.2
DEFAULT_SAFETY_FACTOR = 1.5


def log(message: str = "") -> None:
    print(message, flush=True)


def iter_batches(n_rows: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n_rows, batch_size):
        yield start, min(start + batch_size, n_rows)


def safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def r2_from_sums(sse: float, sum_y: float, sum_y2: float, n_points: int) -> float:
    if n_points <= 0:
        return 0.0
    sst = sum_y2 - (sum_y * sum_y / n_points)
    if sst <= 0.0:
        return 0.0
    return float(1.0 - sse / sst)


def load_model(checkpoint_path: str, device: torch.device) -> HardCoulombLSTM:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = safe_torch_load(checkpoint_path, device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        cfg = checkpoint.get("config", {})
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        cfg = {}
        state_dict = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    model = HardCoulombLSTM(
        num_inputs=cfg.get("num_inputs", NUM_INPUTS),
        hidden_size=cfg.get("hidden_size", DEFAULT_HIDDEN_SIZE),
        num_layers=cfg.get("num_layers", DEFAULT_NUM_LAYERS),
        dropout=cfg.get("dropout", DEFAULT_DROPOUT),
        safety_factor=cfg.get("safety_factor", DEFAULT_SAFETY_FACTOR),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    log(f"  Loaded checkpoint: {checkpoint_path}")
    if isinstance(checkpoint, dict):
        log(f"  Epoch: {checkpoint.get('epoch', '?')} | alpha={checkpoint.get('alpha', 1.0)}")
        log(f"  Config: {cfg}")
    log(f"  Gamma={model.hard_constraint.gamma:.6e} | "
        f"safety_factor={model.hard_constraint.safety_factor}")
    return model


def load_scenario_b_test():
    data_dir = os.path.join(DATA_PROC, f"v3_{SCENARIO}")
    x_path = os.path.join(data_dir, "X_test.npy")
    y_path = os.path.join(data_dir, "y_test.npy")
    i_path = os.path.join(data_dir, "I_unscaled_test.npy")

    missing = [path for path in [x_path, y_path, i_path] if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing Scenario B test arrays:\n  " + "\n  ".join(missing))

    x_test = np.load(x_path, mmap_mode="r")
    y_test = np.load(y_path, mmap_mode="r")
    i_test = np.load(i_path, mmap_mode="r")
    return data_dir, x_test, y_test, i_test


def evaluate_scenario_b(model: HardCoulombLSTM, batch_size: int, device: torch.device) -> dict:
    data_dir, x_test, y_test, i_test = load_scenario_b_test()
    n_rows, seq_len = y_test.shape

    log(f"\n{'=' * 78}")
    log("  FINAL V6 LSTM SCENARIO B EVALUATION")
    log(f"{'=' * 78}")
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
        x_b = torch.from_numpy(np.asarray(x_test[start:end], dtype=np.float32).copy()).to(device)
        i_b = torch.from_numpy(np.asarray(i_test[start:end], dtype=np.float32).copy()).to(device)

        with torch.inference_mode():
            y_pred = model(x_b, i_b).detach().cpu().numpy().squeeze(-1)

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
        "rmse_pct": rmse * 100.0,
        "mae_pct": mae * 100.0,
        "maxe_pct": maxe * 100.0,
        "r2": r2,
        "pvr_pct": pvr,
        "pvr_violations": pvr_violations,
        "pvr_discharge_steps": pvr_discharge,
    }

    log("\n  Scenario B Metrics:")
    log(f"    RMSE : {metrics['rmse_pct']:.4f}%")
    log(f"    MAE  : {metrics['mae_pct']:.4f}%")
    log(f"    MaxE : {metrics['maxe_pct']:.4f}%")
    log(f"    R2   : {metrics['r2']:.6f}")
    log(f"    PVR  : {metrics['pvr_pct']:.4f}% "
        f"({pvr_violations:,} / {pvr_discharge:,} discharge steps)")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate best_model_v6_alpha_1.0.pt on Scenario B only."
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
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = os.path.abspath(args.checkpoint)

    log("=" * 78)
    log("  Sprint 47 Final Decision: V6 LSTM alpha=1.0 on Scenario B")
    log("=" * 78)
    log("  SAFETY: evaluation only; no training and no artifacts are written.")
    log(f"  Device: {device}")
    log(f"  Checkpoint: {checkpoint_path}")

    model = load_model(checkpoint_path, device)
    evaluate_scenario_b(model, args.batch_size, device)

    log(f"\n{'=' * 78}")
    log("  FINAL LSTM VALIDATION COMPLETE")
    log(f"{'=' * 78}")


if __name__ == "__main__":
    main()
