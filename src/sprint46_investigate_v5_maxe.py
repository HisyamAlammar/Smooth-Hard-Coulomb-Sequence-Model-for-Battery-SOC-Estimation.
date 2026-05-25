"""
sprint46_investigate_v5_maxe.py -- RCA for V5 Hard-Coulomb LSTM MaxE anomaly
==============================================================================

Reviewer 3 forensic script. This script is intentionally read-only: it loads the
V5 HardCoulombLSTM checkpoint and Scenario A Full-OOD test arrays, evaluates the
entire test set, finds the worst MaxE sequences, and prints diagnostics only.

Usage:
    python src/sprint46_investigate_v5_maxe.py
    python src/sprint46_investigate_v5_maxe.py --checkpoint outputs/models/best_model_v5_coulomb.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (  # noqa: E402
    BASE_DIR,
    BATCH_SIZE,
    CURRENT_THRESHOLD,
    DATA_PROC,
    NUM_INPUTS,
    OUTPUT_MOD,
    PHYS_MAX_V3,
    PHYS_MIN_V3,
)
from model_v5_coulomb import HardCoulombLSTM  # noqa: E402


DEFAULT_REQUESTED_CKPT = "best_model_v5_coulomb.pt"
SCENARIO = "scenario_A"
DATA_VERSION = "v3_scenario_A"
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
SAFETY_FACTOR = 1.5
EPS = 1e-10

PHYS_MIN = np.asarray(PHYS_MIN_V3, dtype=np.float32)
PHYS_MAX = np.asarray(PHYS_MAX_V3, dtype=np.float32)
PHYS_RNG = PHYS_MAX - PHYS_MIN


def log(message: str = "") -> None:
    print(message, flush=True)


def pct(value: float) -> float:
    return float(value) * 100.0


def unscale_features(x_scaled_seq: np.ndarray) -> np.ndarray:
    """Unscale one (T, 5) v3 feature sequence back to physical units."""
    return x_scaled_seq * PHYS_RNG.reshape(1, 5) + PHYS_MIN.reshape(1, 5)


def resolve_checkpoint(user_path: Optional[str]) -> str:
    """
    Resolve the requested V5 checkpoint without creating or overwriting files.

    Reviewer requested best_model_v5_coulomb.pt. The current Sprint 46 trainer
    stores Scenario A best weights as hybrid_v5_coulomb_scenario_A.pt, so this
    function first honors the requested name and only then falls back to the
    canonical project artifact.
    """
    if user_path:
        candidates = [user_path]
    else:
        candidates = [
            os.path.join(OUTPUT_MOD, DEFAULT_REQUESTED_CKPT),
            os.path.join(BASE_DIR, DEFAULT_REQUESTED_CKPT),
            DEFAULT_REQUESTED_CKPT,
            os.path.join(OUTPUT_MOD, "hybrid_v5_coulomb_scenario_A.pt"),
        ]

    checked = []
    for candidate in candidates:
        path = os.path.abspath(candidate)
        checked.append(path)
        if os.path.exists(path):
            requested_names = {
                os.path.abspath(os.path.join(OUTPUT_MOD, DEFAULT_REQUESTED_CKPT)),
                os.path.abspath(os.path.join(BASE_DIR, DEFAULT_REQUESTED_CKPT)),
                os.path.abspath(DEFAULT_REQUESTED_CKPT),
            }
            if not user_path and path not in requested_names:
                log("  NOTE: best_model_v5_coulomb.pt was not found.")
                log(f"        Falling back to Scenario A V5 best checkpoint: {path}")
            return path

    raise FileNotFoundError(
        "No V5 Coulomb checkpoint found. Checked:\n  " + "\n  ".join(checked)
    )


def safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[HardCoulombLSTM, Dict]:
    ckpt = safe_torch_load(checkpoint_path, device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        cfg = ckpt.get("config", {})
        state_dict = ckpt["model_state_dict"]
        epoch = ckpt.get("epoch", "?")
        val_loss = ckpt.get("val_loss", None)
    elif isinstance(ckpt, dict):
        cfg = {}
        state_dict = ckpt
        epoch = "state_dict_only"
        val_loss = None
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)!r}")

    model = HardCoulombLSTM(
        num_inputs=cfg.get("num_inputs", NUM_INPUTS),
        hidden_size=cfg.get("hidden_size", HIDDEN_SIZE),
        num_layers=cfg.get("num_layers", NUM_LAYERS),
        dropout=cfg.get("dropout", DROPOUT),
        safety_factor=cfg.get("safety_factor", SAFETY_FACTOR),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    metadata = {
        "config": cfg,
        "epoch": epoch,
        "val_loss": val_loss,
        "checkpoint_path": checkpoint_path,
    }
    return model, metadata


def load_test_arrays(data_dir: str):
    required = {
        "X": "X_test.npy",
        "y": "y_test.npy",
        "I": "I_unscaled_test.npy",
        "temp": "temp_labels_test.npy",
    }
    paths = {name: os.path.join(data_dir, filename) for name, filename in required.items()}
    missing = [path for path in paths.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing Scenario A test artifacts:\n  " + "\n  ".join(missing))

    x_test = np.load(paths["X"], mmap_mode="r")
    y_test = np.load(paths["y"], mmap_mode="r")
    i_test = np.load(paths["I"], mmap_mode="r")
    temp_labels = np.load(paths["temp"], allow_pickle=True)
    return x_test, y_test, i_test, temp_labels


def iter_batches(n_rows: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n_rows, batch_size):
        yield start, min(start + batch_size, n_rows)


def predict_batch(
    model: HardCoulombLSTM,
    device: torch.device,
    x_np: np.ndarray,
    i_np: np.ndarray,
) -> np.ndarray:
    x_b = torch.from_numpy(np.asarray(x_np, dtype=np.float32).copy()).to(device)
    i_b = torch.from_numpy(np.asarray(i_np, dtype=np.float32).copy()).to(device)
    with torch.inference_mode():
        y_pred = model(x_b, i_b).squeeze(-1)
    return y_pred.detach().cpu().numpy()


def predict_one(
    model: HardCoulombLSTM,
    device: torch.device,
    x_seq: np.ndarray,
    i_seq: np.ndarray,
) -> np.ndarray:
    return predict_batch(model, device, x_seq[None, ...], i_seq[None, ...])[0]


def evaluate_full_test_set(
    model: HardCoulombLSTM,
    device: torch.device,
    x_test: np.ndarray,
    y_test: np.ndarray,
    i_test: np.ndarray,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    n_rows, seq_len = y_test.shape
    per_seq_maxe = np.empty(n_rows, dtype=np.float32)
    per_seq_worst_step = np.empty(n_rows, dtype=np.int16)
    pred_t0 = np.empty(n_rows, dtype=np.float32)
    pred_t99 = np.empty(n_rows, dtype=np.float32)

    total_abs_err = 0.0
    total_sq_err = 0.0
    total_points = 0

    n_batches = (n_rows + batch_size - 1) // batch_size
    log(f"  Evaluating full test set in {n_batches:,} batches ...")

    for batch_id, (start, end) in enumerate(iter_batches(n_rows, batch_size), start=1):
        y_pred = predict_batch(model, device, x_test[start:end], i_test[start:end])
        y_true = np.asarray(y_test[start:end], dtype=np.float32)
        abs_err = np.abs(y_pred - y_true)

        per_seq_maxe[start:end] = abs_err.max(axis=1)
        per_seq_worst_step[start:end] = abs_err.argmax(axis=1).astype(np.int16)
        pred_t0[start:end] = y_pred[:, 0]
        pred_t99[start:end] = y_pred[:, -1]

        total_abs_err += float(abs_err.sum(dtype=np.float64))
        total_sq_err += float(np.square(y_pred - y_true, dtype=np.float32).sum(dtype=np.float64))
        total_points += abs_err.size

        if batch_id == 1 or batch_id == n_batches or batch_id % 25 == 0:
            log(f"    Batch {batch_id:>4}/{n_batches:<4} | rows {start:,}-{end - 1:,}")

    return {
        "per_seq_maxe": per_seq_maxe,
        "per_seq_worst_step": per_seq_worst_step,
        "pred_t0": pred_t0,
        "pred_t99": pred_t99,
        "mae": np.array(total_abs_err / total_points, dtype=np.float64),
        "rmse": np.array(np.sqrt(total_sq_err / total_points), dtype=np.float64),
    }


def top_indices_desc(values: np.ndarray, k: int) -> np.ndarray:
    k = min(k, values.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    partial = np.argpartition(values, -k)[-k:]
    return partial[np.argsort(values[partial])[::-1]]


def dissect_single_sequence(
    model: HardCoulombLSTM,
    device: torch.device,
    x_seq: np.ndarray,
    y_seq: np.ndarray,
    i_seq: np.ndarray,
) -> Dict[str, np.ndarray | float | int]:
    x_tensor = torch.from_numpy(np.asarray(x_seq, dtype=np.float32).copy()).unsqueeze(0).to(device)
    i_tensor = torch.from_numpy(np.asarray(i_seq, dtype=np.float32).copy()).unsqueeze(0).to(device)

    with torch.inference_mode():
        h, _ = model.lstm(x_tensor)
        soc_anchor = model.anchor_head(h[:, 0, :])
        delta_raw = model.delta_head(h)
        soc_pred = model.hard_constraint(delta_raw, i_tensor, soc_anchor)

    anchor = float(soc_anchor.detach().cpu().item())
    delta_raw_np = delta_raw.detach().cpu().numpy().reshape(-1).astype(np.float64)
    pred_np = soc_pred.detach().cpu().numpy().reshape(-1).astype(np.float64)

    i_np = np.asarray(i_seq, dtype=np.float64)
    y_np = np.asarray(y_seq, dtype=np.float64)
    seq_len = y_np.shape[0]
    threshold = float(model.hard_constraint.threshold)
    gamma = float(model.hard_constraint.gamma)
    safety_factor = float(model.hard_constraint.safety_factor)

    coulomb_limit = np.abs(i_np) * gamma * safety_factor
    discharge = i_np < -threshold
    charge = i_np > threshold
    rest = ~(discharge | charge)

    delta_constrained = np.zeros_like(delta_raw_np)
    delta_constrained[discharge] = np.clip(
        delta_raw_np[discharge], -coulomb_limit[discharge], 0.0
    )
    delta_constrained[charge] = np.clip(
        delta_raw_np[charge], 0.0, coulomb_limit[charge]
    )

    lower_coulomb_clamp = discharge & (delta_raw_np < (-coulomb_limit - EPS))
    upper_coulomb_clamp = charge & (delta_raw_np > (coulomb_limit + EPS))
    zero_direction_clamp = (
        (discharge & (delta_raw_np > EPS))
        | (charge & (delta_raw_np < -EPS))
    )
    rest_forced_zero = rest

    cumulative = np.cumsum(delta_constrained)
    pre_clamp = anchor + cumulative
    soc_reconstructed = np.clip(pre_clamp, 0.0, 1.0)

    abs_err = np.abs(pred_np - y_np)
    worst_step = int(abs_err.argmax())
    x_unscaled = unscale_features(np.asarray(x_seq, dtype=np.float32))

    return {
        "anchor": anchor,
        "delta_raw": delta_raw_np,
        "delta_constrained": delta_constrained,
        "cumulative": cumulative,
        "pre_clamp": pre_clamp,
        "soc_reconstructed": soc_reconstructed,
        "pred": pred_np,
        "abs_err": abs_err,
        "worst_step": worst_step,
        "max_err": float(abs_err[worst_step]),
        "coulomb_limit": coulomb_limit,
        "lower_coulomb_clamp_count": int(lower_coulomb_clamp.sum()),
        "upper_coulomb_clamp_count": int(upper_coulomb_clamp.sum()),
        "zero_direction_clamp_count": int(zero_direction_clamp.sum()),
        "rest_forced_zero_count": int(rest_forced_zero.sum()),
        "soc_clamp_zero_count": int((pre_clamp < 0.0).sum()),
        "soc_clamp_one_count": int((pre_clamp > 1.0).sum()),
        "charge_count": int(charge.sum()),
        "discharge_count": int(discharge.sum()),
        "rest_count": int(rest.sum()),
        "gamma": gamma,
        "safety_factor": safety_factor,
        "threshold": threshold,
        "seq_len": int(seq_len),
        "v_proxy": x_unscaled[:, 0].astype(np.float64),
        "i_feature_unscaled": x_unscaled[:, 1].astype(np.float64),
        "temperature_feature": x_unscaled[:, 2].astype(np.float64),
        "reconstruction_max_abs_diff": float(np.max(np.abs(soc_reconstructed - pred_np))),
    }


def print_top10_report(
    model: HardCoulombLSTM,
    device: torch.device,
    x_test: np.ndarray,
    y_test: np.ndarray,
    i_test: np.ndarray,
    temp_labels: np.ndarray,
    per_seq_maxe: np.ndarray,
    per_seq_worst_step: np.ndarray,
    top10_idx: np.ndarray,
) -> None:
    log(f"\n{'=' * 88}")
    log("  TOP 10 WORST SEQUENCES BY PER-SEQUENCE MAX ERROR")
    log(f"{'=' * 88}")
    log(
        f"  {'Rank':<5} {'Index':<10} {'MaxE(%)':<10} {'Step':<5} {'Temp':<9} "
        f"{'True@Step':<10} {'Pred@Step':<10} {'True[0]':<9} {'Pred[0]':<9} "
        f"{'True[99]':<9} {'Pred[99]':<9}"
    )
    log(f"  {'-' * 112}")

    for rank, idx in enumerate(top10_idx, start=1):
        step = int(per_seq_worst_step[idx])
        pred_seq = predict_one(
            model,
            device,
            np.asarray(x_test[idx], dtype=np.float32),
            np.asarray(i_test[idx], dtype=np.float32),
        )
        true_seq = np.asarray(y_test[idx], dtype=np.float32)
        log(
            f"  {rank:<5} {idx:<10,} {pct(per_seq_maxe[idx]):<10.2f} {step:<5} "
            f"{str(temp_labels[idx]):<9} {pct(true_seq[step]):<10.2f} "
            f"{pct(pred_seq[step]):<10.2f} {pct(true_seq[0]):<9.2f} "
            f"{pct(pred_seq[0]):<9.2f} {pct(true_seq[-1]):<9.2f} {pct(pred_seq[-1]):<9.2f}"
        )


def print_forensic_report(
    seq_idx: int,
    temp_label: str,
    y_seq: np.ndarray,
    i_seq: np.ndarray,
    diagnostic: Dict[str, np.ndarray | float | int],
) -> None:
    y_np = np.asarray(y_seq, dtype=np.float64)
    i_np = np.asarray(i_seq, dtype=np.float64)
    pred = diagnostic["pred"]
    abs_err = diagnostic["abs_err"]
    delta_raw = diagnostic["delta_raw"]
    delta_constrained = diagnostic["delta_constrained"]
    cumulative = diagnostic["cumulative"]
    pre_clamp = diagnostic["pre_clamp"]
    coulomb_limit = diagnostic["coulomb_limit"]
    v_proxy = diagnostic["v_proxy"]
    i_feature_unscaled = diagnostic["i_feature_unscaled"]
    temperature_feature = diagnostic["temperature_feature"]

    seq_len = int(diagnostic["seq_len"])
    anchor = float(diagnostic["anchor"])
    gamma = float(diagnostic["gamma"])
    safety_factor = float(diagnostic["safety_factor"])
    threshold = float(diagnostic["threshold"])
    worst_step = int(diagnostic["worst_step"])
    max_err = float(diagnostic["max_err"])

    i_mean = float(i_np.mean())
    i_mean_abs = float(np.abs(i_np).mean())
    total_budget_sum_limits = float(coulomb_limit.sum())
    total_budget_from_abs_mean = i_mean_abs * seq_len * gamma * safety_factor
    total_budget_from_signed_mean = abs(i_mean) * seq_len * gamma * safety_factor
    is_minus_20 = str(temp_label) == "n20degC" or float(np.median(temperature_feature)) <= -19.0

    log(f"\n{'=' * 88}")
    log("  FORENSIC REPORT FOR ABSOLUTE WORST SEQUENCE (#1)")
    log(f"{'=' * 88}")
    log(f"  Sequence Index                         : {seq_idx:,}")
    log(f"  Temperature label                      : {temp_label}")
    log(f"  Temperature feature median             : {np.median(temperature_feature):.2f} degC")
    log(f"  Is it -20C again?                      : {'YES' if is_minus_20 else 'NO'}")
    log(f"  Worst timestep                         : {worst_step}")
    log(f"  Max absolute error                     : {pct(max_err):.4f}%")

    log("\n  --- SOC Endpoints ---")
    log(f"  True SOC at t=0                        : {pct(y_np[0]):.4f}%")
    log(f"  True SOC at t=99                       : {pct(y_np[-1]):.4f}%")
    log(f"  Pred SOC at t=0                        : {pct(pred[0]):.4f}%")
    log(f"  Pred SOC at t=99                       : {pct(pred[-1]):.4f}%")
    log(f"  True SOC delta (0->99)                 : {pct(y_np[-1] - y_np[0]):+.4f}%")
    log(f"  Pred SOC delta (0->99)                 : {pct(pred[-1] - pred[0]):+.4f}%")

    log("\n  --- Anchor Head ---")
    log(f"  Anchor prediction                      : {pct(anchor):.4f}%")
    log(f"  True SOC at t=0                        : {pct(y_np[0]):.4f}%")
    log(f"  Anchor Error                           : {pct(abs(anchor - y_np[0])):.4f}%")
    log("  Note                                  : Pred[t=0] includes first constrained delta.")

    log("\n  --- Hard-Coulomb Delta Accounting ---")
    log(f"  Current threshold                      : {threshold:.6f} A")
    log(f"  Gamma                                  : {gamma:.10e} SOC/A/step")
    log(f"  Safety factor                          : {safety_factor:.4f}")
    log(f"  Charge / discharge / rest steps        : {diagnostic['charge_count']} / "
        f"{diagnostic['discharge_count']} / {diagnostic['rest_count']}")
    log(f"  delta_raw mean / std                   : {delta_raw.mean():+.8f} / {delta_raw.std():.8f}")
    log(f"  delta_raw min / max                    : {delta_raw.min():+.8f} / {delta_raw.max():+.8f}")
    log(f"  Sum(delta_constrained) over 100 steps  : {delta_constrained.sum():+.8f} "
        f"({pct(delta_constrained.sum()):+.4f}%)")
    log(f"  Cumulative delta at t=99               : {cumulative[-1]:+.8f} "
        f"({pct(cumulative[-1]):+.4f}%)")
    log(f"  Needed true delta (t99 - t0)           : {y_np[-1] - y_np[0]:+.8f} "
        f"({pct(y_np[-1] - y_np[0]):+.4f}%)")

    log("\n  --- Coulomb Budget ---")
    log(f"  I_mean signed                          : {i_mean:+.6f} A")
    log(f"  I_mean absolute                        : {i_mean_abs:.6f} A")
    log(f"  I_min / I_max                          : {i_np.min():+.6f} / {i_np.max():+.6f} A")
    log(f"  Total Coulomb Budget sum(|I_t|)        : {total_budget_sum_limits:.8f} "
        f"({pct(total_budget_sum_limits):.4f}%)")
    log(f"  Total Budget from mean(|I|)            : {total_budget_from_abs_mean:.8f} "
        f"({pct(total_budget_from_abs_mean):.4f}%)")
    log(f"  Budget from |I_mean_signed|            : {total_budget_from_signed_mean:.8f} "
        f"({pct(total_budget_from_signed_mean):.4f}%)")

    log("\n  --- Clamp Counts ---")
    log(f"  Delta clamped to lower Coulomb bound   : "
        f"{diagnostic['lower_coulomb_clamp_count']} / {seq_len}")
    log(f"  Delta clamped to upper Coulomb bound   : "
        f"{diagnostic['upper_coulomb_clamp_count']} / {seq_len}")
    log(f"  Delta clamped/forced to 0.0            : "
        f"{int(diagnostic['zero_direction_clamp_count']) + int(diagnostic['rest_forced_zero_count'])} / {seq_len} "
        f"(direction={diagnostic['zero_direction_clamp_count']}, "
        f"rest={diagnostic['rest_forced_zero_count']})")
    log(f"  SOC steps clamped to 0.0               : {diagnostic['soc_clamp_zero_count']} / {seq_len}")
    log(f"  SOC steps clamped to 1.0               : {diagnostic['soc_clamp_one_count']} / {seq_len}")
    log(f"  Reconstructed-vs-model max abs diff    : "
        f"{diagnostic['reconstruction_max_abs_diff']:.12e}")

    log("\n  --- Physical Features at Endpoints ---")
    log(f"  V_proxy at t=0                         : {v_proxy[0]:.6f} V")
    log(f"  V_proxy at t=99                        : {v_proxy[-1]:.6f} V")
    log(f"  I_unscaled artifact at t=0             : {i_np[0]:+.6f} A")
    log(f"  I_unscaled artifact at t=99            : {i_np[-1]:+.6f} A")
    log(f"  I feature unscaled at t=0              : {i_feature_unscaled[0]:+.6f} A")
    log(f"  I feature unscaled at t=99             : {i_feature_unscaled[-1]:+.6f} A")
    log(f"  Temperature feature at t=0             : {temperature_feature[0]:.2f} degC")
    log(f"  Temperature feature at t=99            : {temperature_feature[-1]:.2f} degC")

    log("\n  --- Worst-Step Local Values ---")
    log(f"  t={worst_step:<3d} True SOC             : {pct(y_np[worst_step]):.4f}%")
    log(f"  t={worst_step:<3d} Pred SOC             : {pct(pred[worst_step]):.4f}%")
    log(f"  t={worst_step:<3d} Abs Error            : {pct(abs_err[worst_step]):.4f}%")
    log(f"  t={worst_step:<3d} Pre-clamp SOC        : {pct(pre_clamp[worst_step]):.4f}%")
    log(f"  t={worst_step:<3d} delta_raw            : {delta_raw[worst_step]:+.8f}")
    log(f"  t={worst_step:<3d} delta_constrained    : {delta_constrained[worst_step]:+.8f}")
    log(f"  t={worst_step:<3d} Coulomb limit        : {coulomb_limit[worst_step]:.8f}")


def print_top50_distribution(temp_labels: np.ndarray, top50_idx: np.ndarray) -> None:
    top50_temps = np.asarray(temp_labels[top50_idx]).astype(str)
    counts = Counter(top50_temps.tolist())

    log(f"\n{'=' * 88}")
    log("  TOP 50 WORST-SEQUENCE TEMPERATURE DISTRIBUTION")
    log(f"{'=' * 88}")
    for temp, count in counts.most_common():
        log(f"  {temp:<10}: {count:>2}/50 ({count / 50.0 * 100:5.1f}%)")

    if set(counts.keys()) == {"n20degC"}:
        log("  Verdict      : Top-50 MaxE failures are completely isolated to -20C.")
    elif "n20degC" in counts:
        log("  Verdict      : -20C is present, but failures are not fully isolated to -20C.")
    else:
        log("  Verdict      : Top-50 MaxE failures are not from -20C.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only RCA for V5 Hard-Coulomb LSTM Scenario A MaxE."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path. Defaults to best_model_v5_coulomb.pt, with Sprint 46 Scenario A fallback.",
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
    data_dir = os.path.join(DATA_PROC, DATA_VERSION)
    checkpoint_path = resolve_checkpoint(args.checkpoint)

    log("=" * 88)
    log("  Sprint 46 RCA -- V5 Hard-Coulomb LSTM MaxE Investigation")
    log("=" * 88)
    log("  SAFETY: read-only forensic script; no models, logs, or figures are written.")
    log(f"  Device      : {device}")
    log(f"  Checkpoint  : {checkpoint_path}")
    log(f"  Test data   : {data_dir}")

    model, model_meta = load_model(checkpoint_path, device)
    hard = model.hard_constraint
    log(f"  Loaded epoch: {model_meta['epoch']}")
    if model_meta["val_loss"] is not None:
        log(f"  Val loss    : {model_meta['val_loss']:.8f}")
    log(f"  Model config: {model_meta['config']}")
    log(f"  Coulomb gamma={hard.gamma:.10e}, safety_factor={hard.safety_factor:.4f}, "
        f"threshold={hard.threshold:.6f} A")

    x_test, y_test, i_test, temp_labels = load_test_arrays(data_dir)
    n_rows, seq_len = y_test.shape
    log(f"\n  Scenario A Full-OOD test set: N={n_rows:,}, T={seq_len}")
    log(f"  X={x_test.shape}, y={y_test.shape}, I={i_test.shape}")
    log(f"  Temperature labels: {dict(Counter(np.asarray(temp_labels).astype(str).tolist()))}")

    metrics = evaluate_full_test_set(model, device, x_test, y_test, i_test, args.batch_size)
    per_seq_maxe = metrics["per_seq_maxe"]
    per_seq_worst_step = metrics["per_seq_worst_step"]

    top10_idx = top_indices_desc(per_seq_maxe, 10)
    top50_idx = top_indices_desc(per_seq_maxe, 50)
    worst_idx = int(top10_idx[0])

    log(f"\n{'=' * 88}")
    log("  GLOBAL TEST METRICS")
    log(f"{'=' * 88}")
    log(f"  MAE over all timesteps                  : {pct(float(metrics['mae'])):.4f}%")
    log(f"  RMSE over all timesteps                 : {pct(float(metrics['rmse'])):.4f}%")
    log(f"  MaxE over all timesteps                 : {pct(float(per_seq_maxe[worst_idx])):.4f}%")
    log(f"  MaxE sequence index                     : {worst_idx:,}")
    log(f"  MaxE timestep                           : {int(per_seq_worst_step[worst_idx])}")
    log(f"  MaxE temperature                        : {temp_labels[worst_idx]}")

    print_top10_report(
        model,
        device,
        x_test,
        y_test,
        i_test,
        temp_labels,
        per_seq_maxe,
        per_seq_worst_step,
        top10_idx,
    )

    x_worst = np.asarray(x_test[worst_idx], dtype=np.float32)
    y_worst = np.asarray(y_test[worst_idx], dtype=np.float32)
    i_worst = np.asarray(i_test[worst_idx], dtype=np.float32)
    diagnostic = dissect_single_sequence(model, device, x_worst, y_worst, i_worst)

    print_forensic_report(
        seq_idx=worst_idx,
        temp_label=str(temp_labels[worst_idx]),
        y_seq=y_worst,
        i_seq=i_worst,
        diagnostic=diagnostic,
    )
    print_top50_distribution(temp_labels, top50_idx)

    log(f"\n{'=' * 88}")
    log("  RCA COMPLETE -- no artifacts were written or overwritten.")
    log(f"{'=' * 88}")


if __name__ == "__main__":
    main()
