"""
sprint48_safety_ablation.py -- Anchor trap and safety-factor analysis.

Evaluation-only script for the final v7 HardCoulombLSTM checkpoints.

Outputs:
  1. Anchor/t=0 absolute error on Scenario A and Scenario B test sets.
  2. Scenario B safety-factor eta ablation for eta in [1.0, 1.5, 2.0, 3.0].

The model architecture is not rewritten. The eta ablation only overrides
model.hard_constraint.safety_factor before inference.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from tqdm import tqdm

from sprint48_common import (
    BASE_DIR,
    BATCH_SIZE,
    OUTPUT_DIR,
    checkpoint_path,
    configure_utf8_stdio,
    load_checkpoint,
    load_test_split,
    make_loader,
    resolve_device,
)

SCENARIO_DATA_DIRS = {
    "scenario_A": BASE_DIR / "data" / "processed" / "v4_scenario_A",
    "scenario_B": BASE_DIR / "data" / "processed" / "v4_scenario_B",
}

DEFAULT_ETAS = (1.0, 1.5, 2.0, 3.0)
DISCHARGE_THRESHOLD_A = -0.05


@dataclass(frozen=True)
class PredictionBundle:
    y_true: np.ndarray
    y_pred: np.ndarray
    I_unscaled: np.ndarray
    anchor_raw: np.ndarray | None


def compute_pvr(y_pred: np.ndarray, I_unscaled: np.ndarray) -> Dict[str, float | int]:
    """
    Sequence-to-sequence PVR using the explicit paper condition:
    discharge iff I_unscaled < -0.05 A.
    """
    if y_pred.ndim != 2 or I_unscaled.ndim != 2:
        raise ValueError(f"PVR expects 2D arrays, got y={y_pred.shape}, I={I_unscaled.shape}")
    if y_pred.shape != I_unscaled.shape:
        raise ValueError(f"PVR shape mismatch: y={y_pred.shape}, I={I_unscaled.shape}")

    delta_soc = y_pred[:, 1:] - y_pred[:, :-1]
    discharge_mask = I_unscaled[:, 1:] < DISCHARGE_THRESHOLD_A
    violations = (delta_soc > 0.0) & discharge_mask
    discharge_steps = int(discharge_mask.sum())
    violation_count = int(violations.sum())
    pvr_pct = 0.0 if discharge_steps == 0 else (violation_count / discharge_steps) * 100.0
    return {
        "pvr_pct": float(pvr_pct),
        "violations": violation_count,
        "discharge_steps": discharge_steps,
    }


def error_summary_pct(abs_error_fraction: np.ndarray) -> Dict[str, float]:
    values_pct = abs_error_fraction.astype(np.float64) * 100.0
    return {
        "mae_pct": float(np.mean(values_pct)),
        "rmse_pct": float(np.sqrt(np.mean(values_pct**2))),
        "median_pct": float(np.percentile(values_pct, 50)),
        "p90_pct": float(np.percentile(values_pct, 90)),
        "p95_pct": float(np.percentile(values_pct, 95)),
        "p99_pct": float(np.percentile(values_pct, 99)),
        "maxe_pct": float(np.max(values_pct)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, I_unscaled: np.ndarray) -> Dict[str, float | int]:
    errors = y_pred - y_true
    pvr = compute_pvr(y_pred, I_unscaled)
    return {
        "rmse_pct": float(np.sqrt(np.mean(errors**2)) * 100.0),
        "mae_pct": float(np.mean(np.abs(errors)) * 100.0),
        "maxe_pct": float(np.max(np.abs(errors)) * 100.0),
        "pvr_pct": pvr["pvr_pct"],
        "pvr_violations": pvr["violations"],
        "pvr_discharge_steps": pvr["discharge_steps"],
    }


def set_safety_factor(model: torch.nn.Module, eta: float) -> None:
    if not hasattr(model, "hard_constraint"):
        raise AttributeError("Expected HardCoulombLSTM with hard_constraint attribute.")
    model.hard_constraint.safety_factor = float(eta)


def raw_anchor_prediction(model: torch.nn.Module, X_batch: torch.Tensor) -> torch.Tensor | None:
    """
    Return anchor_head(h[:, 0, :]) if the finalized model exposes the expected
    HardCoulombLSTM internals. This is diagnostic only; it does not modify the
    forward path used for reported metrics.
    """
    if not hasattr(model, "lstm") or not hasattr(model, "anchor_head"):
        return None
    hidden, _ = model.lstm(X_batch)
    return model.anchor_head(hidden[:, 0, :]).squeeze(-1)


def predict_hard_coulomb(
    scenario_key: str,
    eta: float | None,
    batch_size: int,
    device: torch.device,
    collect_anchor_raw: bool,
) -> PredictionBundle:
    data_dir = SCENARIO_DATA_DIRS[scenario_key]
    checkpoint = checkpoint_path("hard_coulomb_lstm", scenario_key, latest=False)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing finalized HardCoulombLSTM checkpoint: {checkpoint}")

    test_split = load_test_split(data_dir, scenario_key)
    loader = make_loader(
        test_split.X_test,
        test_split.y_test,
        test_split.I_test,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        device=device,
    )

    model, checkpoint_payload = load_checkpoint(checkpoint, device)
    if checkpoint_payload.get("model_kind") != "hard_coulomb_lstm":
        raise ValueError(f"Checkpoint is not HardCoulombLSTM: {checkpoint}")
    if eta is not None:
        set_safety_factor(model, eta)

    model.eval()
    predictions: List[np.ndarray] = []
    anchors: List[np.ndarray] = []
    desc_eta = f" eta={eta:g}" if eta is not None else ""
    with torch.no_grad():
        for X_batch, _y_batch, I_batch in tqdm(
            loader,
            desc=f"Predict {scenario_key}{desc_eta}",
            leave=False,
            dynamic_ncols=True,
        ):
            X_batch = X_batch.to(device, non_blocking=True)
            I_batch = I_batch.to(device, non_blocking=True)

            if collect_anchor_raw:
                anchor = raw_anchor_prediction(model, X_batch)
                if anchor is not None:
                    anchors.append(anchor.detach().cpu().numpy())

            y_pred = model(X_batch, I_batch)
            predictions.append(y_pred.detach().cpu().numpy())

    y_pred_all = np.concatenate(predictions, axis=0).squeeze(-1)
    anchor_raw = np.concatenate(anchors, axis=0) if anchors else None
    return PredictionBundle(
        y_true=test_split.y_test,
        y_pred=y_pred_all,
        I_unscaled=test_split.I_test,
        anchor_raw=anchor_raw,
    )


def analyze_anchor_error(scenario_key: str, batch_size: int, device: torch.device) -> Dict[str, Any]:
    bundle = predict_hard_coulomb(
        scenario_key=scenario_key,
        eta=None,
        batch_size=batch_size,
        device=device,
        collect_anchor_raw=True,
    )

    t0_abs = np.abs(bundle.y_pred[:, 0] - bundle.y_true[:, 0])
    full_abs = np.abs(bundle.y_pred - bundle.y_true)
    window_max_abs = np.max(full_abs, axis=1)
    window_argmax_t = np.argmax(full_abs, axis=1)
    t0_is_window_max = window_argmax_t == 0

    result: Dict[str, Any] = {
        "scenario": scenario_key,
        "n_windows": int(bundle.y_true.shape[0]),
        "t0_output_error": error_summary_pct(t0_abs),
        "full_sequence_error": error_summary_pct(full_abs.reshape(-1)),
        "window_max_error": error_summary_pct(window_max_abs),
        "window_max_at_t0_count": int(t0_is_window_max.sum()),
        "window_max_at_t0_pct": float(t0_is_window_max.mean() * 100.0),
    }

    if bundle.anchor_raw is not None:
        anchor_abs = np.abs(bundle.anchor_raw - bundle.y_true[:, 0])
        delta0_abs = np.abs(bundle.y_pred[:, 0] - bundle.anchor_raw)
        result["raw_anchor_head_error"] = error_summary_pct(anchor_abs)
        result["t0_constraint_adjustment"] = error_summary_pct(delta0_abs)

    return result


def evaluate_eta_ablation(
    scenario_key: str,
    etas: Iterable[float],
    batch_size: int,
    device: torch.device,
) -> List[Dict[str, float | int]]:
    rows: List[Dict[str, float | int]] = []
    for eta in etas:
        bundle = predict_hard_coulomb(
            scenario_key=scenario_key,
            eta=float(eta),
            batch_size=batch_size,
            device=device,
            collect_anchor_raw=False,
        )
        metrics = regression_metrics(bundle.y_true, bundle.y_pred, bundle.I_unscaled)
        rows.append({"eta": float(eta), **metrics})
    return rows


def print_anchor_report(result: Dict[str, Any]) -> None:
    t0 = result["t0_output_error"]
    full = result["full_sequence_error"]
    win = result["window_max_error"]
    print(f"\n  Anchor / t=0 analysis -- {result['scenario']}")
    print(f"    Windows                         : {result['n_windows']:,}")
    print(
        f"    Output error at t=0             : MAE={t0['mae_pct']:.4f}% | "
        f"RMSE={t0['rmse_pct']:.4f}% | P95={t0['p95_pct']:.4f}% | MaxE={t0['maxe_pct']:.4f}%"
    )
    if "raw_anchor_head_error" in result:
        raw = result["raw_anchor_head_error"]
        adj = result["t0_constraint_adjustment"]
        print(
            f"    Raw anchor-head error           : MAE={raw['mae_pct']:.4f}% | "
            f"RMSE={raw['rmse_pct']:.4f}% | P95={raw['p95_pct']:.4f}% | MaxE={raw['maxe_pct']:.4f}%"
        )
        print(
            f"    |output_t0 - raw_anchor|        : MAE={adj['mae_pct']:.6f}% | "
            f"Max={adj['maxe_pct']:.6f}%"
        )
    print(
        f"    Full-sequence absolute error    : MAE={full['mae_pct']:.4f}% | "
        f"RMSE={full['rmse_pct']:.4f}% | MaxE={full['maxe_pct']:.4f}%"
    )
    print(
        f"    Per-window maximum absolute err : Median={win['median_pct']:.4f}% | "
        f"P95={win['p95_pct']:.4f}% | MaxE={win['maxe_pct']:.4f}%"
    )
    print(
        f"    Window max error occurs at t=0  : {result['window_max_at_t0_count']:,} "
        f"windows ({result['window_max_at_t0_pct']:.2f}%)"
    )


def print_eta_table(rows: List[Dict[str, float | int]]) -> None:
    print("\n  Scenario B safety-factor eta ablation")
    print("  Note: eta changes the Coulomb magnitude envelope only; direction PVR remains structurally clamped.")
    print("  " + "-" * 78)
    print("  Eta | RMSE (%) | MAE (%) | MaxE (%) | PVR (%) | Violations / Discharge Steps")
    print("  " + "-" * 78)
    for row in rows:
        print(
            f"  {row['eta']:>3.1f} | "
            f"{row['rmse_pct']:>8.4f} | "
            f"{row['mae_pct']:>7.4f} | "
            f"{row['maxe_pct']:>8.4f} | "
            f"{row['pvr_pct']:>7.6f} | "
            f"{int(row['pvr_violations']):,} / {int(row['pvr_discharge_steps']):,}"
        )
    print("  " + "-" * 78)


def parse_eta_list(raw: str) -> List[float]:
    etas = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not etas:
        raise argparse.ArgumentTypeError("At least one eta is required.")
    if any(eta <= 0.0 for eta in etas):
        raise argparse.ArgumentTypeError("Eta values must be positive.")
    return etas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze HardCoulombLSTM anchor error and eta ablation.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--etas", type=parse_eta_list, default=list(DEFAULT_ETAS), help="Comma-separated eta list.")
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(OUTPUT_DIR / "sprint48_safety_ablation_results.json"),
        help="Path for JSON results.",
    )
    return parser.parse_args()


def main() -> None:
    configure_utf8_stdio()
    args = parse_args()
    device = resolve_device(args.device)
    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path

    print("=" * 90)
    print("  Sprint 48 Safety Ablation -- Anchor Trap and Eta Rigidity")
    print("=" * 90)
    print(f"  Device      : {device}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Eta values  : {', '.join(f'{eta:g}' for eta in args.etas)}")
    print(f"  PVR rule    : I_unscaled < {DISCHARGE_THRESHOLD_A:.2f} A")

    anchor_results = [
        analyze_anchor_error("scenario_A", args.batch_size, device),
        analyze_anchor_error("scenario_B", args.batch_size, device),
    ]
    for result in anchor_results:
        print_anchor_report(result)

    eta_rows = evaluate_eta_ablation("scenario_B", args.etas, args.batch_size, device)
    print_eta_table(eta_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "anchor_analysis": anchor_results,
        "scenario_B_eta_ablation": eta_rows,
        "pvr_rule": f"I_unscaled < {DISCHARGE_THRESHOLD_A:.2f} A",
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\n  Results saved: {output_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
