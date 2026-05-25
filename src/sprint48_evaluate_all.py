"""
sprint48_evaluate_all.py -- Unified final v7 evaluator.

Loads only outputs/v7_final checkpoints and evaluates them against the matching
v4 test split. Legacy src/evaluate.py is intentionally not imported.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import r2_score
from tqdm import tqdm

from sprint48_common import (
    BASE_DIR,
    BATCH_SIZE,
    MODEL_KINDS,
    OUTPUT_DIR,
    checkpoint_path,
    configure_utf8_stdio,
    count_trainable_parameters,
    forward_model,
    load_checkpoint,
    load_test_split,
    make_loader,
    model_display_name,
    resolve_device,
)

SCENARIO_DATA_DIRS = {
    "scenario_A": BASE_DIR / "data" / "processed" / "v4_scenario_A",
    "scenario_B": BASE_DIR / "data" / "processed" / "v4_scenario_B",
}


def compute_pvr(y_pred: np.ndarray, I_unscaled: np.ndarray) -> Dict[str, float | int]:
    """
    Physics Violation Rate for sequence-to-sequence predictions.

    Violation: predicted SOC increases from t-1 to t during discharge.
    Discharge condition is deliberately explicit: I_unscaled < -0.05 A.
    """
    if y_pred.ndim != 2 or I_unscaled.ndim != 2:
        raise ValueError(f"PVR expects 2D arrays, got y_pred={y_pred.shape}, I={I_unscaled.shape}")
    if y_pred.shape != I_unscaled.shape:
        raise ValueError(f"PVR shape mismatch: y_pred={y_pred.shape}, I={I_unscaled.shape}")
    delta_soc = y_pred[:, 1:] - y_pred[:, :-1]
    discharge_mask = I_unscaled[:, 1:] < -0.05
    violations = (delta_soc > 0.0) & discharge_mask
    discharge_steps = int(discharge_mask.sum())
    violation_count = int(violations.sum())
    pvr_pct = 0.0 if discharge_steps == 0 else (violation_count / discharge_steps) * 100.0
    return {
        "pvr_pct": float(pvr_pct),
        "violations": violation_count,
        "discharge_steps": discharge_steps,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, I_unscaled: np.ndarray) -> Dict[str, Any]:
    errors = y_pred - y_true
    last_errors = y_pred[:, -1] - y_true[:, -1]
    pvr = compute_pvr(y_pred, I_unscaled)
    return {
        "rmse_full_pct": float(np.sqrt(np.mean(errors ** 2)) * 100.0),
        "mae_full_pct": float(np.mean(np.abs(errors)) * 100.0),
        "maxe_full_pct": float(np.max(np.abs(errors)) * 100.0),
        "r2_full": float(r2_score(y_true.reshape(-1), y_pred.reshape(-1))),
        "rmse_last_pct": float(np.sqrt(np.mean(last_errors ** 2)) * 100.0),
        "mae_last_pct": float(np.mean(np.abs(last_errors)) * 100.0),
        "maxe_last_pct": float(np.max(np.abs(last_errors)) * 100.0),
        "r2_last": float(r2_score(y_true[:, -1], y_pred[:, -1])),
        "pvr_pct": pvr["pvr_pct"],
        "pvr_violations": pvr["violations"],
        "pvr_discharge_steps": pvr["discharge_steps"],
        "n_windows": int(y_true.shape[0]),
        "window": int(y_true.shape[1]),
    }


def compute_per_temp_last_step(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    temp_labels: np.ndarray | None,
) -> Dict[str, Dict[str, float | int]]:
    if temp_labels is None or len(temp_labels) != y_true.shape[0]:
        return {}
    results: Dict[str, Dict[str, float | int]] = {}
    y_true_last = y_true[:, -1]
    y_pred_last = y_pred[:, -1]
    for temp in sorted(np.unique(temp_labels)):
        mask = temp_labels == temp
        errors = y_pred_last[mask] - y_true_last[mask]
        results[str(temp)] = {
            "rmse_pct": float(np.sqrt(np.mean(errors ** 2)) * 100.0),
            "mae_pct": float(np.mean(np.abs(errors)) * 100.0),
            "n_windows": int(mask.sum()),
        }
    return results


def run_inference(
    scenario_key: str,
    model_kind: str,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    data_dir = SCENARIO_DATA_DIRS[scenario_key]
    checkpoint = checkpoint_path(model_kind, scenario_key, latest=False)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

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
    model.eval()
    all_predictions: List[np.ndarray] = []
    with torch.no_grad():
        for X_batch, _y_batch, I_batch in tqdm(
            loader,
            desc=f"Eval {model_display_name(model_kind)} {scenario_key}",
            leave=False,
            dynamic_ncols=True,
        ):
            X_batch = X_batch.to(device, non_blocking=True)
            I_batch = I_batch.to(device, non_blocking=True)
            y_pred = forward_model(model, model_kind, X_batch, I_batch)
            all_predictions.append(y_pred.detach().cpu().numpy())

    y_pred_all = np.concatenate(all_predictions, axis=0).squeeze(-1)
    y_true_all = test_split.y_test
    I_unscaled = test_split.I_test
    metrics = compute_metrics(y_true_all, y_pred_all, I_unscaled)
    per_temp = compute_per_temp_last_step(y_true_all, y_pred_all, test_split.temp_labels)

    return {
        "scenario": scenario_key,
        "model_kind": model_kind,
        "model_name": model_display_name(model_kind),
        "checkpoint": str(checkpoint.relative_to(BASE_DIR)),
        "checkpoint_epoch": int(checkpoint_payload.get("epoch", -1)),
        "parameter_count": int(count_trainable_parameters(model)),
        "data_source": str(data_dir.relative_to(BASE_DIR)),
        "metrics": metrics,
        "per_temp_last_step": per_temp,
    }


def print_result(result: Dict[str, Any]) -> None:
    metrics = result["metrics"]
    print(
        f"  {result['scenario']} | {result['model_name']} | "
        f"params={result['parameter_count']:,} | epoch={result['checkpoint_epoch']}"
    )
    print(
        f"    Full: RMSE={metrics['rmse_full_pct']:.4f}% | "
        f"MAE={metrics['mae_full_pct']:.4f}% | MaxE={metrics['maxe_full_pct']:.4f}% | "
        f"R2={metrics['r2_full']:.6f}"
    )
    print(
        f"    Last: RMSE={metrics['rmse_last_pct']:.4f}% | "
        f"MAE={metrics['mae_last_pct']:.4f}% | MaxE={metrics['maxe_last_pct']:.4f}% | "
        f"R2={metrics['r2_last']:.6f}"
    )
    print(
        f"    PVR : {metrics['pvr_pct']:.6f}% "
        f"({metrics['pvr_violations']:,} / {metrics['pvr_discharge_steps']:,}; I_unscaled < -0.05 A)"
    )
    if result["per_temp_last_step"]:
        print("    Per-temperature last-step RMSE:")
        for temp, values in result["per_temp_last_step"].items():
            print(
                f"      {temp:>8s}: RMSE={values['rmse_pct']:.4f}% | "
                f"MAE={values['mae_pct']:.4f}% | N={values['n_windows']:,}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all sprint48 v7_final checkpoints.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_utf8_stdio()
    args = parse_args()
    device = resolve_device(args.device)
    results: List[Dict[str, Any]] = []

    print("=" * 90)
    print("  Sprint 48 Unified Evaluation -- outputs/v7_final only")
    print("=" * 90)
    print(f"  Device: {device}")

    missing: List[str] = []
    for scenario_key in ("scenario_A", "scenario_B"):
        for model_kind in MODEL_KINDS:
            try:
                result = run_inference(scenario_key, model_kind, args.batch_size, device)
            except FileNotFoundError as exc:
                if args.allow_missing:
                    missing.append(str(exc))
                    continue
                raise
            print_result(result)
            results.append(result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "sprint48_evaluation_results.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    if missing:
        print("\n  Missing checkpoints skipped:")
        for item in missing:
            print(f"    {item}")
    print(f"\n  Results saved: {output_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
