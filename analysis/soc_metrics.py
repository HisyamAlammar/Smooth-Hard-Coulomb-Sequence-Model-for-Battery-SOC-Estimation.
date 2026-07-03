"""
soc_metrics.py -- Shared SOC evaluation metrics (Phase 1 audit fix).

Single implementation of PVR and delta-magnitude metrics so that model code
and evaluation code cannot drift apart. Replaces the duplicated `compute_pvr`
in sprint48_evaluate_all.py and sprint48_safety_ablation.py.

Design decisions (from the red-team audit):
  * PVR is reported per current region (discharge / charge / rest) and per
    dead-band epsilon, because a zero-epsilon, discharge-only PVR (a) counts
    float-noise wiggles as violations for unconstrained models and (b) is
    satisfied trivially by a frozen output.
  * Delta-magnitude tracking is reported alongside PVR: sign-consistency says
    nothing about whether predicted SOC moves at the physically correct rate.
  * The current threshold comes from config.CURRENT_THRESHOLD_A -- the same
    constant the Hard-Coulomb layer routes on. This makes the circularity of
    structural PVR explicit rather than accidental.

All SOC quantities are fractions in [0, 1]; deltas are per-step fractions.
Sequence convention: arrays are (N_windows, T). Delta index t corresponds to
y[:, t+1] - y[:, t] and is classified by the current at t+1, matching the
Hard-Coulomb layer, which routes delta[t] with I[t] and accumulates via cumsum.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, Sequence

import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config import CURRENT_THRESHOLD_A, PVR_EPSILONS  # noqa: E402

REGIONS = ("discharge", "charge", "rest")


def _check_2d_pair(y_pred: np.ndarray, I: np.ndarray) -> None:
    if y_pred.ndim != 2 or I.ndim != 2:
        raise ValueError(f"expected 2D arrays, got y_pred={y_pred.shape}, I={I.shape}")
    if y_pred.shape != I.shape:
        raise ValueError(f"shape mismatch: y_pred={y_pred.shape}, I={I.shape}")


def region_masks(I: np.ndarray, threshold: float = CURRENT_THRESHOLD_A) -> Dict[str, np.ndarray]:
    """Masks over delta positions: classify delta[t] = y[t+1]-y[t] by I[t+1]."""
    if I.ndim != 2:
        raise ValueError(f"expected 2D current array, got {I.shape}")
    I_step = I[:, 1:]
    return {
        "discharge": I_step < -threshold,
        "charge": I_step > threshold,
        "rest": np.abs(I_step) <= threshold,
    }


def _violation_deltas(delta: np.ndarray, region: str) -> np.ndarray:
    """Signed violation magnitude (positive = size of the violation)."""
    if region == "discharge":
        return delta            # violation when delta > eps
    if region == "charge":
        return -delta           # violation when -delta > eps (delta < -eps)
    if region == "rest":
        return np.abs(delta)    # violation when |delta| > eps (drift at rest)
    raise ValueError(f"unknown region: {region}")


def pvr_metrics(
    y_pred: np.ndarray,
    I: np.ndarray,
    threshold: float = CURRENT_THRESHOLD_A,
    epsilons: Sequence[float] = tuple(PVR_EPSILONS),
) -> Dict[str, object]:
    """
    Region- and epsilon-resolved Physics Violation Rate.

    Returns per region: step count, violation rate (%) per epsilon, and
    magnitude statistics of zero-epsilon violations. `total` pools all steps.
    Rest-region note: at eps=0 any nonzero drift counts, which is maximally
    strict; interpret rest PVR primarily at eps > 0.
    """
    _check_2d_pair(y_pred, I)
    delta = y_pred[:, 1:] - y_pred[:, :-1]
    masks = region_masks(I, threshold)

    out: Dict[str, object] = {
        "threshold_A": float(threshold),
        "epsilons": [float(e) for e in epsilons],
        "n_delta_steps": int(delta.size),
    }
    total_viol_by_eps = {float(e): 0 for e in epsilons}

    for region in REGIONS:
        mask = masks[region]
        n_steps = int(mask.sum())
        region_out: Dict[str, object] = {"n_steps": n_steps}
        v = _violation_deltas(delta[mask], region) if n_steps else np.empty(0, dtype=delta.dtype)

        rates = {}
        for eps in epsilons:
            count = int((v > eps).sum())
            total_viol_by_eps[float(eps)] += count
            rates[f"{eps:g}"] = {
                "rate_pct": float(100.0 * count / n_steps) if n_steps else 0.0,
                "violations": count,
            }
        region_out["by_epsilon"] = rates

        pos = v[v > 0]
        region_out["violation_magnitude_eps0"] = (
            {
                "mean": float(pos.mean()),
                "median": float(np.median(pos)),
                "p95": float(np.percentile(pos, 95)),
                "max": float(pos.max()),
            }
            if pos.size
            else None
        )
        if n_steps == 0:
            region_out["note"] = "empty region for this dataset/threshold"
        out[region] = region_out

    n_total = int(delta.size)
    out["total"] = {
        "n_steps": n_total,
        "by_epsilon": {
            f"{eps:g}": {
                "rate_pct": float(100.0 * total_viol_by_eps[float(eps)] / n_total) if n_total else 0.0,
                "violations": int(total_viol_by_eps[float(eps)]),
            }
            for eps in epsilons
        },
    }
    return out


def legacy_pvr(y_pred: np.ndarray, I: np.ndarray, threshold: float = CURRENT_THRESHOLD_A) -> Dict[str, object]:
    """
    Backward-compatible discharge-only, eps=0 PVR (the pre-audit definition).
    Kept so historical numbers remain reproducible; new work should read
    pvr_metrics() instead.
    """
    full = pvr_metrics(y_pred, I, threshold, epsilons=(0.0,))
    d = full["discharge"]
    return {
        "pvr_pct": d["by_epsilon"]["0"]["rate_pct"],
        "violations": d["by_epsilon"]["0"]["violations"],
        "discharge_steps": d["n_steps"],
    }


def delta_magnitude_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    I: np.ndarray,
    threshold: float = CURRENT_THRESHOLD_A,
) -> Dict[str, object]:
    """
    Does predicted SOC move at the right rate? (PVR cannot see this.)
    Per region and overall: MAE/RMSE of per-step deltas, mean absolute
    predicted vs true delta, and their ratio (1.0 = correct average rate,
    0.0 = frozen output).
    """
    _check_2d_pair(y_pred, I)
    _check_2d_pair(y_true, I)
    d_pred = y_pred[:, 1:] - y_pred[:, :-1]
    d_true = y_true[:, 1:] - y_true[:, :-1]
    masks = region_masks(I, threshold)
    masks = {**masks, "all": np.ones_like(d_pred, dtype=bool)}

    out: Dict[str, object] = {"threshold_A": float(threshold)}
    for name, mask in masks.items():
        n = int(mask.sum())
        if n == 0:
            out[name] = {"n_steps": 0, "note": "empty region"}
            continue
        dp, dt = d_pred[mask], d_true[mask]
        err = dp - dt
        mean_abs_true = float(np.abs(dt).mean())
        mean_abs_pred = float(np.abs(dp).mean())
        out[name] = {
            "n_steps": n,
            "delta_soc_mae": float(np.abs(err).mean()),
            "delta_soc_rmse": float(np.sqrt(np.mean(err**2))),
            "mean_abs_pred_delta": mean_abs_pred,
            "mean_abs_true_delta": mean_abs_true,
            "pred_true_delta_ratio": float(mean_abs_pred / mean_abs_true) if mean_abs_true > 0 else None,
        }
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """RMSE/MAE/MaxE/R2 over the full sequence and at the last step, in %."""
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError(f"bad shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")
    err = (y_pred - y_true).astype(np.float64)
    last = err[:, -1]

    def _r2(t: np.ndarray, p: np.ndarray) -> float:
        ss_res = float(np.sum((t - p) ** 2))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "rmse_full_pct": float(np.sqrt(np.mean(err**2)) * 100.0),
        "mae_full_pct": float(np.mean(np.abs(err)) * 100.0),
        "maxe_full_pct": float(np.max(np.abs(err)) * 100.0),
        "r2_full": _r2(y_true.reshape(-1).astype(np.float64), y_pred.reshape(-1).astype(np.float64)),
        "rmse_last_pct": float(np.sqrt(np.mean(last**2)) * 100.0),
        "mae_last_pct": float(np.mean(np.abs(last)) * 100.0),
        "maxe_last_pct": float(np.max(np.abs(last)) * 100.0),
        "r2_last": _r2(y_true[:, -1].astype(np.float64), y_pred[:, -1].astype(np.float64)),
    }


def per_temperature_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    I: np.ndarray,
    temp_labels: np.ndarray | None,
    threshold: float = CURRENT_THRESHOLD_A,
    epsilons: Sequence[float] = tuple(PVR_EPSILONS),
) -> Dict[str, Dict[str, object]]:
    """Full metric bundle split by per-window temperature label."""
    if temp_labels is None or len(temp_labels) != y_true.shape[0]:
        return {}
    out: Dict[str, Dict[str, object]] = {}
    for temp in sorted(np.unique(temp_labels)):
        m = temp_labels == temp
        out[str(temp)] = {
            "n_windows": int(m.sum()),
            "regression": regression_metrics(y_true[m], y_pred[m]),
            "pvr": pvr_metrics(y_pred[m], I[m], threshold, epsilons),
            "delta_magnitude": delta_magnitude_metrics(y_pred[m], y_true[m], I[m], threshold),
        }
    return out


def evaluate_soc_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    I: np.ndarray,
    temp_labels: np.ndarray | None = None,
    threshold: float = CURRENT_THRESHOLD_A,
    epsilons: Sequence[float] = tuple(PVR_EPSILONS),
) -> Dict[str, object]:
    """One-call bundle: regression + region/epsilon PVR + delta magnitude (+ per-temp)."""
    bundle: Dict[str, object] = {
        "n_windows": int(y_true.shape[0]),
        "window": int(y_true.shape[1]),
        "regression": regression_metrics(y_true, y_pred),
        "pvr": pvr_metrics(y_pred, I, threshold, epsilons),
        "delta_magnitude": delta_magnitude_metrics(y_pred, y_true, I, threshold),
        "legacy": legacy_pvr(y_pred, I, threshold),
    }
    per_temp = per_temperature_metrics(y_true, y_pred, I, temp_labels, threshold, epsilons)
    if per_temp:
        bundle["per_temperature"] = per_temp
    return bundle
