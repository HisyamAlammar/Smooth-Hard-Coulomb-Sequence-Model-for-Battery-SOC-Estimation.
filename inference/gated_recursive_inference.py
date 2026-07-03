"""
gated_recursive_inference.py -- Phase 4: inference policies over window chains.

All policies operate on the SAME trained Hard-Coulomb checkpoint; they differ
only in how each window's anchor is chosen. No model weights change.

Policies
  windowed_independent : legacy -- learned anchor every window
  carried_anchor       : previous window's estimate at the overlap step
  temperature_gated    : carried when cold (T(t=0) < 5 degC), learned otherwise
  load_gated           : carried when window starts under load (|I(t=0)| > thr)
  rest_gated_reanchor  : re-anchor ONLY at warm-ish rest with stable voltage
                         (|I|<=thr and |dVp/dt|<1 mV/s); carried otherwise
  confidence_weighted_blend : alpha*carried + (1-alpha)*learned, alpha rises
                         with cold and load (voltage-anchor unreliability)
  hybrid_temperature_load_gate : carried when cold OR loaded start

Gating parameters are fixed a priori (physics-motivated, NOT tuned on test):
  COLD_T_C = 5.0 degC; load threshold = config.CURRENT_THRESHOLD_A;
  voltage stability = 1e-3 V/s; blend: cold ramp 5..-20 degC, load ramp 0..2 A.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "src"), str(BASE_DIR / "analysis"), str(BASE_DIR / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import CURRENT_THRESHOLD_A, PHYS_MAX_V3, PHYS_MIN_V3  # noqa: E402
from oracle_anchor import hc_cumulative_delta  # noqa: E402
from preprocessing_v4 import PROFILE_KEY_SCALE  # noqa: E402

STRIDE = 10
COLD_T_C = 5.0
DVDT_STABLE = 1e-3  # V/s
BLEND_LOAD_REF_A = 2.0

POLICIES = (
    "windowed_independent", "carried_anchor", "temperature_gated", "load_gated",
    "rest_gated_reanchor", "confidence_weighted_blend", "hybrid_temperature_load_gate",
)


def _unscale(X: np.ndarray, col: int) -> np.ndarray:
    lo, hi = PHYS_MIN_V3[col], PHYS_MAX_V3[col]
    return X[:, :, col] * (hi - lo) + lo


def precompute(model: torch.nn.Module, X: np.ndarray, I: np.ndarray,
               device: torch.device, batch: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """Per-window routed cumulative deltas + learned anchor value."""
    cumulative = np.empty(I.shape, dtype=np.float32)
    anchor = np.empty(len(I), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i+batch]).to(device)
            ib = torch.from_numpy(I[i:i+batch]).to(device)
            cum = hc_cumulative_delta(model, xb, ib).cpu().numpy()
            y = model(xb, ib).squeeze(-1).cpu().numpy()
            cumulative[i:i+batch] = cum
            anchor[i:i+batch] = y[:, 0] - cum[:, 0]
    return cumulative, anchor


def gating_features(X: np.ndarray, I: np.ndarray) -> Dict[str, np.ndarray]:
    T0 = _unscale(X, 2)[:, 0]
    dvp0 = np.abs(_unscale(X, 3)[:, 0])
    I0 = np.abs(I[:, 0])
    return {"T0_C": T0, "absI0_A": I0, "abs_dvpdt0": dvp0,
            "rest_start": I0 <= CURRENT_THRESHOLD_A,
            "cold": T0 < COLD_T_C,
            "stable_v": dvp0 < DVDT_STABLE}


def _alpha_blend(feat: Dict[str, np.ndarray], j: int) -> float:
    cold = float(np.clip((COLD_T_C - feat["T0_C"][j]) / (COLD_T_C + 20.0), 0.0, 1.0))
    load = float(np.clip(feat["absI0_A"][j] / BLEND_LOAD_REF_A, 0.0, 1.0))
    return float(np.clip(0.5 * cold + 0.5 * load, 0.0, 1.0))


def _use_carried(policy: str, feat: Dict[str, np.ndarray], j: int) -> bool:
    if policy == "carried_anchor":
        return True
    if policy == "temperature_gated":
        return bool(feat["cold"][j])
    if policy == "load_gated":
        return not bool(feat["rest_start"][j])
    if policy == "rest_gated_reanchor":
        reanchor = bool(feat["rest_start"][j]) and bool(feat["stable_v"][j]) and not bool(feat["cold"][j])
        return not reanchor
    if policy == "hybrid_temperature_load_gate":
        return bool(feat["cold"][j]) or not bool(feat["rest_start"][j])
    raise ValueError(policy)


def run_policy(
    policy: str,
    cumulative: np.ndarray,
    anchor_model: np.ndarray,
    timestamp_keys: np.ndarray,
    feat: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Returns stitched predictions (N,T) and policy statistics."""
    n = len(cumulative)
    if policy == "windowed_independent":
        y = np.clip(anchor_model[:, None] + cumulative, 0.0, 1.0)
        return y, {"reanchor_pct": 100.0, "chain_start_pct": 100.0}

    start_key = timestamp_keys[:, 0]
    order = np.argsort(start_key, kind="stable")
    y = np.empty_like(cumulative)
    reanchors = chain_starts = 0
    prev = -1
    for j in order:
        contiguous = (
            prev >= 0
            and start_key[j] == start_key[prev] + STRIDE
            and start_key[j] // PROFILE_KEY_SCALE == start_key[prev] // PROFILE_KEY_SCALE
        )
        if not contiguous:
            anchor = anchor_model[j]
            chain_starts += 1
            reanchors += 1
        else:
            carried = y[prev, STRIDE] - cumulative[j, 0]
            if policy == "confidence_weighted_blend":
                a = _alpha_blend(feat, j)
                anchor = a * carried + (1.0 - a) * anchor_model[j]
                if a < 0.5:
                    reanchors += 1
            elif _use_carried(policy, feat, j):
                anchor = carried
            else:
                anchor = anchor_model[j]
                reanchors += 1
        y[j] = np.clip(anchor + cumulative[j], 0.0, 1.0)
        prev = j
    return y, {"reanchor_pct": round(100.0 * reanchors / n, 2),
               "chain_start_pct": round(100.0 * chain_starts / n, 2)}
