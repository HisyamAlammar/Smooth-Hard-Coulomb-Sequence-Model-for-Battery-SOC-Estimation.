"""
null_ocv_coulomb.py -- Baseline A: zero-parameter OCV-inverse + Coulomb counting.

Structure mirrors the Hard-Coulomb model's decomposition (anchor + bounded
increments) with NO learned parameters:
    SOC_0   = OCV_inverse(V_proxy at t=0)     (PCHIP lookup built from HPPC rests)
    SOC_t   = SOC_0 + cumsum(I * dt) / (3600 * Q)
V_proxy already removes the Ohmic drop (V - I*R_int), so the OCV lookup is the
best-case voltage anchor available to the learned model as well.

Variants (calibration assumptions, all documented in output):
    ocv25_qnom : 25 degC OCV curve + nominal Q          (strictest: only
                 train-temperature knowledge, Scenario-A-consistent)
    ocv25_qtemp: 25 degC OCV curve + Q_actual(T) table
    ocvT_qnom  : per-temperature OCV curve + nominal Q
    ocvT_qtemp : per-temperature OCV curve + Q_actual(T) (full calibration,
                 standard BMS practice: OCV/Q tables come from cell
                 characterization, not from the drive-cycle test data)
    oracle_qtemp: TRUE SOC at t=0 + Q_actual(T)          (upper bound; shows
                 how far a perfect anchor alone would go -- ties to Phase 4)

Windows, splits, and metrics are identical to the learned models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "src"), str(BASE_DIR / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import Q_NOMINAL  # noqa: E402
from predict_utils import load_test_bundle, provenance, unscale_feature  # noqa: E402
from preprocessing_v4 import Q_ACTUAL_PER_TEMP, build_ocv_soc_lookup  # noqa: E402
from soc_metrics import evaluate_soc_predictions  # noqa: E402

RESULTS_DIR = BASE_DIR / "results" / "baselines"
VARIANTS = ("ocv25_qnom", "ocv25_qtemp", "ocvT_qnom", "ocvT_qtemp", "oracle_qtemp")


def coulomb_trajectory(soc0: np.ndarray, I: np.ndarray, q_ah: np.ndarray) -> np.ndarray:
    """soc0 (N,), I (N,T) in A, q_ah (N,) in Ah -> SOC (N,T), clipped to [0,1].

    Increment convention matches the Hard-Coulomb layer: delta[t] is applied
    at step t (cumsum over the window), dt = 1 s. np.clip is monotone, so
    clipping preserves sign-consistency of increments.
    """
    delta = I / (3600.0 * q_ah[:, None])
    soc = soc0[:, None] + np.cumsum(delta, axis=1)
    return np.clip(soc, 0.0, 1.0)


def run_scenario(scenario: str) -> Dict[str, object]:
    bundle = load_test_bundle(scenario)
    v_proxy_t0 = unscale_feature(bundle.X, 0)[:, 0]
    temps = bundle.temp_labels
    if temps is None:
        raise RuntimeError(f"{scenario}: temp_labels_test.npy required for capacity/OCV variants")

    # Build OCV->SOC lookups once per temperature present in the split
    lookups = {}
    for temp in np.unique(temps):
        interp, q_meas = build_ocv_soc_lookup(str(temp))
        lookups[str(temp)] = interp
    ocv25, _ = build_ocv_soc_lookup("25degC")
    if ocv25 is None:
        raise RuntimeError("25degC HPPC OCV lookup could not be built")

    q_nom = np.full(len(temps), Q_NOMINAL, dtype=np.float64)
    q_temp = np.array([Q_ACTUAL_PER_TEMP.get(str(t), Q_NOMINAL) for t in temps], dtype=np.float64)

    def anchor_ocv25() -> np.ndarray:
        return np.clip(ocv25(v_proxy_t0), 0.0, 1.0)

    def anchor_ocvT() -> np.ndarray:
        soc0 = np.empty(len(temps), dtype=np.float64)
        for temp in np.unique(temps):
            m = temps == temp
            fn = lookups[str(temp)] or ocv25  # fall back if a temp lacks HPPC rests
            soc0[m] = np.clip(fn(v_proxy_t0[m]), 0.0, 1.0)
        return soc0

    anchors = {
        "ocv25_qnom": (anchor_ocv25(), q_nom),
        "ocv25_qtemp": (anchor_ocv25(), q_temp),
        "ocvT_qnom": (anchor_ocvT(), q_nom),
        "ocvT_qtemp": (anchor_ocvT(), q_temp),
        "oracle_qtemp": (bundle.y_true[:, 0].astype(np.float64), q_temp),
    }

    out: Dict[str, object] = {"provenance": provenance(scenario, checkpoint=None, extra={
        "model": "null_ocv_coulomb", "parameter_count": 0,
        "assumptions": "OCV/Q from HPPC characterization files (calibration data, "
                       "not drive-cycle test data); dt=1s; V_proxy anchor voltage",
    })}
    preds_to_save = {}
    for name in VARIANTS:
        soc0, q = anchors[name]
        y_pred = coulomb_trajectory(soc0, bundle.I.astype(np.float64), q).astype(np.float32)
        metrics = evaluate_soc_predictions(bundle.y_true, y_pred, bundle.I, temps)
        out[name] = metrics
        preds_to_save[name] = y_pred
        reg = metrics["regression"]
        print(f"  {scenario} null[{name:12s}]  RMSE {reg['rmse_full_pct']:7.4f}% | "
              f"MAE {reg['mae_full_pct']:7.4f}% | MaxE {reg['maxe_full_pct']:7.4f}% | "
              f"PVR(disch,0) {metrics['pvr']['discharge']['by_epsilon']['0']['rate_pct']:.4f}%")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS_DIR / f"null_ocv_coulomb_predictions_{scenario}.npz", **preds_to_save)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-parameter OCV+Coulomb null baseline.")
    parser.add_argument("--scenarios", nargs="+", default=["scenario_A", "scenario_B"])
    parser.add_argument("--output", default=str(RESULTS_DIR / "null_ocv_coulomb_results.json"))
    args = parser.parse_args()

    results = {s: run_scenario(s) for s in args.scenarios}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
