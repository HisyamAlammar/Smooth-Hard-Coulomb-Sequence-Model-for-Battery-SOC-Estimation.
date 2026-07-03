"""
ekf_ocv_rint.py -- Phase 8: minimal OCV-Rint Extended Kalman Filter baseline.

State:        x = SOC (scalar per window, vectorized across windows)
Process:      SOC_t = SOC_{t-1} + I_t * dt / (3600 * Q(T)) + w,  w ~ N(0, Q_proc)
Measurement:  V_proxy_t = OCV(SOC_t) + v,                        v ~ N(0, R)
              (V_proxy = V_terminal - I*R_int(T) already removes the Ohmic
               drop in preprocessing, so the ECM reduces to OCV + noise; the
               un-modeled residual is diffusion/polarization overpotential,
               which R must absorb -- worst at cold temperatures.)

Assumptions (documented, chosen a priori, NOT tuned on test data):
  * OCV curve: 25 degC HPPC rests only (train-temperature knowledge,
    Scenario-A-consistent); numerically inverted to SOC -> OCV.
  * Q(T): Q_ACTUAL_PER_TEMP calibration table (same as null baseline).
  * Init: SOC_0 = OCV_inverse(V_proxy at t=0), P_0 = 0.2^2 (weak prior).
  * Q_proc = (5e-5)^2 per 1 s step (Coulomb-integration uncertainty).
  * R sensitivity set {0.01^2, 0.03^2, 0.1^2} V^2 -- ALL reported; no
    test-set selection is performed by this script.

Protocol identical to all other baselines: per-window estimation on the same
test tensors and metrics module. Note the EKF *by design* may raise SOC during
discharge when voltage disagrees (that is its correction mechanism), so its
PVR > 0; the comparison quantifies the accuracy-vs-sign-consistency trade.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.interpolate import PchipInterpolator

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "src"), str(BASE_DIR / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

from predict_utils import load_test_bundle, provenance, unscale_feature  # noqa: E402
from preprocessing_v4 import Q_ACTUAL_PER_TEMP, build_ocv_soc_lookup  # noqa: E402
from soc_metrics import evaluate_soc_predictions  # noqa: E402

OUT_DIR = BASE_DIR / "results" / "baselines"
Q_PROC_STD = 5e-5
P0 = 0.2**2
R_SET = (0.01**2, 0.03**2, 0.1**2)


def build_soc_to_ocv():
    """Invert the 25 degC OCV->SOC PCHIP numerically; return (h, dh/dSOC)."""
    ocv_to_soc, _ = build_ocv_soc_lookup("25degC")
    if ocv_to_soc is None:
        raise RuntimeError("25degC OCV lookup unavailable")
    ocv_grid = np.linspace(2.5, 4.25, 800)
    soc_grid = np.clip(ocv_to_soc(ocv_grid), 0.0, 1.0)
    # strictly increasing SOC support for the inverse (clip creates flats)
    soc_m, first_idx = np.unique(np.round(soc_grid, 6), return_index=True)
    ocv_m = ocv_grid[first_idx]
    if len(soc_m) < 4:
        raise RuntimeError("OCV curve inversion failed: too few unique SOC points")
    h = PchipInterpolator(soc_m, ocv_m, extrapolate=True)
    return h, h.derivative(), ocv_to_soc


def run_ekf(v_proxy: np.ndarray, I: np.ndarray, q_ah: np.ndarray, R: float,
            h, dh, ocv_to_soc) -> np.ndarray:
    """Vectorized scalar EKF over all windows. v_proxy, I: (N, T); q_ah: (N,)."""
    N, T = v_proxy.shape
    soc = np.clip(ocv_to_soc(v_proxy[:, 0]), 0.0, 1.0).astype(np.float64)
    P = np.full(N, P0)
    coeff = 1.0 / (3600.0 * q_ah)
    out = np.empty((N, T))
    out[:, 0] = soc
    for t in range(1, T):
        # predict
        soc = soc + I[:, t] * coeff
        P = P + Q_PROC_STD**2
        # update
        H = np.asarray(dh(np.clip(soc, 0.0, 1.0)))
        residual = v_proxy[:, t] - np.asarray(h(np.clip(soc, 0.0, 1.0)))
        S = H * P * H + R
        K = P * H / S
        soc = np.clip(soc + K * residual, 0.0, 1.0)
        P = (1.0 - K * H) * P
        out[:, t] = soc
    return out.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="OCV-Rint EKF baseline (Phase 8).")
    parser.add_argument("--scenarios", nargs="+", default=["scenario_A", "scenario_B"])
    args = parser.parse_args()

    h, dh, ocv_to_soc = build_soc_to_ocv()
    all_json: Dict[str, object] = {"provenance": provenance("scenario_A", None, {
        "experiment": "ekf_ocv_rint", "q_proc_std": Q_PROC_STD, "P0": P0,
        "R_set_V2": list(R_SET),
        "assumptions": "25degC OCV (train-temp only); Q_actual(T) calibration table; "
                       "no test-set tuning; per-window protocol",
    })}
    rows = []
    for scenario in args.scenarios:
        b = load_test_bundle(scenario)
        v_proxy = unscale_feature(b.X, 0).astype(np.float64)
        q_ah = np.array([Q_ACTUAL_PER_TEMP.get(str(t), 3.0) for t in b.temp_labels])
        scen_out = {}
        for R in R_SET:
            y = run_ekf(v_proxy, b.I.astype(np.float64), q_ah, R, h, dh, ocv_to_soc)
            m = evaluate_soc_predictions(b.y_true, y, b.I, b.temp_labels)
            key = f"R={R:g}"
            scen_out[key] = m
            reg = m["regression"]
            row = {
                "scenario": scenario, "R_V2": R,
                "rmse_pct": round(reg["rmse_full_pct"], 4),
                "mae_pct": round(reg["mae_full_pct"], 4),
                "maxe_pct": round(reg["maxe_full_pct"], 4),
                "pvr_disch_eps0": round(m["pvr"]["discharge"]["by_epsilon"]["0"]["rate_pct"], 4),
                "pvr_disch_eps0.005": round(m["pvr"]["discharge"]["by_epsilon"]["0.005"]["rate_pct"], 4),
            }
            for temp in ("n20degC", "n10degC"):
                pt = m.get("per_temperature", {}).get(temp)
                if pt:
                    row[f"rmse_{temp}"] = round(pt["regression"]["rmse_full_pct"], 4)
            rows.append(row)
            print(f"  {scenario} EKF R={R:g}: RMSE {row['rmse_pct']:8.4f}% | MaxE {row['maxe_pct']:8.4f}% | "
                  f"PVRd0 {row['pvr_disch_eps0']:7.3f}% | n20 RMSE {row.get('rmse_n20degC', '-')}")
        all_json[scenario] = scen_out

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ekf_results.json").write_text(json.dumps(all_json, indent=2, default=float))
    with (OUT_DIR / "ekf_comparison_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved to {OUT_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
