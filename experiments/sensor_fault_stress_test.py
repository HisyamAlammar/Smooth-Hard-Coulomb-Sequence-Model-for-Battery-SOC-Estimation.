"""
sensor_fault_stress_test.py -- Phase 7: guarantee vs measured current, not physics.

Simulates current-sensor faults consistently across EVERY signal the BMS would
derive from the faulty sensor:
    I_meas       = I_true + offset + noise   (or stuck at 0)
    V_proxy_meas = V_terminal - I_meas*R_int(T) = V_proxy_true - (I_meas-I_true)*R_int(T)
    dV_proxy_dt, dI_dt recomputed from the measured-domain signals (with the
    original preprocessing clips); t=0 derivative kept from stored context.

For each fault we report BOTH:
    PVR vs the measured (faulty) current  -> stays 0 by construction (HC)
    PVR vs the TRUE current               -> the physically meaningful rate
plus forced rest drift (mean |per-step delta| during true-rest steps) and
regression error. This quantifies audit finding 3: the constraint can emit
physically-plausible-but-wrong trajectories under sensor bias, invisible to
the audited PVR.

Scenario A test split; production HardCoulombLSTM; VanillaLSTM included at
selected faults for contrast.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "src"), str(BASE_DIR / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import CURRENT_THRESHOLD_A, R_INT_PER_TEMP  # noqa: E402
from predict_utils import (  # noqa: E402
    X_MAX,
    X_MIN,
    load_test_bundle,
    provenance,
    scale_feature,
    unscale_feature,
)
from soc_metrics import evaluate_soc_predictions, pvr_metrics, region_masks  # noqa: E402
from sprint48_common import checkpoint_path, forward_model, load_checkpoint, resolve_device  # noqa: E402

OUT_DIR = BASE_DIR / "results" / "robustness"
SEED = 42

FAULTS = (
    ("offset", +0.05), ("offset", -0.05),
    ("offset", +0.10), ("offset", -0.10),
    ("offset", +0.50), ("offset", -0.50),
    ("offset", +1.00), ("offset", -1.00),
    ("gauss_noise", 0.10), ("gauss_noise", 0.50),
    ("stuck_zero", 0.0),
)


def build_faulty_inputs(bundle, kind: str, value: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_meas, I_meas) with the fault propagated through derived features."""
    I_true = bundle.I.astype(np.float64)
    if kind == "offset":
        I_meas = I_true + value
    elif kind == "gauss_noise":
        I_meas = I_true + rng.normal(0.0, value, size=I_true.shape)
    elif kind == "stuck_zero":
        I_meas = np.zeros_like(I_true)
    else:
        raise ValueError(kind)

    r_int = np.array([R_INT_PER_TEMP.get(str(t), 0.03) for t in bundle.temp_labels])[:, None]
    v_proxy_true = unscale_feature(bundle.X, 0).astype(np.float64)
    v_proxy_meas = v_proxy_true - (I_meas - I_true) * r_int

    X_meas = bundle.X.copy()
    X_meas[:, :, 0] = scale_feature(v_proxy_meas, 0)
    X_meas[:, :, 1] = scale_feature(I_meas, 1)
    # derivatives in the measured domain (dt = 1 s), original preprocessing clips
    dvp = np.zeros_like(v_proxy_meas)
    dvp[:, 1:] = np.clip(np.diff(v_proxy_meas, axis=1), -2.0, 2.0)
    dvp[:, 0] = unscale_feature(bundle.X, 3)[:, 0]  # keep stored context value at t=0
    di = np.zeros_like(I_meas)
    di[:, 1:] = np.clip(np.diff(I_meas, axis=1), -20.0, 20.0)
    di[:, 0] = unscale_feature(bundle.X, 4)[:, 0]
    X_meas[:, :, 3] = scale_feature(dvp, 3)
    X_meas[:, :, 4] = scale_feature(di, 4)
    return X_meas.astype(np.float32), I_meas.astype(np.float32)


def predict(model, model_kind, X, I, device, batch=1024) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i : i + batch]).to(device)
            ib = torch.from_numpy(I[i : i + batch]).to(device)
            preds.append(forward_model(model, model_kind, xb, ib).squeeze(-1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def rest_drift(y_pred: np.ndarray, I_true: np.ndarray) -> Dict[str, float]:
    """Signed/absolute per-step drift during TRUE-rest steps."""
    delta = y_pred[:, 1:] - y_pred[:, :-1]
    rest = region_masks(I_true)["rest"]
    if not rest.any():
        return {"n_steps": 0}
    d = delta[rest]
    return {
        "n_steps": int(rest.sum()),
        "mean_signed_delta": float(d.mean()),
        "mean_abs_delta": float(np.abs(d).mean()),
        "drift_pct_per_hour_equiv": float(d.mean() * 3600.0 * 100.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Current-sensor fault stress test (Phase 7).")
    parser.add_argument("--scenario", default="scenario_A")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    rng = np.random.default_rng(SEED)
    bundle = load_test_bundle(args.scenario)
    if bundle.temp_labels is None:
        raise RuntimeError("temp labels required for R_int propagation")

    hc, _ = load_checkpoint(checkpoint_path("hard_coulomb_lstm", args.scenario, latest=False), device)
    vn, _ = load_checkpoint(checkpoint_path("vanilla_lstm", args.scenario, latest=False), device)
    hc.eval(), vn.eval()

    results: List[Dict] = []
    rows: List[Dict] = []

    # clean reference
    conditions = [("clean", 0.0)] + list(FAULTS)
    for kind, value in conditions:
        if kind == "clean":
            X_meas, I_meas = bundle.X, bundle.I
        else:
            X_meas, I_meas = build_faulty_inputs(bundle, kind, value, rng)

        for model_kind, model in (("hard_coulomb_lstm", hc), ("vanilla_lstm", vn)):
            # vanilla only at selected conditions to bound runtime
            if model_kind == "vanilla_lstm" and (kind, value) not in (
                ("clean", 0.0), ("offset", 0.5), ("offset", -0.5), ("stuck_zero", 0.0)
            ):
                continue
            y = predict(model, model_kind, X_meas, I_meas, device)
            m = evaluate_soc_predictions(bundle.y_true, y, bundle.I)  # regression vs truth
            pvr_meas = pvr_metrics(y, I_meas)     # audited condition (faulty sensor)
            pvr_true = pvr_metrics(y, bundle.I)   # physical condition (true current)
            drift = rest_drift(y, bundle.I)

            entry = {
                "fault": kind, "value": value, "model": model_kind,
                "regression": m["regression"],
                "pvr_vs_measured_disch_eps0": pvr_meas["discharge"]["by_epsilon"]["0"]["rate_pct"],
                "pvr_vs_true_disch_eps0": pvr_true["discharge"]["by_epsilon"]["0"]["rate_pct"],
                "pvr_vs_true_charge_eps0": pvr_true["charge"]["by_epsilon"]["0"]["rate_pct"],
                "pvr_vs_true_rest_eps0": pvr_true["rest"]["by_epsilon"]["0"]["rate_pct"],
                "true_rest_drift": drift,
                "delta_magnitude_vs_truth": m["delta_magnitude"]["discharge"],
            }
            results.append(entry)
            reg = m["regression"]
            rows.append({
                "fault": kind, "value": value, "model": model_kind,
                "rmse_pct": round(reg["rmse_full_pct"], 4),
                "mae_pct": round(reg["mae_full_pct"], 4),
                "maxe_pct": round(reg["maxe_full_pct"], 4),
                "pvr_measured_disch_eps0": round(entry["pvr_vs_measured_disch_eps0"], 4),
                "pvr_true_disch_eps0": round(entry["pvr_vs_true_disch_eps0"], 4),
                "pvr_true_rest_eps0": round(entry["pvr_vs_true_rest_eps0"], 4),
                "rest_drift_pct_per_hour": round(drift.get("drift_pct_per_hour_equiv", 0.0), 3),
            })
            r = rows[-1]
            print(f"  {kind:12s} {value:+5.2f} {model_kind:18s} RMSE {r['rmse_pct']:8.4f}% | "
                  f"PVR(meas) {r['pvr_measured_disch_eps0']:7.4f}% | PVR(true) {r['pvr_true_disch_eps0']:7.4f}% | "
                  f"rest drift {r['rest_drift_pct_per_hour']:+8.3f} %SOC/h")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sensor_fault_results.json").write_text(json.dumps({
        "provenance": provenance(args.scenario, "outputs/v7_final/*_scenario_A.pt", {
            "experiment": "sensor_fault_stress_test", "noise_seed": SEED,
            "fault_propagation": "I, V_proxy(I*R_int), dV_proxy/dt, dI/dt all recomputed in measured domain",
            "threshold_A": CURRENT_THRESHOLD_A,
        }),
        "results": results,
    }, indent=2, default=float))
    with (OUT_DIR / "sensor_fault_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved to {OUT_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
