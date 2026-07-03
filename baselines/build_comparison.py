"""
build_comparison.py -- Phase 2 unified baseline comparison table.

Rows: vanilla LSTM, vanilla+post-hoc clamp, null OCV+Coulomb (all variants),
Hard-Coulomb LSTM, Hard-Coulomb TCN (sprint52 checkpoint, if present).
One row per (model, scenario, temperature-condition) with regression,
region/epsilon PVR, and delta-magnitude columns.

Outputs:
  results/baselines/baseline_comparison_results.json
  results/baselines/baseline_comparison_table.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "src"), str(BASE_DIR / "analysis"), str(BASE_DIR / "baselines")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import PVR_EPSILONS  # noqa: E402
from null_ocv_coulomb import VARIANTS as NULL_VARIANTS  # noqa: E402
from predict_utils import load_test_bundle, predict_checkpoint, provenance  # noqa: E402
from soc_metrics import evaluate_soc_predictions  # noqa: E402

RESULTS_DIR = BASE_DIR / "results" / "baselines"
SCENARIOS = ("scenario_A", "scenario_B")
PARAM_COUNTS = {"vanilla_lstm": 53569, "hard_coulomb_lstm": 54626}


def flatten_rows(name: str, scenario: str, bundle: Dict, params) -> List[Dict]:
    """Bundle -> one ALL row + one row per temperature."""

    def one(cond: str, m: Dict, n_windows: int) -> Dict:
        reg, pvr, dm = m["regression"], m["pvr"], m["delta_magnitude"]
        row = {
            "model": name,
            "scenario": scenario,
            "temp_condition": cond,
            "n_windows": n_windows,
            "params": params if params is not None else "",
            "mae_pct": round(reg["mae_full_pct"], 4),
            "rmse_pct": round(reg["rmse_full_pct"], 4),
            "maxe_pct": round(reg["maxe_full_pct"], 4),
        }
        for region in ("discharge", "charge", "rest"):
            by_eps = pvr[region]["by_epsilon"]
            row[f"pvr_{region}_eps0"] = round(by_eps["0"]["rate_pct"], 4)
        row["pvr_total_eps0"] = round(pvr["total"]["by_epsilon"]["0"]["rate_pct"], 4)
        for eps in PVR_EPSILONS:
            row[f"pvr_discharge_eps{eps:g}"] = round(
                pvr["discharge"]["by_epsilon"][f"{eps:g}"]["rate_pct"], 4
            )
        d_all, d_dis = dm["all"], dm["discharge"]
        row["delta_soc_mae"] = f"{d_all['delta_soc_mae']:.3e}" if d_all.get("delta_soc_mae") is not None else ""
        ratio = d_dis.get("pred_true_delta_ratio")
        row["pred_true_delta_ratio_discharge"] = round(ratio, 4) if ratio is not None else ""
        return row

    rows = [one("ALL", bundle, bundle["n_windows"])]
    for temp, m in bundle.get("per_temperature", {}).items():
        rows.append(one(temp, m, m["n_windows"]))
    return rows


def predict_hc_tcn(scenario: str) -> np.ndarray | None:
    ckpt_path = BASE_DIR / "outputs" / "v8_tcn_redemption" / "sprint52" / f"hard_coulomb_tcn_{scenario}.pt"
    if not ckpt_path.exists():
        return None
    from model_v5_coulomb_tcn import HardCoulombTCN  # noqa: E402

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HardCoulombTCN(
        num_inputs=cfg["num_inputs"],
        num_filters=cfg["num_filters"],
        kernel_size=cfg["kernel_size"],
        dropout=cfg["dropout"],
        dilation_rates=cfg["dilation_rates"],
        safety_factor=cfg["safety_factor"],
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    b = load_test_bundle(scenario)
    preds = []
    with torch.no_grad():
        for i in range(0, len(b.X), 1024):
            xb = torch.from_numpy(b.X[i : i + 1024]).to(device)
            ib = torch.from_numpy(b.I[i : i + 1024]).to(device)
            out = model(xb, ib)
            preds.append(out.cpu().numpy())
    y = np.concatenate(preds, axis=0)
    return y.squeeze(-1) if y.ndim == 3 else y


def main() -> None:
    from null_ocv_coulomb import run_scenario as run_null  # reuse exact implementation
    from posthoc_clamp import apply_hard_coulomb_clamp

    all_rows: List[Dict] = []
    all_json: List[Dict] = []

    for scenario in SCENARIOS:
        b = load_test_bundle(scenario)

        # learned models
        for kind in ("vanilla_lstm", "hard_coulomb_lstm"):
            run = predict_checkpoint(scenario, kind)
            m = evaluate_soc_predictions(b.y_true, run["y_pred"], b.I, b.temp_labels)
            all_rows += flatten_rows(kind, scenario, m, PARAM_COUNTS[kind])
            all_json.append({"model": kind, "scenario": scenario,
                             "provenance": provenance(scenario, run["checkpoint"], {"model": kind}),
                             "metrics": m})
            if kind == "vanilla_lstm":
                y_c = apply_hard_coulomb_clamp(run["y_pred"], b.I)
                mc = evaluate_soc_predictions(b.y_true, y_c, b.I, b.temp_labels)
                all_rows += flatten_rows("vanilla_lstm+posthoc_clamp", scenario, mc, PARAM_COUNTS[kind])
                all_json.append({"model": "vanilla_lstm+posthoc_clamp", "scenario": scenario,
                                 "provenance": provenance(scenario, run["checkpoint"],
                                                          {"model": "vanilla+clamp", "eta": 1.5}),
                                 "metrics": mc})

        # null baselines (reuse saved predictions if present, else recompute)
        npz_path = RESULTS_DIR / f"null_ocv_coulomb_predictions_{scenario}.npz"
        if npz_path.exists():
            preds = np.load(npz_path)
            null_preds = {k: preds[k] for k in preds.files}
        else:
            run_null(scenario)
            preds = np.load(npz_path)
            null_preds = {k: preds[k] for k in preds.files}
        for variant in NULL_VARIANTS:
            m = evaluate_soc_predictions(b.y_true, null_preds[variant], b.I, b.temp_labels)
            all_rows += flatten_rows(f"null_ocv_coulomb[{variant}]", scenario, m, 0)
            all_json.append({"model": f"null_ocv_coulomb[{variant}]", "scenario": scenario,
                             "provenance": provenance(scenario, None, {"model": "null", "variant": variant}),
                             "metrics": m})

        # Hard-Coulomb TCN from sprint52, if available
        try:
            y_tcn = predict_hc_tcn(scenario)
        except Exception as exc:  # loader mismatch -> report as blocker, don't fake
            print(f"  [BLOCKER] HC-TCN {scenario}: {exc}")
            y_tcn = None
        if y_tcn is not None:
            m = evaluate_soc_predictions(b.y_true, y_tcn, b.I, b.temp_labels)
            all_rows += flatten_rows("hard_coulomb_tcn", scenario, m, 208546)
            all_json.append({"model": "hard_coulomb_tcn", "scenario": scenario,
                             "provenance": provenance(scenario,
                                                      f"outputs/v8_tcn_redemption/sprint52/hard_coulomb_tcn_{scenario}.pt",
                                                      {"model": "hard_coulomb_tcn"}),
                             "metrics": m})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "baseline_comparison_table.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    (RESULTS_DIR / "baseline_comparison_results.json").write_text(
        json.dumps(all_json, indent=2, default=float)
    )

    for row in all_rows:
        if row["temp_condition"] == "ALL":
            print(f"  {row['scenario']:10s} {row['model']:32s} RMSE {row['rmse_pct']:8.4f}% | "
                  f"MaxE {row['maxe_pct']:8.4f}% | PVRd0 {row['pvr_discharge_eps0']:7.3f}% | "
                  f"ratio {row['pred_true_delta_ratio_discharge']}")
    print(f"\nSaved comparison to {RESULTS_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
