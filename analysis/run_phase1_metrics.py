"""
run_phase1_metrics.py -- Phase 1 deliverable runner.

Evaluates all finalized v7 checkpoints with the audited metrics layer and
writes:
  results/metrics/pvr_deadband_results.json
  results/metrics/delta_magnitude_results.json
  results/metrics/phase1_summary.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict_utils import BASE_DIR, predict_checkpoint, provenance  # noqa: E402
from soc_metrics import evaluate_soc_predictions  # noqa: E402

RESULTS_DIR = BASE_DIR / "results" / "metrics"
MODEL_KINDS = ("vanilla_lstm", "hard_coulomb_lstm")
SCENARIOS = ("scenario_A", "scenario_B")


def main() -> None:
    pvr_out, delta_out, csv_rows = [], [], []
    for scenario in SCENARIOS:
        for kind in MODEL_KINDS:
            run = predict_checkpoint(scenario, kind)
            b = run["bundle"]
            bundle = evaluate_soc_predictions(b.y_true, run["y_pred"], b.I, b.temp_labels)
            prov = provenance(scenario, run["checkpoint"], {"model_kind": kind})

            pvr_out.append({"provenance": prov, "regression": bundle["regression"],
                            "pvr": bundle["pvr"], "legacy": bundle["legacy"],
                            "per_temperature": {
                                t: v["pvr"] for t, v in bundle.get("per_temperature", {}).items()
                            }})
            delta_out.append({"provenance": prov, "delta_magnitude": bundle["delta_magnitude"],
                              "per_temperature": {
                                  t: v["delta_magnitude"] for t, v in bundle.get("per_temperature", {}).items()
                              }})

            reg, pvr, dm = bundle["regression"], bundle["pvr"], bundle["delta_magnitude"]
            row = {
                "scenario": scenario,
                "model": kind,
                "rmse_full_pct": round(reg["rmse_full_pct"], 4),
                "mae_full_pct": round(reg["mae_full_pct"], 4),
                "maxe_full_pct": round(reg["maxe_full_pct"], 4),
            }
            for region in ("discharge", "charge", "rest"):
                for eps, vals in pvr[region]["by_epsilon"].items():
                    row[f"pvr_{region}_eps{eps}"] = round(vals["rate_pct"], 4)
            row["delta_ratio_discharge"] = (
                round(dm["discharge"]["pred_true_delta_ratio"], 4)
                if dm["discharge"].get("pred_true_delta_ratio") is not None else ""
            )
            row["delta_mae_all"] = f"{dm['all']['delta_soc_mae']:.3e}"
            csv_rows.append(row)
            print(f"{scenario} / {kind}: RMSE {row['rmse_full_pct']}% | "
                  f"PVR(disch,eps0) {row['pvr_discharge_eps0']}% | "
                  f"PVR(disch,eps0.005) {row['pvr_discharge_eps0.005']}% | "
                  f"delta ratio(disch) {row['delta_ratio_discharge']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "pvr_deadband_results.json").write_text(json.dumps(pvr_out, indent=2))
    (RESULTS_DIR / "delta_magnitude_results.json").write_text(json.dumps(delta_out, indent=2))
    with (RESULTS_DIR / "phase1_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved to {RESULTS_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
