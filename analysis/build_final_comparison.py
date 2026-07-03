"""
build_final_comparison.py -- Phase 10: aggregate every model/baseline into one
machine-readable comparison (results/final_model_comparison.{csv,json}).

Pulls from the phase result files (no re-inference), plus per-seed averaging
for the anchor_last/pooled variants.
"""

from __future__ import annotations

import csv
import datetime
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
RES = BASE_DIR / "results"


def load_json(p: Path):
    return json.loads(p.read_text())


def main() -> None:
    rows = []

    # 1. baseline comparison table (vanilla, clamp, null variants, HC-LSTM, HC-TCN)
    with (RES / "baselines" / "baseline_comparison_table.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["temp_condition"] != "ALL":
                continue
            rows.append({
                "model": r["model"], "scenario": r["scenario"], "source": "phase2",
                "params": r["params"],
                "rmse_pct": float(r["rmse_pct"]), "mae_pct": float(r["mae_pct"]),
                "maxe_pct": float(r["maxe_pct"]),
                "pvr_disch_eps0": float(r["pvr_discharge_eps0"]),
                "pvr_disch_eps0.005": float(r["pvr_discharge_eps0.005"]),
                "delta_ratio_disch": r["pred_true_delta_ratio_discharge"],
                "notes": "",
            })

    # 2. anchor variants (Scenario A)
    with (RES / "model_variants" / "anchor_variant_comparison.csv").open(encoding="utf-8") as fh:
        variant_rows = list(csv.DictReader(fh))
    for name in ("HC_LSTM_anchor_last", "HC_LSTM_anchor_pooled"):
        seeds = [r for r in variant_rows if r["variant"] == name]
        if not seeds:
            continue
        rmse = np.array([float(r["rmse_pct"]) for r in seeds])
        maxe = np.array([float(r["maxe_pct"]) for r in seeds])
        rows.append({
            "model": name, "scenario": "scenario_A", "source": "phase6",
            "params": 54626,
            "rmse_pct": round(float(rmse.mean()), 4), "mae_pct": "",
            "maxe_pct": round(float(maxe.mean()), 4),
            "pvr_disch_eps0": 0.0, "pvr_disch_eps0.005": 0.0, "delta_ratio_disch": "",
            "notes": f"mean of {len(seeds)} seeds; rmse std {rmse.std(ddof=1):.2f}, maxe std {maxe.std(ddof=1):.2f}",
        })
    for r in variant_rows:
        if r["variant"] == "HC_LSTM_recursive_infer(prod)":
            rows.append({
                "model": "HC_LSTM_recursive_infer", "scenario": r["scenario"], "source": "phase6",
                "params": 54626,
                "rmse_pct": float(r["rmse_pct"]), "mae_pct": float(r["mae_pct"]),
                "maxe_pct": float(r["maxe_pct"]),
                "pvr_disch_eps0": float(r["pvr_discharge_eps0"]),
                "pvr_disch_eps0.005": "", "delta_ratio_disch": "",
                "notes": "carried-anchor stitched inference on production checkpoint; "
                         f"n20 RMSE {r.get('rmse_n20degC','-')}%, n20 MaxE {r.get('maxe_n20degC','-')}%",
            })

    # 3. EKF rows
    with (RES / "baselines" / "ekf_comparison_table.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "model": f"EKF_ocv_rint[R={float(r['R_V2']):g}]", "scenario": r["scenario"],
                "source": "phase8", "params": 0,
                "rmse_pct": float(r["rmse_pct"]), "mae_pct": float(r["mae_pct"]),
                "maxe_pct": float(r["maxe_pct"]),
                "pvr_disch_eps0": float(r["pvr_disch_eps0"]),
                "pvr_disch_eps0.005": float(r["pvr_disch_eps0.005"]),
                "delta_ratio_disch": "",
                "notes": f"n20 RMSE {r.get('rmse_n20degC','-')}%",
            })

    # 4. oracle-anchor HC reference (upper bound for the learned delta path)
    oracle = load_json(RES / "diagnostics" / "oracle_anchor_results.json")
    for scen, entry in oracle.items():
        reg = entry["oracle"]["regression"]
        rows.append({
            "model": "HC_LSTM_oracle_anchor(ref)", "scenario": scen, "source": "phase4",
            "params": 54626,
            "rmse_pct": round(reg["rmse_full_pct"], 4), "mae_pct": round(reg["mae_full_pct"], 4),
            "maxe_pct": round(reg["maxe_full_pct"], 4),
            "pvr_disch_eps0": 0.0, "pvr_disch_eps0.005": 0.0, "delta_ratio_disch": "",
            "notes": "diagnostic reference (true SOC at t=0), not deployable",
        })

    rows.sort(key=lambda r: (r["scenario"], r["rmse_pct"] if isinstance(r["rmse_pct"], float) else 999))
    out_csv = RES / "final_model_comparison.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (RES / "final_model_comparison.json").write_text(json.dumps({
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "note": "aggregated from phase 2/4/6/8 result files; per-temperature detail in the source files",
        "rows": rows,
    }, indent=2, default=float))

    for r in rows:
        print(f"  {r['scenario']:10s} {r['model']:34s} RMSE {r['rmse_pct']:>8} | MaxE {r['maxe_pct']:>8} | "
              f"PVRd0 {r['pvr_disch_eps0']:>7}")
    print(f"\nSaved {out_csv.relative_to(BASE_DIR)} and .json")


if __name__ == "__main__":
    main()
