"""
compare_v4_v5.py -- Phase 2 aggregation: v5c headline results vs frozen v4.

Reads:
  results/v5/headline_models/runs_v5c_*.json   (neural, seed 42)
  results/v5/baselines/deterministic_baselines_v5c.json
  results/v5/legacy_freeze_manifest.json       (frozen v4 numbers)

Writes:
  results/v5/headline_models/v5_headline_model_comparison.{csv,json}
  results/v5/headline_models/v4_vs_v5_comparison.csv
  results/v5/figures/v4_vs_v5_rmse_by_model.png
  results/v5/figures/v5_temperature_breakdown.png
"""

from __future__ import annotations

import csv
import datetime
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "results" / "v5" / "headline_models"
FIG = BASE_DIR / "results" / "v5" / "figures"
C1, C2, C_MUTED, C_INK = "#2a78d6", "#1baf7a", "#898781", "#0b0b0b"

V4_KEYMAP = {
    "vanilla_lstm": "vanilla_lstm",
    "hard_coulomb_lstm": "hard_coulomb_lstm",
    "hard_coulomb_tcn": "hard_coulomb_tcn",
    "null[ocv25_qnom]": "null_ocv_coulomb_ocv25_qnom",
    "vanilla+posthoc_clamp": "vanilla_posthoc_clamp",
}


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color="#e1e0d9", linewidth=0.6, axis="y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=C_MUTED, labelsize=8)


def row_from_metrics(model, scenario, m, prov):
    reg = m["regression"]
    r = {"model": model, "scenario": scenario,
         "dataset_version": prov.get("dataset_version"), "label_mode": prov.get("label_mode"),
         "decimation_mode": prov.get("decimation_mode"), "seed": prov.get("seed", ""),
         "rmse_pct": round(reg["rmse_full_pct"], 4), "mae_pct": round(reg["mae_full_pct"], 4),
         "maxe_pct": round(reg["maxe_full_pct"], 4),
         "pvr_disch_eps0": round(m["pvr"]["discharge"]["by_epsilon"]["0"]["rate_pct"], 4),
         "pvr_disch_eps0.005": round(m["pvr"]["discharge"]["by_epsilon"]["0.005"]["rate_pct"], 4),
         "delta_ratio_disch": round(m["delta_magnitude"]["discharge"]["pred_true_delta_ratio"], 4)
         if m["delta_magnitude"]["discharge"].get("pred_true_delta_ratio") is not None else ""}
    for temp in ("n20degC", "n10degC", "40degC"):
        pt = m.get("per_temperature", {}).get(temp)
        if pt:
            r[f"rmse_{temp}"] = round(pt["regression"]["rmse_full_pct"], 4)
            r[f"maxe_{temp}"] = round(pt["regression"]["maxe_full_pct"], 4)
    return r


def main() -> None:
    rows = []
    for f in glob.glob(str(OUT / "runs_v5c_*.json")):
        for res in json.loads(Path(f).read_text()):
            rows.append(row_from_metrics(res["model"], res["provenance"]["scenario"],
                                         res["metrics"], res["provenance"]))
    det_path = BASE_DIR / "results" / "v5" / "baselines" / "deterministic_baselines_v5c.json"
    if det_path.exists():
        det = json.loads(det_path.read_text())
        for scen, models in det.items():
            for name, entry in models.items():
                if "metrics" in entry:
                    rows.append(row_from_metrics(name, scen, entry["metrics"], entry["provenance"]))

    rows.sort(key=lambda r: (r["scenario"], r["rmse_pct"]))
    with (OUT / "v5_headline_model_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("model", "scenario"), k))
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (OUT / "v5_headline_model_comparison.json").write_text(json.dumps(
        {"generated": datetime.datetime.now().isoformat(timespec="seconds"), "rows": rows},
        indent=2, default=float))

    # v4 vs v5 delta table
    v4 = json.loads((BASE_DIR / "results" / "v5" / "legacy_freeze_manifest.json").read_text())["key_v4_metrics"]
    cmp_rows = []
    for r in rows:
        scen = "scenario_A" if r["scenario"].endswith("A") else "scenario_B"
        key = V4_KEYMAP.get(r["model"].replace("null[ocv25_qnom]", "null[ocv25_qnom]"))
        v4m = v4.get(scen, {}).get(key) if key else None
        if not v4m:
            continue
        cmp_rows.append({
            "model": r["model"], "scenario": scen,
            "v4_rmse_pct": v4m.get("rmse_pct"), "v5c_rmse_pct": r["rmse_pct"],
            "rmse_delta_pp": round(r["rmse_pct"] - v4m.get("rmse_pct", np.nan), 4),
            "v4_maxe_pct": v4m.get("maxe_pct", ""), "v5c_maxe_pct": r["maxe_pct"],
            "v4_n20_rmse": v4m.get("n20_rmse_pct", ""), "v5c_n20_rmse": r.get("rmse_n20degC", ""),
        })
    with (OUT / "v4_vs_v5_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cmp_rows[0].keys()))
        w.writeheader()
        w.writerows(cmp_rows)

    # figures
    FIG.mkdir(parents=True, exist_ok=True)
    ca = [c for c in cmp_rows if c["scenario"] == "scenario_A"]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    x = np.arange(len(ca))
    ax.bar(x - 0.18, [c["v4_rmse_pct"] for c in ca], width=0.34, color=C1, label="v4 legacy")
    ax.bar(x + 0.18, [c["v5c_rmse_pct"] for c in ca], width=0.34, color=C2, label="v5c corrected")
    ax.set_xticks(x, [c["model"].replace("+", "\n+") for c in ca], fontsize=7)
    ax.set_ylabel("RMSE (%SOC)", fontsize=8, color=C_MUTED)
    ax.set_title("Scenario A: v4 vs v5c RMSE by model (seed 42)", fontsize=9, color=C_INK)
    style(ax)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "v4_vs_v5_rmse_by_model.png", dpi=160, facecolor="#fcfcfb")

    neural_a = [r for r in rows if r["scenario"].endswith("A") and f"rmse_n20degC" in r]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    x = np.arange(len(neural_a))
    ax.bar(x - 0.18, [r.get("rmse_n20degC", np.nan) for r in neural_a], width=0.34, color=C1, label="−20 °C")
    ax.bar(x + 0.18, [r.get("rmse_40degC", np.nan) for r in neural_a], width=0.34, color=C2, label="40 °C")
    ax.set_xticks(x, [r["model"].replace("_", "\n") for r in neural_a], fontsize=6.5)
    ax.set_ylabel("RMSE (%SOC)", fontsize=8, color=C_MUTED)
    ax.set_title("v5c Scenario A: per-temperature RMSE (seed 42)", fontsize=9, color=C_INK)
    style(ax)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "v5_temperature_breakdown.png", dpi=160, facecolor="#fcfcfb")

    for r in rows:
        print(f"  {r['scenario']:12s} {r['model']:26s} RMSE {r['rmse_pct']:8.4f} | MaxE {r['maxe_pct']:8.4f} | "
              f"n20 {r.get('rmse_n20degC','-')}")
    print("\n  v4 -> v5c deltas:")
    for c in cmp_rows:
        print(f"  {c['scenario']:12s} {c['model']:26s} {c['v4_rmse_pct']} -> {c['v5c_rmse_pct']} "
              f"({c['rmse_delta_pp']:+.2f} pp)")


if __name__ == "__main__":
    main()
