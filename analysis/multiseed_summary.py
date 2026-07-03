"""
multiseed_summary.py -- Phase 3 aggregation.

Reads results/v5/multiseed/runs_{A,B}.json (5 seeds per model), emits
seed-level and aggregated tables + boxplots, and checks ranking stability.
"""

from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MS = BASE_DIR / "results" / "v5" / "multiseed"
FIG = BASE_DIR / "results" / "v5" / "figures"
C1, C_MUTED, C_INK = "#2a78d6", "#898781", "#0b0b0b"


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color="#e1e0d9", linewidth=0.6, axis="y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=C_MUTED, labelsize=8)


def main() -> None:
    seed_rows = []
    for f in ("runs_A.json", "runs_B.json"):
        p = MS / f
        if not p.exists():
            continue
        for res in json.loads(p.read_text()):
            prov, m = res["provenance"], res["metrics"]
            reg = m["regression"]
            row = {"model": res["model"], "scenario": prov["scenario"], "seed": prov["seed"],
                   "dataset_version": prov["dataset_version"],
                   "rmse_pct": round(reg["rmse_full_pct"], 4),
                   "mae_pct": round(reg["mae_full_pct"], 4),
                   "maxe_pct": round(reg["maxe_full_pct"], 4),
                   "pvr_disch_eps0": round(m["pvr"]["discharge"]["by_epsilon"]["0"]["rate_pct"], 4)}
            for temp in ("n20degC", "40degC"):
                pt = m.get("per_temperature", {}).get(temp)
                if pt:
                    row[f"rmse_{temp}"] = round(pt["regression"]["rmse_full_pct"], 4)
            seed_rows.append(row)

    MS.mkdir(parents=True, exist_ok=True)
    with (MS / "seed_level_results.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = sorted({k for r in seed_rows for k in r}, key=lambda k: (k not in ("model", "scenario", "seed"), k))
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(seed_rows)
    (MS / "seed_level_results.json").write_text(json.dumps(
        {"generated": datetime.datetime.now().isoformat(timespec="seconds"), "rows": seed_rows},
        indent=2, default=float))

    # aggregate
    agg_rows = []
    groups = {}
    for r in seed_rows:
        groups.setdefault((r["scenario"], r["model"]), []).append(r)
    for (scen, model), rs in sorted(groups.items()):
        def stats(key):
            v = np.array([r[key] for r in rs if key in r], dtype=float)
            if v.size == 0:
                return {}
            ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            return {f"{key}_mean": round(float(v.mean()), 4), f"{key}_std": round(float(v.std(ddof=1)), 4),
                    f"{key}_min": round(float(v.min()), 4), f"{key}_max": round(float(v.max()), 4),
                    f"{key}_ci95": round(float(ci), 4)}
        row = {"scenario": scen, "model": model, "n_seeds": len(rs)}
        for key in ("rmse_pct", "maxe_pct", "rmse_n20degC", "rmse_40degC"):
            row.update(stats(key))
        agg_rows.append(row)
    with (MS / "multiseed_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = sorted({k for r in agg_rows for k in r}, key=lambda k: (k not in ("scenario", "model", "n_seeds"), k))
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(agg_rows)

    # ranking stability: per seed, rank models by RMSE within scenario
    stability = {}
    for scen in sorted({r["scenario"] for r in seed_rows}):
        per_seed_rank = {}
        seeds = sorted({r["seed"] for r in seed_rows if r["scenario"] == scen})
        for seed in seeds:
            rs = sorted([r for r in seed_rows if r["scenario"] == scen and r["seed"] == seed],
                        key=lambda r: r["rmse_pct"])
            per_seed_rank[seed] = [r["model"] for r in rs]
        winners = [v[0] for v in per_seed_rank.values()]
        stability[scen] = {"winner_by_seed": {str(k): v[0] for k, v in per_seed_rank.items()},
                           "winner_stable": len(set(winners)) == 1}
    (MS / "ranking_stability.json").write_text(json.dumps(stability, indent=2))

    # boxplots
    FIG.mkdir(parents=True, exist_ok=True)
    for key, fname, label in (("rmse_pct", "multiseed_rmse_boxplot.png", "RMSE (%SOC)"),
                              ("maxe_pct", "multiseed_maxe_boxplot.png", "MaxE (%SOC)")):
        scens = sorted({r["scenario"] for r in seed_rows})
        fig, axes = plt.subplots(1, len(scens), figsize=(6.0 * len(scens), 3.8))
        if len(scens) == 1:
            axes = [axes]
        for ax, scen in zip(axes, scens):
            models = sorted({r["model"] for r in seed_rows if r["scenario"] == scen})
            data = [[r[key] for r in seed_rows if r["scenario"] == scen and r["model"] == m] for m in models]
            bp = ax.boxplot(data, tick_labels=[m.replace("_", "\n") for m in models], widths=0.5,
                            patch_artist=True)
            for box in bp["boxes"]:
                box.set(facecolor="#cde2fb", edgecolor=C1, linewidth=1.2)
            for med in bp["medians"]:
                med.set(color=C1, linewidth=1.6)
            ax.set_title(f"{scen} (5 seeds)", fontsize=9, color=C_INK)
            ax.set_ylabel(label, fontsize=8, color=C_MUTED)
            ax.tick_params(axis="x", labelsize=6.5)
            style(ax)
        fig.tight_layout()
        fig.savefig(FIG / fname, dpi=160, facecolor="#fcfcfb")

    for r in agg_rows:
        print(f"  {r['scenario']:12s} {r['model']:20s} RMSE {r.get('rmse_pct_mean')}±{r.get('rmse_pct_std')} "
              f"[{r.get('rmse_pct_min')},{r.get('rmse_pct_max')}] | MaxE {r.get('maxe_pct_mean')}±{r.get('maxe_pct_std')}")
    print(json.dumps(stability, indent=1))


if __name__ == "__main__":
    main()
