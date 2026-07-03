"""
dataset_variant_stats.py -- Phase 1: compare v4/v5 dataset variants.

Sources:
  * metadata_v4.json of each built variant (window counts, soc_initial stats,
    q_actual, verification)
  * direct tensor comparison of labels vs v4 (test split): label shift caused
    by ohmic correction and/or decimation change
  * legacy audit artifacts for first-sample decimation defect rates
    (results/diagnostics/preprocessing_audit.json); mean-per-second modes have
    routing-conflict/envelope rates structurally equal to 0 because the kept
    current IS the intra-second mean.

Outputs:
  results/v5/dataset_variant_comparison.{csv,json}
  results/v5/figures/soc_initial_bias_by_temperature.png
  results/v5/figures/routing_conflict_by_decimation_mode.png
"""

from __future__ import annotations

import csv
import datetime
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA = BASE_DIR / "data" / "processed"
OUT = BASE_DIR / "results" / "v5"
FIG = OUT / "figures"
VARIANTS = ("v4", "v5a", "v5b", "v5c")
C1, C2, C_GRID, C_MUTED, C_INK = "#2a78d6", "#1baf7a", "#e1e0d9", "#898781", "#0b0b0b"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE_DIR).decode().strip()
    except Exception:
        return "unknown"


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=C_GRID, linewidth=0.6, axis="y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=C_MUTED, labelsize=8)


def main() -> None:
    rows, detail = [], {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
                        "git_commit": git_commit()}
    y_v4 = {s: np.load(DATA / f"v4_scenario_{s}" / "y_test.npy") for s in ("A", "B")}

    for variant in VARIANTS:
        for scen in ("A", "B"):
            d = DATA / f"{variant}_scenario_{scen}"
            meta_path = d / "metadata_v4.json"
            if not meta_path.exists():
                rows.append({"variant": variant, "scenario": scen, "status": "MISSING"})
                continue
            meta = json.loads(meta_path.read_text())
            counts = meta.get("split_window_counts", {})
            n_windows = {k: sum(v.values()) for k, v in counts.items()}
            seg_total = sum(t.get("segments", 0) for t in meta.get("profile_stats", {}).values())

            y_test = np.load(d / "y_test.npy")
            if y_test.shape == y_v4[scen].shape:
                dy = np.abs(y_test - y_v4[scen]) * 100
                label_shift = {"mean_abs_pct": round(float(dy.mean()), 4),
                               "p95_abs_pct": round(float(np.percentile(dy, 95)), 4),
                               "max_abs_pct": round(float(dy.max()), 4)}
            else:
                label_shift = {"note": f"shape changed {y_v4[scen].shape}->{y_test.shape}; "
                                        "window alignment differs, per-element shift undefined"}

            soc_init = meta.get("soc_initial_stats", {})
            row = {
                "variant": variant, "scenario": scen, "status": "OK",
                "label_mode": meta.get("label_mode", "legacy"),
                "decimation_mode": meta.get("decimation_mode", "first_sample"),
                "segments": seg_total,
                "train_windows": n_windows.get("train", 0),
                "val_windows": n_windows.get("val", 0),
                "test_windows": n_windows.get("test", 0),
                "label_shift_vs_v4_mean_pct": label_shift.get("mean_abs_pct", ""),
                "label_shift_vs_v4_max_pct": label_shift.get("max_abs_pct", ""),
                "leakage_overlaps": sum(meta.get("verification", {}).values()),
            }
            rows.append(row)
            detail[f"{variant}_scenario_{scen}"] = {
                "metadata": {k: meta.get(k) for k in
                             ("dataset_version", "label_mode", "decimation_mode", "q_actual",
                              "split_window_counts", "verification")},
                "soc_initial_stats": soc_init,
                "label_shift_vs_v4_test": label_shift,
            }

    # decimation defect rates: first_sample measured (phase-5 audit); mean modes structural 0
    audit = json.loads((BASE_DIR / "results" / "diagnostics" / "preprocessing_audit.json").read_text())
    temps = [t for t in ("0degC", "10degC", "25degC", "40degC", "n10degC", "n20degC") if t in audit]
    fs_conflict = [audit[t]["decimation"]["sign_conflict_pct"] for t in temps]
    fs_envelope = [audit[t]["decimation"]["envelope_exceeded_pct"] for t in temps]
    detail["decimation_defect_rates"] = {
        "first_sample": {"sign_conflict_pct_mean": round(float(np.mean(fs_conflict)), 3),
                         "envelope_exceeded_pct_mean": round(float(np.mean(fs_envelope)), 3)},
        "mean_per_second": {"sign_conflict_pct_mean": 0.0, "envelope_exceeded_pct_mean": 0.0,
                            "justification": "kept current equals the intra-second mean by construction"},
    }

    # figures
    FIG.mkdir(parents=True, exist_ok=True)
    seg = {}
    with (BASE_DIR / "results" / "diagnostics" / "segment_start_condition_report.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(r["loaded_start"]):
                bias = abs(float(r["soc_initial_ocv"]) - float(r["soc_initial_ocv_ohmic_corrected"])) * 100
                seg.setdefault(r["temp"], []).append(bias)
    t_order = [t for t in ("40degC", "25degC", "10degC", "0degC", "n10degC", "n20degC") if t in seg]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    vals = [float(np.mean(seg[t])) for t in t_order]
    ax.bar(range(len(t_order)), vals, color=C1, width=0.62)
    ax.set_xticks(range(len(t_order)), t_order, fontsize=8)
    ax.set_ylabel("removed ohmic anchor bias (%SOC)", fontsize=8, color=C_MUTED)
    ax.set_title("Loaded-start soc_initial bias corrected by v5 labels (mean per temperature)",
                 fontsize=9, color=C_INK)
    style(ax)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=7, color=C_INK)
    fig.tight_layout()
    fig.savefig(FIG / "soc_initial_bias_by_temperature.png", dpi=160, facecolor="#fcfcfb")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(2)
    ax.bar(x - 0.17, [float(np.mean(fs_conflict)), 0.0], width=0.3, color=C1, label="routing-sign conflict %")
    ax.bar(x + 0.17, [float(np.mean(fs_envelope)), 0.0], width=0.3, color=C2, label="envelope-unsatisfiable %")
    ax.set_xticks(x, ["first_sample (v4/v5a)", "mean_per_second (v5b/v5c)"], fontsize=8)
    ax.set_ylabel("% of multi-sample seconds", fontsize=8, color=C_MUTED)
    ax.set_title("Decimation defect rates by mode (mean over temperatures)", fontsize=9, color=C_INK)
    style(ax)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "routing_conflict_by_decimation_mode.png", dpi=160, facecolor="#fcfcfb")

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "dataset_variant_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT / "dataset_variant_comparison.json").write_text(json.dumps({"rows": rows, "detail": detail},
                                                                    indent=2, default=float))
    for r in rows:
        print(r)
    print(f"Saved to {OUT.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
