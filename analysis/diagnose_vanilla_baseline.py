"""
diagnose_vanilla_baseline.py -- Phase 3: is the vanilla baseline broken or fair?

Produces:
  results/diagnostics/vanilla_range_diagnostics.csv
  results/diagnostics/vanilla_delta_histogram.png
  results/diagnostics/vanilla_prediction_examples.png
  results/diagnostics/vanilla_mechanism_probe.json

Mechanism probe: correlates the vanilla model's per-step predicted delta-SOC
with each per-step input feature. If predicted deltas track instantaneous
voltage/current dynamics (which fluctuate under load while true SOC does not),
the oscillation is a modeling behavior, not a data or shape bug.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "src"), str(BASE_DIR / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

from predict_utils import predict_checkpoint, provenance  # noqa: E402

OUT_DIR = BASE_DIR / "results" / "diagnostics"
FIG_DIR = BASE_DIR / "results" / "figures"

# dataviz reference palette (validated): slot1 blue = truth, slot2 aqua = prediction
C_TRUE, C_PRED, C_GRID, C_MUTED, C_INK = "#2a78d6", "#1baf7a", "#e1e0d9", "#898781", "#0b0b0b"
FEATURES = ["V_proxy", "Current", "Temperature", "dV_proxy_dt", "dI_dt"]


def style_axis(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=C_GRID, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors=C_MUTED, labelsize=8)


def main() -> None:
    rows, probe = [], {}
    fig_h, axes_h = plt.subplots(1, 2, figsize=(10, 3.6))
    fig_e, axes_e = plt.subplots(2, 3, figsize=(12, 5.6), sharey=False)

    for col, scenario in enumerate(("scenario_A", "scenario_B")):
        run = predict_checkpoint(scenario, "vanilla_lstm")
        b = run["bundle"]
        y_pred, y_true = run["y_pred"], b.y_true

        pred_range = y_pred.max(axis=1) - y_pred.min(axis=1)
        true_range = y_true.max(axis=1) - y_true.min(axis=1)
        d_pred = np.diff(y_pred, axis=1)
        d_true = np.diff(y_true, axis=1)

        for name, arr in (("pred_window_range", pred_range), ("true_window_range", true_range)):
            rows.append({
                "scenario": scenario, "quantity": name,
                "mean": round(float(arr.mean()), 5), "median": round(float(np.median(arr)), 5),
                "p90": round(float(np.percentile(arr, 90)), 5),
                "p99": round(float(np.percentile(arr, 99)), 5),
                "max": round(float(arr.max()), 5),
            })
        rows.append({
            "scenario": scenario, "quantity": "range_ratio_pred_over_true",
            "mean": round(float(pred_range.mean() / max(true_range.mean(), 1e-12)), 3),
            "median": "", "p90": "", "p99": "", "max": "",
        })

        # mechanism probe: corr(pred delta, per-step feature deltas/values)
        corr = {}
        for f_idx, f_name in enumerate(FEATURES):
            feat_step = b.X[:, 1:, f_idx].reshape(-1).astype(np.float64)
            corr[f"corr_delta_pred__{f_name}"] = round(
                float(np.corrcoef(d_pred.reshape(-1).astype(np.float64), feat_step)[0, 1]), 4
            )
        corr["corr_delta_pred__delta_true"] = round(
            float(np.corrcoef(d_pred.reshape(-1), d_true.reshape(-1))[0, 1]), 4
        )
        # lag-1 autocorrelation of predicted deltas: strong negative = zig-zag
        dp = d_pred.astype(np.float64)
        a, c = dp[:, 1:].reshape(-1), dp[:, :-1].reshape(-1)
        corr["lag1_autocorr_delta_pred"] = round(float(np.corrcoef(a, c)[0, 1]), 4)
        probe[scenario] = corr

        # delta histogram (log-y): predicted vs true per-step deltas
        ax = axes_h[col]
        bins = np.linspace(-0.03, 0.03, 121)
        ax.hist(d_true.reshape(-1), bins=bins, color=C_TRUE, alpha=0.85, label="true ΔSOC/step")
        ax.hist(d_pred.reshape(-1), bins=bins, color=C_PRED, alpha=0.55, label="vanilla predicted ΔSOC/step")
        ax.set_yscale("log")
        ax.set_title(f"{scenario} — per-step ΔSOC distribution", fontsize=9, color=C_INK)
        ax.set_xlabel("ΔSOC per 1 s step", fontsize=8, color=C_MUTED)
        style_axis(ax)
        ax.legend(fontsize=7, frameon=False)

        # example trajectories: median-error, p99-error, worst-error windows
        win_err = np.abs(y_pred - y_true).max(axis=1)
        order = np.argsort(win_err)
        picks = {
            "median-error window": order[len(order) // 2],
            "p99-error window": order[int(len(order) * 0.99)],
            "worst window": order[-1],
        }
        for k, (label, idx) in enumerate(picks.items()):
            ax = axes_e[col, k]
            t = np.arange(y_true.shape[1])
            ax.plot(t, y_true[idx], color=C_TRUE, linewidth=1.8, label="true SOC")
            ax.plot(t, y_pred[idx], color=C_PRED, linewidth=1.4, label="vanilla prediction")
            temp = b.temp_labels[idx] if b.temp_labels is not None else "?"
            ax.set_title(f"{scenario} · {label} · {temp}", fontsize=8, color=C_INK)
            ax.set_xlabel("t (s)", fontsize=7, color=C_MUTED)
            if k == 0:
                ax.set_ylabel("SOC", fontsize=8, color=C_MUTED)
            style_axis(ax)
            if col == 0 and k == 0:
                ax.legend(fontsize=7, frameon=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with (OUT_DIR / "vanilla_range_diagnostics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (OUT_DIR / "vanilla_mechanism_probe.json").write_text(json.dumps({
        "provenance": provenance("scenario_A", "outputs/v7_final/vanilla_lstm_*.pt",
                                 {"model": "vanilla_lstm", "note": "both scenarios probed"}),
        "correlations": probe,
    }, indent=2))

    fig_h.tight_layout()
    fig_h.savefig(FIG_DIR / "vanilla_delta_histogram.png", dpi=160, facecolor="#fcfcfb")
    fig_e.tight_layout()
    fig_e.savefig(FIG_DIR / "vanilla_prediction_examples.png", dpi=160, facecolor="#fcfcfb")

    print(json.dumps(probe, indent=1))
    for r in rows:
        print(r)
    print(f"Saved diagnostics to {OUT_DIR.relative_to(BASE_DIR)} and figures to {FIG_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
