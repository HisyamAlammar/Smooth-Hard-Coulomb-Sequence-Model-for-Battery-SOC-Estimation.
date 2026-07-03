"""
anchor_error_analysis.py -- Phase 4: where does the anchor fail?

Current anchor mechanism (documented, not changed):
    model_v5_coulomb.HardCoulombLSTM.forward
        anchor_logit = anchor_head(h[:, 0, :])   # hidden state after ONE timestep
        soc_anchor   = lo + (hi - lo) * sigmoid(anchor_logit)
The anchor therefore sees a single (V_proxy, I, T, dVp/dt, dI/dt) sample.

Outputs:
  results/diagnostics/anchor_error_by_temperature.csv
  results/figures/anchor_error_by_temperature.png
  results/figures/anchor_error_vs_current_start.png
  (JSON bundle with rest/load split and voltage binning)
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
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "src"), str(BASE_DIR / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import CURRENT_THRESHOLD_A  # noqa: E402
from predict_utils import load_test_bundle, provenance, unscale_feature  # noqa: E402
from sprint48_common import checkpoint_path, load_checkpoint, resolve_device  # noqa: E402

OUT_DIR = BASE_DIR / "results" / "diagnostics"
FIG_DIR = BASE_DIR / "results" / "figures"
C1, C2, C_GRID, C_MUTED, C_INK = "#2a78d6", "#1baf7a", "#e1e0d9", "#898781", "#0b0b0b"
TEMP_ORDER = ["40degC", "25degC", "10degC", "0degC", "n10degC", "n20degC"]


def style_axis(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=C_GRID, linewidth=0.6, axis="y")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=C_MUTED, labelsize=8)


def raw_anchor(model, X: np.ndarray, device, batch_size=1024) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(device)
            h, _ = model.lstm(xb)
            outs.append(torch.sigmoid(model.anchor_head(h[:, 0, :])).squeeze(-1).cpu().numpy())
    return np.concatenate(outs, axis=0)


def summarize(err: np.ndarray) -> dict:
    e = err.astype(np.float64) * 100.0
    return {
        "mae_pct": round(float(e.mean()), 4),
        "rmse_pct": round(float(np.sqrt(np.mean(e**2))), 4),
        "p95_pct": round(float(np.percentile(e, 95)), 4),
        "maxe_pct": round(float(e.max()), 4),
        "n": int(e.size),
    }


def main() -> None:
    device = resolve_device(None)
    rows, json_out = [], {}
    fig_t, axes_t = plt.subplots(1, 2, figsize=(10, 3.6))
    fig_c, axes_c = plt.subplots(1, 2, figsize=(10, 3.6))

    for col, scenario in enumerate(("scenario_A", "scenario_B")):
        b = load_test_bundle(scenario)
        ckpt = checkpoint_path("hard_coulomb_lstm", scenario, latest=False)
        model, _ = load_checkpoint(ckpt, device)
        model.eval()

        anchor = raw_anchor(model, b.X, device)
        err = np.abs(anchor - b.y_true[:, 0])
        I0 = np.abs(b.I[:, 0])
        v0 = unscale_feature(b.X, 0)[:, 0]
        rest0 = I0 <= CURRENT_THRESHOLD_A

        scen_json = {
            "overall": summarize(err),
            "rest_start": summarize(err[rest0]) if rest0.any() else None,
            "load_start": summarize(err[~rest0]) if (~rest0).any() else None,
        }

        # per temperature
        temps_here = [t for t in TEMP_ORDER if b.temp_labels is not None and (b.temp_labels == t).any()]
        maes = []
        for temp in temps_here:
            m = b.temp_labels == temp
            s = summarize(err[m])
            s_rest = summarize(err[m & rest0]) if (m & rest0).any() else None
            s_load = summarize(err[m & ~rest0]) if (m & ~rest0).any() else None
            rows.append({"scenario": scenario, "temp": temp, **s,
                         "mae_rest_start_pct": s_rest["mae_pct"] if s_rest else "",
                         "mae_load_start_pct": s_load["mae_pct"] if s_load else "",
                         "n_rest_start": s_rest["n"] if s_rest else 0})
            maes.append(s["mae_pct"])
        scen_json["per_temperature"] = {t: r for t, r in zip(temps_here, maes)}

        ax = axes_t[col]
        ax.bar(range(len(temps_here)), maes, color=C1, width=0.62)
        ax.set_xticks(range(len(temps_here)), temps_here, fontsize=8)
        ax.set_title(f"{scenario} — raw anchor-head MAE by temperature", fontsize=9, color=C_INK)
        ax.set_ylabel("anchor MAE (%SOC)", fontsize=8, color=C_MUTED)
        style_axis(ax)
        for i, v in enumerate(maes):
            ax.text(i, v + 0.15, f"{v:.1f}", ha="center", fontsize=7, color=C_INK)

        # anchor error vs |I| at window start (binned)
        bins = np.array([0.0, CURRENT_THRESHOLD_A, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0])
        centers, mae_bin, v_mae_bin = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (I0 >= lo) & (I0 < hi)
            if m.sum() < 20:
                continue
            centers.append((lo + hi) / 2)
            mae_bin.append(float(err[m].mean() * 100))
        ax = axes_c[col]
        ax.plot(centers, mae_bin, color=C1, linewidth=2.0, marker="o", markersize=5,
                label="anchor MAE")
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_title(f"{scenario} — anchor MAE vs |I| at window start", fontsize=9, color=C_INK)
        ax.set_xlabel("|I(t=0)| (A, symlog)", fontsize=8, color=C_MUTED)
        ax.set_ylabel("anchor MAE (%SOC)", fontsize=8, color=C_MUTED)
        style_axis(ax)
        scen_json["mae_by_I0_bin"] = {f"{c:g}A": round(v, 3) for c, v in zip(centers, mae_bin)}

        # anchor error vs voltage at start (quartile bins) -- goes to JSON
        q = np.quantile(v0, [0, 0.25, 0.5, 0.75, 1.0])
        scen_json["mae_by_V0_quartile"] = {
            f"[{q[k]:.3f},{q[k+1]:.3f}]V": round(float(err[(v0 >= q[k]) & (v0 <= q[k + 1])].mean() * 100), 3)
            for k in range(4)
        }
        json_out[scenario] = scen_json
        print(f"{scenario}: anchor MAE {scen_json['overall']['mae_pct']}% | "
              f"rest-start {scen_json['rest_start']['mae_pct'] if scen_json['rest_start'] else 'n/a'}% "
              f"(n={scen_json['rest_start']['n'] if scen_json['rest_start'] else 0}) | "
              f"load-start {scen_json['load_start']['mae_pct']}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "anchor_error_by_temperature.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT_DIR / "anchor_error_analysis.json").write_text(json.dumps({
        "provenance": provenance("scenario_A", "outputs/v7_final/hard_coulomb_lstm_*.pt", {
            "experiment": "anchor_error_analysis",
            "anchor_mechanism": "anchor_head(h[:,0,:]) -> sigmoid -> remapped to [lo,hi]",
        }),
        "results": json_out,
    }, indent=2))
    fig_t.tight_layout()
    fig_t.savefig(FIG_DIR / "anchor_error_by_temperature.png", dpi=160, facecolor="#fcfcfb")
    fig_c.tight_layout()
    fig_c.savefig(FIG_DIR / "anchor_error_vs_current_start.png", dpi=160, facecolor="#fcfcfb")
    print(f"Saved CSV/JSON to {OUT_DIR.relative_to(BASE_DIR)}, figures to {FIG_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
