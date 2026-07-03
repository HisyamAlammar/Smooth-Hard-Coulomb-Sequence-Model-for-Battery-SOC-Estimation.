"""
compare_recursive_policies_v5.py -- Phase 4: evaluate anchor-policy family.

Runs every policy in inference/gated_recursive_inference.py on a trained
Hard-Coulomb checkpoint (v5 variant), reports full metrics + per-temperature
breakdown + re-anchor statistics, and renders case-study figures.

Usage:
  python experiments/compare_recursive_policies_v5.py --variant v5c --scenario A --seed 42
"""

from __future__ import annotations

import argparse
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
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "src"), str(BASE_DIR / "analysis"),
          str(BASE_DIR / "experiments"), str(BASE_DIR / "inference")):
    if p not in sys.path:
        sys.path.insert(0, p)

from gated_recursive_inference import POLICIES, gating_features, precompute, run_policy  # noqa: E402
from run_multiseed_v5 import build_model, load_splits  # noqa: E402
from soc_metrics import evaluate_soc_predictions  # noqa: E402

OUT_DIR = BASE_DIR / "results" / "v5" / "recursive_inference"
FIG_DIR = BASE_DIR / "results" / "v5" / "figures"
C1, C2, C3, C_GRID, C_MUTED, C_INK = "#2a78d6", "#1baf7a", "#eda100", "#e1e0d9", "#898781", "#0b0b0b"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE_DIR).decode().strip()
    except Exception:
        return "unknown"


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=C_GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=C_MUTED, labelsize=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v5c")
    ap.add_argument("--scenario", default="A")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="hard_coulomb_lstm")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_splits(args.variant, args.scenario)
    keys = np.load(BASE_DIR / "data" / "processed" /
                   f"{args.variant}_scenario_{args.scenario}" / "timestamp_key_test.npy")
    ckpt = BASE_DIR / "results" / "v5" / "headline_models" / "checkpoints" / \
        f"{args.model}_{args.variant}_scenario_{args.scenario}_seed{args.seed}.pt"
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_model(args.model).to(device)
    model.load_state_dict(payload["model_state_dict"])

    cumulative, anchor_model = precompute(model, data["X_test"], data["I_test"], device)
    feat = gating_features(data["X_test"], data["I_test"])

    rows, detail, preds = [], {}, {}
    for policy in POLICIES:
        y, stats = run_policy(policy, cumulative, anchor_model, keys, feat)
        m = evaluate_soc_predictions(data["y_test"], y, data["I_test"], data["temp_labels"])
        preds[policy] = y
        reg = m["regression"]
        row = {"policy": policy,
               "rmse_pct": round(reg["rmse_full_pct"], 4),
               "mae_pct": round(reg["mae_full_pct"], 4),
               "maxe_pct": round(reg["maxe_full_pct"], 4),
               "pvr_disch_eps0": round(m["pvr"]["discharge"]["by_epsilon"]["0"]["rate_pct"], 4),
               "delta_ratio_disch": round(m["delta_magnitude"]["discharge"]["pred_true_delta_ratio"], 4),
               **stats}
        for temp in ("n20degC", "n10degC", "40degC", "25degC"):
            pt = m.get("per_temperature", {}).get(temp)
            if pt:
                row[f"rmse_{temp}"] = round(pt["regression"]["rmse_full_pct"], 4)
                row[f"maxe_{temp}"] = round(pt["regression"]["maxe_full_pct"], 4)
        rows.append(row)
        detail[policy] = m
        print("  " + " | ".join(f"{k}={v}" for k, v in row.items() if not k.startswith("mae")))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "recursive_policy_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT_DIR / "recursive_policy_results.json").write_text(json.dumps({
        "provenance": {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                       "git_commit": git_commit(), "dataset_version": args.variant,
                       "label_mode": payload.get("label_mode"),
                       "decimation_mode": payload.get("decimation_mode"),
                       "scenario": f"scenario_{args.scenario}",
                       "checkpoint": str(ckpt.relative_to(BASE_DIR)), "seed": args.seed,
                       "gating_params": {"cold_T_C": 5.0, "load_thr_A": 0.05,
                                         "dvdt_stable_Vps": 1e-3, "blend_load_ref_A": 2.0}},
        "results": detail,
    }, indent=2, default=float))

    # temperature-breakdown figure (per-policy RMSE at n20 vs 40)
    temps_avail = [t for t in ("n20degC", "n10degC", "25degC", "40degC") if f"rmse_{t}" in rows[0]]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    x = np.arange(len(rows))
    width = 0.8 / max(len(temps_avail), 1)
    colors = [C1, C2, C3, "#4a3aa7"]
    for i, t in enumerate(temps_avail):
        ax.bar(x + (i - len(temps_avail) / 2 + 0.5) * width,
               [r.get(f"rmse_{t}", np.nan) for r in rows], width=width * 0.9,
               color=colors[i % len(colors)], label=t)
    ax.set_xticks(x, [r["policy"].replace("_", "\n") for r in rows], fontsize=6.5)
    ax.set_ylabel("RMSE (%SOC)", fontsize=8, color=C_MUTED)
    ax.set_title(f"Recursive-policy RMSE by temperature ({args.variant} S{args.scenario}, seed {args.seed})",
                 fontsize=9, color=C_INK)
    style(ax)
    ax.legend(fontsize=7, frameon=False, ncol=len(temps_avail))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "recursive_policy_temperature_breakdown.png", dpi=160, facecolor="#fcfcfb")

    # case studies: coldest worst window, hottest degraded window
    def case_plot(mask_temp: str, fname: str, title: str, policies=("windowed_independent", "carried_anchor")):
        mask = data["temp_labels"] == mask_temp
        if not mask.any():
            return
        idx_all = np.where(mask)[0]
        err = np.abs(preds[policies[1]][idx_all] - data["y_test"][idx_all]).max(1)
        j = idx_all[int(np.argmax(err))] if "failure" in fname else idx_all[int(np.argsort(
            np.abs(preds[policies[0]][idx_all] - data["y_test"][idx_all]).max(1))[-1])]
        fig, ax = plt.subplots(figsize=(6.8, 3.2))
        t = np.arange(data["y_test"].shape[1])
        ax.plot(t, data["y_test"][j], color=C_INK, linewidth=1.8, label="true SOC")
        ax.plot(t, preds["windowed_independent"][j], color=C1, linewidth=1.4, label="windowed")
        ax.plot(t, preds["carried_anchor"][j], color=C2, linewidth=1.4, label="carried")
        best_gated = "hybrid_temperature_load_gate"
        ax.plot(t, preds[best_gated][j], color=C3, linewidth=1.4, label=best_gated)
        ax.set_title(title, fontsize=9, color=C_INK)
        ax.set_xlabel("t (s)", fontsize=8, color=C_MUTED)
        ax.set_ylabel("SOC", fontsize=8, color=C_MUTED)
        style(ax)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(FIG_DIR / fname, dpi=160, facecolor="#fcfcfb")

    case_plot("n20degC", "cold_sequence_recursive_case_study.png",
              "-20 degC: worst windowed-anchor window vs recursive policies")
    case_plot("40degC", "hot_sequence_recursive_failure_case.png",
              "40 degC: worst carried-anchor window (drift accumulation)")

    print(f"Saved to {OUT_DIR.relative_to(BASE_DIR)} and {FIG_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
