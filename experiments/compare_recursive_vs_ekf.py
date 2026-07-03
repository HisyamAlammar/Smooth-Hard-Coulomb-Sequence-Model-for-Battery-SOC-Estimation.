"""
compare_recursive_vs_ekf.py -- Phase 6: continuous classical filters vs
recursive Hard-Coulomb inference, on identical reconstructed profile chains.

Rows produced:
  null_ocv_coulomb (per-window, reference)
  HC windowed / carried / hybrid gate   (from Phase 4 machinery)
  EKF OCV-Rint continuous  (R sensitivity set)
  EKF 1RC-ECM continuous   (R sensitivity set)

All estimates are mapped back to the (N, T) test-window grid so every row is
scored by the same metrics module on the same targets.
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
          str(BASE_DIR / "baselines"), str(BASE_DIR / "experiments"), str(BASE_DIR / "inference")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ekf_1rc_ecm_continuous import run_continuous_1rc_ekf  # noqa: E402
from ekf_ocv_rint import build_soc_to_ocv  # noqa: E402
from ekf_ocv_rint_continuous import map_back_to_windows, reconstruct_sequences, run_continuous_scalar_ekf  # noqa: E402
from gated_recursive_inference import gating_features, precompute, run_policy  # noqa: E402
from run_multiseed_v5 import build_model, load_splits  # noqa: E402
from soc_metrics import evaluate_soc_predictions  # noqa: E402

OUT_DIR = BASE_DIR / "results" / "v5" / "ekf_ecm"
FIG_DIR = BASE_DIR / "results" / "v5" / "figures"
R_SET = (1e-4, 9e-4, 1e-2)
C1, C2, C3, C_INK, C_MUTED = "#2a78d6", "#1baf7a", "#eda100", "#0b0b0b", "#898781"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE_DIR).decode().strip()
    except Exception:
        return "unknown"


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color="#e1e0d9", linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=C_MUTED, labelsize=8)


def summarize(name, m, extra=None):
    reg = m["regression"]
    row = {"model": name,
           "rmse_pct": round(reg["rmse_full_pct"], 4), "mae_pct": round(reg["mae_full_pct"], 4),
           "maxe_pct": round(reg["maxe_full_pct"], 4),
           "pvr_disch_eps0": round(m["pvr"]["discharge"]["by_epsilon"]["0"]["rate_pct"], 4)}
    for temp, short in (("n20degC", "n20"), ("n10degC", "n10"), ("40degC", "40")):
        pt = m.get("per_temperature", {}).get(temp)
        if pt:
            row[f"rmse_{short}"] = round(pt["regression"]["rmse_full_pct"], 4)
            row[f"maxe_{short}"] = round(pt["regression"]["maxe_full_pct"], 4)
    if extra:
        row.update(extra)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v5c")
    ap.add_argument("--scenario", default="A")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_splits(args.variant, args.scenario)
    data_dir = BASE_DIR / "data" / "processed" / f"{args.variant}_scenario_{args.scenario}"
    keys = np.load(data_dir / "timestamp_key_test.npy")

    rows, detail = [], {}

    # HC policies
    ckpt = BASE_DIR / "results" / "v5" / "headline_models" / "checkpoints" / \
        f"hard_coulomb_lstm_{args.variant}_scenario_{args.scenario}_seed{args.seed}.pt"
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_model("hard_coulomb_lstm").to(device)
    model.load_state_dict(payload["model_state_dict"])
    cum, anchor = precompute(model, data["X_test"], data["I_test"], device)
    feat = gating_features(data["X_test"], data["I_test"])
    for policy, label in (("windowed_independent", "HC_windowed"),
                          ("carried_anchor", "HC_carried"),
                          ("hybrid_temperature_load_gate", "HC_hybrid_gate")):
        y, _ = run_policy(policy, cum, anchor, keys, feat)
        m = evaluate_soc_predictions(data["y_test"], y, data["I_test"], data["temp_labels"])
        rows.append(summarize(label, m))
        detail[label] = m

    # continuous filters on reconstructed chains
    sequences, _ = reconstruct_sequences(data_dir)
    n_steps = sum(len(s["I"]) for s in sequences)
    print(f"  reconstructed {len(sequences)} chains, {n_steps:,} unique steps")
    for R in R_SET:
        run_continuous_scalar_ekf(sequences, R)
        y = map_back_to_windows(sequences, keys)
        m = evaluate_soc_predictions(data["y_test"], y, data["I_test"], data["temp_labels"])
        rows.append(summarize(f"EKF_Rint_cont[R={R:g}]", m))
        detail[f"EKF_Rint_cont[R={R:g}]"] = m

        run_continuous_1rc_ekf(sequences, R)
        y = map_back_to_windows(sequences, keys)
        m = evaluate_soc_predictions(data["y_test"], y, data["I_test"], data["temp_labels"])
        rows.append(summarize(f"EKF_1RC_cont[R={R:g}]", m))
        detail[f"EKF_1RC_cont[R={R:g}]"] = m

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "recursive_vs_ekf_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "model", k))
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (OUT_DIR / "continuous_ekf_results.json").write_text(json.dumps({
        "provenance": {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                       "git_commit": git_commit(), "dataset_version": args.variant,
                       "scenario": f"scenario_{args.scenario}", "seed": args.seed,
                       "hc_checkpoint": str(ckpt.relative_to(BASE_DIR)),
                       "ekf_assumptions": "25degC OCV inversion; Q(T) calibration table; "
                                          "1RC: tau=50s, R1=0.5*R_int(T) (literature-like, not identified); "
                                          "no test tuning; R sensitivity fully reported"},
        "results": detail}, indent=2, default=float))

    for r in rows:
        print("  " + " | ".join(f"{k}={v}" for k, v in r.items()
                                if k in ("model", "rmse_pct", "maxe_pct", "rmse_n20", "rmse_40", "pvr_disch_eps0")))

    # temperature breakdown figure
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    x = np.arange(len(rows))
    for i, (col, c, lab) in enumerate((("rmse_n20", C1, "−20 °C"), ("rmse_40", C2, "40 °C"),
                                       ("rmse_pct", C3, "overall"))):
        ax.bar(x + (i - 1) * 0.27, [r.get(col, np.nan) for r in rows], width=0.25, color=c, label=lab)
    ax.set_xticks(x, [r["model"].replace("[", "\n[") for r in rows], fontsize=6.5)
    ax.set_ylabel("RMSE (%SOC)", fontsize=8, color=C_MUTED)
    ax.set_title(f"Recursive HC vs continuous EKF ({args.variant} S{args.scenario})", fontsize=9, color=C_INK)
    style(ax)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "recursive_vs_ekf_temperature_breakdown.png", dpi=160, facecolor="#fcfcfb")

    # voltage-residual case study at -20degC (1RC, mid R)
    h, _, _ = build_soc_to_ocv()
    cold = [s for s in sequences if s["temp"] == "n20degC"]
    if cold:
        s = max(cold, key=lambda q: len(q["I"]))
        t = np.arange(len(s["I"]))
        fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.2), sharex=True)
        axes[0].plot(t, s["y_true"], color=C_INK, linewidth=1.8, label="true SOC")
        axes[0].plot(t, s["soc_est"], color=C1, linewidth=1.4, label="1RC-EKF SOC")
        axes[0].set_ylabel("SOC", fontsize=8, color=C_MUTED)
        axes[0].legend(fontsize=7, frameon=False)
        style(axes[0])
        ocv_true = np.asarray(h(np.clip(s["y_true"], 0, 1)))
        axes[1].plot(t, s["v_proxy"] - ocv_true, color=C2, linewidth=1.2,
                     label="V_proxy − OCV(true SOC)  (polarization residual)")
        axes[1].plot(t, s.get("vrc_est", np.zeros_like(t, dtype=float)), color=C3, linewidth=1.2,
                     label="EKF V_rc estimate")
        axes[1].set_ylabel("V", fontsize=8, color=C_MUTED)
        axes[1].set_xlabel("t (s)", fontsize=8, color=C_MUTED)
        axes[1].legend(fontsize=7, frameon=False)
        style(axes[1])
        axes[0].set_title("−20 °C chain: 1RC-EKF tracking and polarization residual", fontsize=9, color=C_INK)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "ekf_voltage_residual_case_study.png", dpi=160, facecolor="#fcfcfb")

    print(f"Saved to {OUT_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
