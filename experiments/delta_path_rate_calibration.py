"""
delta_path_rate_calibration.py -- Phase 5: eta/gamma envelope calibration.

Two questions:
  1. Inference-time sweep: with the TRAINED model fixed (eta_train = 1.5,
     gamma nominal), how do eta and temperature-aware gamma scaling change
     rate fidelity, windowed accuracy, and recursive drift?
  2. Train-time check: retrain the model at selected eta values (1.287 = the
     physical minimum max_T Q_nom/Q_act(T); 2.0) to test whether the magnitude
     head re-calibrates -- and to show the v4 eta ablation's eta=1.0 row was
     structurally incapable at -20 degC.

Envelope math is replicated exactly (routing thresholds, sigmoid magnitude,
[lo,hi] anchor remap) with a parameterized limit:
    limit = |I| * eta * gamma_w,   gamma_w = dt / (3600 * Q_w)
    Q_w = Q_nominal            (gamma_mode = "nominal")
    Q_w = Q_actual(T_window)   (gamma_mode = "temp_aware")
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "src"), str(BASE_DIR / "analysis"),
          str(BASE_DIR / "experiments"), str(BASE_DIR / "inference")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import Q_NOMINAL  # noqa: E402
from gated_recursive_inference import gating_features, run_policy  # noqa: E402
from preprocessing_v4 import Q_ACTUAL_PER_TEMP  # noqa: E402
from run_multiseed_v5 import build_model, load_splits, train_one  # noqa: E402
from soc_metrics import evaluate_soc_predictions  # noqa: E402

OUT_DIR = BASE_DIR / "results" / "v5" / "delta_calibration"
FIG_DIR = BASE_DIR / "results" / "v5" / "figures"
ETAS = (1.287, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0)
POLICIES_TESTED = ("windowed_independent", "carried_anchor", "hybrid_temperature_load_gate")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE_DIR).decode().strip()
    except Exception:
        return "unknown"


def envelope_forward(model, X, I, eta: float, gamma_w: np.ndarray, device, batch=1024):
    """Exact replica of the Hard-Coulomb forward with parameterized envelope.

    Returns cumulative (N,T), anchor value (N,), and magnitude saturation stats.
    """
    thr = model.hard_constraint.threshold
    cum_all, anchor_all = [], []
    sat_hi = sat_lo = total = 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i+batch]).to(device)
            ib = torch.from_numpy(I[i:i+batch]).to(device)
            gw = torch.from_numpy(gamma_w[i:i+batch]).to(device).float().view(-1, 1, 1)
            h, _ = model.lstm(xb)
            dl = model.delta_head(h)
            al = model.anchor_head(h[:, 0, :])
            I3 = ib.unsqueeze(-1)
            mag = torch.sigmoid(dl)
            active = I3.abs() > thr
            sat_hi += int(((mag > 0.99) & active).sum())
            sat_lo += int(((mag < 0.01) & active).sum())
            total += int(active.sum())
            limit = I3.abs() * (eta * gw)
            zero = torch.zeros_like(dl)
            delta = torch.where(I3 < -thr, -limit * mag, zero)
            delta = torch.where(I3 > thr, limit * mag, delta)
            cum = torch.cumsum(delta, dim=1).squeeze(-1)
            lo = (-cum.min(dim=1).values).clamp(0.0, 1.0)
            hi = (1.0 - cum.max(dim=1).values).clamp(0.0, 1.0)
            width = (hi - lo).clamp_min(1e-6)
            anchor = lo + width * torch.sigmoid(al.squeeze(-1))
            cum_all.append(cum.cpu().numpy())
            anchor_all.append(anchor.cpu().numpy())
    stats = {"mag_sat_high_pct": round(100.0 * sat_hi / max(total, 1), 3),
             "mag_sat_low_pct": round(100.0 * sat_lo / max(total, 1), 3)}
    return np.concatenate(cum_all), np.concatenate(anchor_all), stats


def evaluate_config(model, data, keys, feat, eta, gamma_mode, device) -> Dict:
    temps = data["temp_labels"]
    if gamma_mode == "temp_aware":
        q_w = np.array([Q_ACTUAL_PER_TEMP.get(str(t), Q_NOMINAL) for t in temps])
    else:
        q_w = np.full(len(temps), Q_NOMINAL)
    gamma_w = (1.0 / (3600.0 * q_w)).astype(np.float32)
    cum, anchor, sat = envelope_forward(model, data["X_test"], data["I_test"], eta, gamma_w, device)

    out = {"eta": eta, "gamma_mode": gamma_mode, **sat}
    for policy in POLICIES_TESTED:
        y, pstats = run_policy(policy, cum, anchor, keys, feat)
        m = evaluate_soc_predictions(data["y_test"], y, data["I_test"], temps)
        reg = m["regression"]
        pfx = {"windowed_independent": "win", "carried_anchor": "rec",
               "hybrid_temperature_load_gate": "gated"}[policy]
        out[f"{pfx}_rmse_pct"] = round(reg["rmse_full_pct"], 4)
        out[f"{pfx}_maxe_pct"] = round(reg["maxe_full_pct"], 4)
        out[f"{pfx}_delta_ratio"] = round(m["delta_magnitude"]["discharge"]["pred_true_delta_ratio"], 4)
        for temp, short in (("n20degC", "n20"), ("40degC", "40")):
            pt = m.get("per_temperature", {}).get(temp)
            if pt:
                out[f"{pfx}_rmse_{short}"] = round(pt["regression"]["rmse_full_pct"], 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v5c")
    ap.add_argument("--scenario", default="A")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--retrain-etas", nargs="+", type=float, default=[1.287, 2.0])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_splits(args.variant, args.scenario)
    keys = np.load(BASE_DIR / "data" / "processed" /
                   f"{args.variant}_scenario_{args.scenario}" / "timestamp_key_test.npy")
    feat = gating_features(data["X_test"], data["I_test"])

    ckpt = BASE_DIR / "results" / "v5" / "headline_models" / "checkpoints" / \
        f"hard_coulomb_lstm_{args.variant}_scenario_{args.scenario}_seed{args.seed}.pt"
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_model("hard_coulomb_lstm").to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    rows = []
    print("== inference-time sweep (model trained at eta=1.5, gamma nominal) ==")
    for gamma_mode in ("nominal", "temp_aware"):
        for eta in ETAS:
            row = {"mode": "inference_sweep", **evaluate_config(model, data, keys, feat, eta, gamma_mode, device)}
            rows.append(row)
            print("  " + " | ".join(f"{k}={v}" for k, v in row.items()
                                    if k in ("eta", "gamma_mode", "win_rmse_pct", "rec_rmse_pct",
                                             "rec_rmse_40", "rec_rmse_n20", "win_delta_ratio")))

    print("== train-time check (retrained envelopes) ==")
    import model_v5_coulomb as m5
    for eta_tr in args.retrain_etas:
        # temporary model class with different safety factor via monkey-config:
        ckpt_tr = BASE_DIR / "results" / "v5" / "delta_calibration" / \
            f"hc_eta{eta_tr:g}_{args.variant}_scenario_{args.scenario}_seed{args.seed}.pt"
        if not ckpt_tr.exists():
            model_tr = m5.HardCoulombLSTM(safety_factor=eta_tr).to(device)
            # reuse the generic trainer loop by temporary registration
            from run_multiseed_v5 import BATCH_SIZE, EPOCHS, LEARNING_RATE, PATIENCE  # noqa
            import copy as _copy
            import time as _time
            import torch.nn as nn
            from sprint48_common import make_loader, set_reproducibility
            set_reproducibility(args.seed)
            tl = make_loader(data["X_train"], data["y_train"], data["I_train"], BATCH_SIZE, True, args.seed, device)
            vl_loader = make_loader(data["X_val"], data["y_val"], data["I_val"], BATCH_SIZE, False, args.seed, device)
            crit = nn.MSELoss()
            opt = torch.optim.AdamW(model_tr.parameters(), lr=LEARNING_RATE)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
            best, best_state, pat = float("inf"), None, 0
            for ep in range(1, EPOCHS + 1):
                model_tr.train()
                for xb, yb, ib in tl:
                    xb, ib = xb.to(device), ib.to(device)
                    yt = yb.unsqueeze(-1).to(device)
                    opt.zero_grad(set_to_none=True)
                    loss = crit(model_tr(xb, ib), yt)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model_tr.parameters(), 1.0)
                    opt.step()
                sch.step()
                model_tr.eval()
                v, nb = 0.0, 0
                with torch.no_grad():
                    for xb, yb, ib in vl_loader:
                        xb, ib = xb.to(device), ib.to(device)
                        v += float(crit(model_tr(xb, ib), yb.unsqueeze(-1).to(device)).item())
                        nb += 1
                v /= max(nb, 1)
                if v < best:
                    best, best_state, pat = v, _copy.deepcopy(model_tr.state_dict()), 0
                else:
                    pat += 1
                if pat >= PATIENCE:
                    break
            ckpt_tr.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": best_state, "eta_train": eta_tr, "seed": args.seed,
                        "dataset_variant": args.variant, "scenario": args.scenario,
                        "best_val_loss": best, "git_commit": git_commit()}, ckpt_tr)
        payload_tr = torch.load(ckpt_tr, map_location=device, weights_only=False)
        model_tr = m5.HardCoulombLSTM(safety_factor=eta_tr).to(device)
        model_tr.load_state_dict(payload_tr["model_state_dict"])
        model_tr.eval()
        row = {"mode": f"retrained_eta{eta_tr:g}",
               **evaluate_config(model_tr, data, keys, feat, eta_tr, "nominal", device)}
        rows.append(row)
        print("  " + " | ".join(f"{k}={v}" for k, v in row.items()
                                if k in ("mode", "win_rmse_pct", "rec_rmse_pct", "rec_rmse_40",
                                         "rec_rmse_n20", "win_delta_ratio")))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "eta_gamma_sweep.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = sorted({k for r in rows for k in r})
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (OUT_DIR / "eta_gamma_sweep.json").write_text(json.dumps({
        "provenance": {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                       "git_commit": git_commit(), "dataset_version": args.variant,
                       "scenario": f"scenario_{args.scenario}", "seed": args.seed,
                       "checkpoint": str(ckpt.relative_to(BASE_DIR)),
                       "note": "inference sweep uses fixed weights trained at eta=1.5"},
        "rows": rows}, indent=2, default=float))

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C1, C2, C3 = "#2a78d6", "#1baf7a", "#eda100"

    def style(ax):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color="#e1e0d9", linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors="#898781", labelsize=8)

    sweep_nom = [r for r in rows if r["mode"] == "inference_sweep" and r["gamma_mode"] == "nominal"]
    etas = [r["eta"] for r in sweep_nom]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(etas, [r.get("rec_rmse_n20") for r in sweep_nom], color=C1, marker="o", linewidth=2, label="recursive −20 °C")
    ax.plot(etas, [r.get("rec_rmse_40") for r in sweep_nom], color=C2, marker="o", linewidth=2, label="recursive 40 °C")
    ax.plot(etas, [r.get("win_rmse_pct") for r in sweep_nom], color=C3, marker="o", linewidth=2, label="windowed overall")
    ax.set_xlabel("eta (inference)", fontsize=8, color="#898781")
    ax.set_ylabel("RMSE (%SOC)", fontsize=8, color="#898781")
    ax.set_title("Eta sweep vs RMSE (weights fixed, gamma nominal)", fontsize=9, color="#0b0b0b")
    style(ax)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eta_vs_rmse_by_temperature.png", dpi=160, facecolor="#fcfcfb")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(etas, [r.get("win_delta_ratio") for r in sweep_nom], color=C1, marker="o", linewidth=2, label="gamma nominal")
    sweep_ta = [r for r in rows if r["mode"] == "inference_sweep" and r["gamma_mode"] == "temp_aware"]
    ax.plot(etas, [r.get("win_delta_ratio") for r in sweep_ta], color=C2, marker="o", linewidth=2, label="gamma temp-aware")
    ax.axhline(1.0, color="#898781", linewidth=1, linestyle="--")
    ax.set_xlabel("eta (inference)", fontsize=8, color="#898781")
    ax.set_ylabel("pred/true delta ratio (discharge)", fontsize=8, color="#898781")
    ax.set_title("Rate fidelity vs eta", fontsize=9, color="#0b0b0b")
    style(ax)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eta_vs_delta_ratio.png", dpi=160, facecolor="#fcfcfb")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(etas, [r.get("rec_rmse_pct") for r in sweep_nom], color=C1, marker="o", linewidth=2, label="carried, gamma nominal")
    ax.plot(etas, [r.get("rec_rmse_pct") for r in sweep_ta], color=C2, marker="o", linewidth=2, label="carried, gamma temp-aware")
    ax.plot(etas, [r.get("gated_rmse_pct") for r in sweep_nom], color=C3, marker="o", linewidth=2, label="gated, gamma nominal")
    ax.set_xlabel("eta (inference)", fontsize=8, color="#898781")
    ax.set_ylabel("RMSE (%SOC)", fontsize=8, color="#898781")
    ax.set_title("Recursive drift vs eta", fontsize=9, color="#0b0b0b")
    style(ax)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eta_vs_recursive_drift.png", dpi=160, facecolor="#fcfcfb")

    print(f"Saved to {OUT_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
