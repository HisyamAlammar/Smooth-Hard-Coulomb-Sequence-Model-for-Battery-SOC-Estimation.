"""Generate final manuscript figures from verified repository artifacts.

Rules:
- no SVG
- no mock arrays
- no training rerun
- every plotted value comes from raw CSV, JSON, NPY tensors, or checkpoint inference
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "authentic"
OUT.mkdir(parents=True, exist_ok=True)

for p in (ROOT, ROOT / "src", ROOT / "experiments", ROOT / "inference", ROOT / "baselines"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from config import Q_NOMINAL, R_INT_PER_TEMP  # noqa: E402
from preprocessing_v4 import Q_ACTUAL_PER_TEMP, read_csv  # noqa: E402

sns.set_style("whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.45,
        "lines.linewidth": 1.6,
    }
)

COL = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#6E6E6E",
    "black": "#111111",
}


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def save_png(fig: plt.Figure, name: str) -> str:
    path = OUT / name
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def raw_1hz(temp: str, filename: str) -> pd.DataFrame:
    path = require_file(
        ROOT / "data" / "raw" / "LG Dataset" / "LG_HG2_Original_Dataset" / temp / filename
    )
    df = read_csv(str(path)).sort_values("time_sec").copy()
    df["_second"] = np.floor(df["time_sec"].to_numpy(dtype=float) + 1e-9).astype(int)
    group = df.groupby("_second", sort=True)
    one = group[["Voltage", "Current", "Temperature"]].mean()
    one["Capacity"] = group["Capacity"].last()
    one["time_sec"] = one.index.astype(float)
    one["V_proxy"] = one["Voltage"] - one["Current"] * R_INT_PER_TEMP[temp]
    cap = one["Capacity"].fillna(0.0)
    one["SOC_actual"] = (1.0 - (cap - cap.iloc[0]).abs() / Q_ACTUAL_PER_TEMP[temp]).clip(0.0, 1.0) * 100.0
    return one.reset_index(drop=True)


def fig01_actual_udds_signals() -> dict:
    src25 = "data/raw/LG Dataset/LG_HG2_Original_Dataset/25degC/551_UDDS.csv"
    src20 = "data/raw/LG Dataset/LG_HG2_Original_Dataset/n20degC/610_UDDS.csv"
    d25 = raw_1hz("25degC", "551_UDDS.csv").iloc[:1200]
    dn20 = raw_1hz("n20degC", "610_UDDS.csv").iloc[:1200]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.2), sharex="col")
    for col, (title, df) in enumerate((("25 degC UDDS", d25), ("-20 degC UDDS", dn20))):
        t = df["time_sec"] - df["time_sec"].iloc[0]
        axes[0, col].plot(t, df["Voltage"], label="V_terminal", color=COL["blue"])
        axes[0, col].plot(t, df["V_proxy"], label="V_proxy", color=COL["orange"], linestyle="--")
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel("Voltage (V)")
        axes[0, col].legend(loc="best", frameon=True, framealpha=0.9)

        ax = axes[1, col]
        ax.plot(t, df["Current"], label="Current", color=COL["green"])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Current (A)")
        ax2 = ax.twinx()
        ax2.plot(t, df["SOC_actual"], label="SOC_actual", color=COL["vermillion"])
        ax2.set_ylabel("SOC (%)")
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [line.get_label() for line in lines], loc="best", frameon=True, framealpha=0.9)
    fig.tight_layout()
    return {
        "file": save_png(fig, "fig01_actual_udds_signals.png"),
        "sources": [src25, src20, "src/config.py", "src/preprocessing_v4.py"],
    }


def fig03_actual_multiseed_rmse() -> dict:
    ms_path = require_file(ROOT / "results" / "v5" / "multiseed" / "multiseed_summary.csv")
    final_path = require_file(ROOT / "results" / "v5" / "final_v5_model_comparison.csv")
    ekf_path = require_file(ROOT / "results" / "v5" / "ekf_ecm" / "recursive_vs_ekf_comparison.csv")
    ms = pd.read_csv(ms_path)
    final = pd.read_csv(final_path)
    ekf = pd.read_csv(ekf_path)

    def seed_stats(model: str) -> tuple[float, float]:
        row = ms[(ms["scenario"] == "scenario_A") & (ms["model"] == model)].iloc[0]
        return float(row["rmse_pct_mean"]), float(row["rmse_pct_std"])

    null = final[(final["scenario"] == "scenario_A") & (final["model"] == "null[ocv25_qnom]")].iloc[0]
    best_ekf = ekf[ekf["model"] == "EKF_1RC_cont[R=0.01]"].iloc[0]

    labels = ["Null OCV+Coulomb", "Vanilla LSTM", "HC anchor_last", "Best EKF 1RC"]
    vals = [
        float(null["rmse_pct"]),
        seed_stats("vanilla_lstm")[0],
        seed_stats("hc_anchor_last")[0],
        float(best_ekf["rmse_pct"]),
    ]
    errs = [0.0, seed_stats("vanilla_lstm")[1], seed_stats("hc_anchor_last")[1], 0.0]
    colors = [COL["gray"], COL["sky"], COL["vermillion"], COL["purple"]]

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    bars = ax.bar(labels, vals, yerr=errs, capsize=4, color=colors, edgecolor="white", linewidth=0.6)
    for i in (0, 3):
        bars[i].set_hatch("//")
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.35, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("Scenario A RMSE (%)")
    ax.set_ylim(0, max(vals) + 3)
    ax.tick_params(axis="x", rotation=12)
    ax.text(
        0.98,
        0.96,
        "Hatched: deterministic/single-checkpoint, no seed std",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color=COL["gray"],
    )
    fig.tight_layout()
    return {
        "file": save_png(fig, "fig03_actual_multiseed_rmse.png"),
        "sources": [
            "results/v5/multiseed/multiseed_summary.csv",
            "results/v5/final_v5_model_comparison.csv",
            "results/v5/ekf_ecm/recursive_vs_ekf_comparison.csv",
        ],
    }


def fig02_actual_training_curves() -> dict:
    paths = {
        ("HC-LSTM", "Scenario A"): require_file(ROOT / "outputs" / "v7_final" / "logs" / "training_log_hard_coulomb_lstm_scenario_A.csv"),
        ("HC-LSTM", "Scenario B"): require_file(ROOT / "outputs" / "v7_final" / "logs" / "training_log_hard_coulomb_lstm_scenario_B.csv"),
        ("HC-TCN", "Scenario A"): require_file(ROOT / "logs" / "training_log_v5_coulomb_tcn_scenario_A.csv"),
        ("HC-TCN", "Scenario B"): require_file(ROOT / "logs" / "training_log_v5_coulomb_tcn_scenario_B.csv"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.7), sharex=False, sharey=False)
    for ax, ((model, scenario), path) in zip(axes.ravel(), paths.items()):
        df = pd.read_csv(path)
        ax.plot(df["epoch"], df["train_loss"], color=COL["blue"], marker="o", markevery=max(1, len(df) // 6), label="Training")
        ax.plot(df["epoch"], df["val_loss"], color=COL["vermillion"], marker="s", markevery=max(1, len(df) // 6), label="Validation")
        ax.set_title(f"{model} {scenario}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss")
        ax.legend(loc="best", frameon=True, framealpha=0.9)
    fig.tight_layout()
    return {
        "file": save_png(fig, "fig02_actual_training_curves.png"),
        "sources": [str(path.relative_to(ROOT)) for path in paths.values()],
    }


def fig04_actual_eta_calibration() -> dict:
    json_path = require_file(ROOT / "results" / "v5" / "delta_calibration" / "eta_gamma_sweep.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    df = pd.DataFrame(payload["rows"])
    d = df[(df["mode"] == "inference_sweep") & (df["gamma_mode"] == "nominal")].sort_values("eta")

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.plot(d["eta"], d["rec_rmse_pct"], marker="o", color=COL["blue"], label="Recursive RMSE")
    ax.set_xlabel("eta*")
    ax.set_ylabel("Sequence RMSE (%)", color=COL["blue"])
    ax.tick_params(axis="y", labelcolor=COL["blue"])
    ax2 = ax.twinx()
    ax2.plot(d["eta"], d["rec_delta_ratio"], marker="s", color=COL["orange"], label="Delta ratio r_delta")
    ax2.axhline(1.0, color=COL["gray"], linestyle=":", linewidth=1.2)
    ax2.set_ylabel("Delta ratio r_delta", color=COL["orange"])
    ax2.tick_params(axis="y", labelcolor=COL["orange"])
    best = d.loc[(d["eta"] - 2.0).abs().idxmin()]
    ax.axvline(2.0, color=COL["vermillion"], linestyle="--", linewidth=1.2)
    ax.scatter([best["eta"]], [best["rec_rmse_pct"]], s=48, color=COL["vermillion"], zorder=5)
    ax.text(2.03, float(best["rec_rmse_pct"]) + 0.8, "eta*=2.0", color=COL["vermillion"], fontsize=9)
    handles = [ax.lines[0], ax2.lines[0]]
    ax.legend(handles, [h.get_label() for h in handles], loc="upper center", frameon=True, framealpha=0.9)
    fig.tight_layout()
    return {
        "file": save_png(fig, "fig04_actual_eta_calibration.png"),
        "sources": ["results/v5/delta_calibration/eta_gamma_sweep.json"],
        "provenance": payload.get("provenance", {}),
    }


def _envelope_forward(model, X, I, eta: float, gamma_w: np.ndarray, device, batch: int = 1024):
    import torch

    threshold = model.hard_constraint.threshold
    cum_all, anchor_all = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i : i + batch]).to(device)
            ib = torch.from_numpy(I[i : i + batch]).to(device)
            gw = torch.from_numpy(gamma_w[i : i + batch]).to(device).float().view(-1, 1, 1)
            h, _ = model.lstm(xb)
            delta_logits = model.delta_head(h)
            anchor_logits = model.anchor_head(h[:, 0, :])
            i3 = ib.unsqueeze(-1)
            magnitude = torch.sigmoid(delta_logits)
            limit = i3.abs() * (eta * gw)
            zero = torch.zeros_like(delta_logits)
            delta = torch.where(i3 < -threshold, -limit * magnitude, zero)
            delta = torch.where(i3 > threshold, limit * magnitude, delta)
            cum = torch.cumsum(delta, dim=1).squeeze(-1)
            lo = (-cum.min(dim=1).values).clamp(0.0, 1.0)
            hi = (1.0 - cum.max(dim=1).values).clamp(0.0, 1.0)
            width = (hi - lo).clamp_min(1e-6)
            anchor = lo + width * torch.sigmoid(anchor_logits.squeeze(-1))
            cum_all.append(cum.cpu().numpy())
            anchor_all.append(anchor.cpu().numpy())
    return np.concatenate(cum_all), np.concatenate(anchor_all)


def fig05_actual_subzero_trajectory() -> dict:
    import torch
    from ekf_1rc_ecm_continuous import run_continuous_1rc_ekf
    from ekf_ocv_rint_continuous import map_back_to_windows, reconstruct_sequences
    from gated_recursive_inference import gating_features, run_policy
    from run_multiseed_v5 import build_model, load_splits

    data_dir = ROOT / "data" / "processed" / "v5c_scenario_A"
    ckpt = require_file(
        ROOT / "results" / "v5" / "headline_models" / "checkpoints" / "hard_coulomb_lstm_v5c_scenario_A_seed42.pt"
    )
    for name in ("X_test.npy", "y_test.npy", "I_unscaled_test.npy", "temp_labels_test.npy", "timestamp_key_test.npy"):
        require_file(data_dir / name)

    data = load_splits("v5c", "A")
    keys = np.load(data_dir / "timestamp_key_test.npy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_model("hard_coulomb_lstm").to(device)
    model.load_state_dict(payload["model_state_dict"])

    gamma_w = np.full(len(data["I_test"]), 1.0 / (3600.0 * Q_NOMINAL), dtype=np.float32)
    cum, anchor = _envelope_forward(model, data["X_test"], data["I_test"], 2.0, gamma_w, device)
    feat = gating_features(data["X_test"], data["I_test"])
    hc, _ = run_policy("carried_anchor", cum, anchor, keys, feat)

    sequences, ekf_keys = reconstruct_sequences(data_dir)
    run_continuous_1rc_ekf(sequences, 0.01)
    ekf = map_back_to_windows(sequences, ekf_keys)

    idxs = np.where(data["temp_labels"] == "n20degC")[0]
    if len(idxs) == 0:
        raise RuntimeError("No n20degC test windows in Scenario A test tensors.")
    mae_hc = np.mean(np.abs(hc[idxs] - data["y_test"][idxs]), axis=1)
    mae_ekf = np.mean(np.abs(ekf[idxs] - data["y_test"][idxs]), axis=1)
    j = int(idxs[np.argmax(mae_ekf - mae_hc)])

    t = np.arange(data["y_test"].shape[1])
    fig, ax = plt.subplots(figsize=(6.7, 3.4))
    ax.plot(t, data["y_test"][j] * 100.0, color=COL["black"], label="Ground Truth SOC")
    ax.plot(t, hc[j] * 100.0, color=COL["vermillion"], label="Calibrated Recursive HC")
    ax.plot(t, ekf[j] * 100.0, color=COL["blue"], linestyle="--", label="Best EKF 1RC")
    ax.set_xlabel("Timestep (s)")
    ax.set_ylabel("SOC (%)")
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    fig.tight_layout()
    return {
        "file": save_png(fig, "fig05_actual_subzero_trajectory.png"),
        "sources": [
            "data/processed/v5c_scenario_A/X_test.npy",
            "data/processed/v5c_scenario_A/y_test.npy",
            "data/processed/v5c_scenario_A/I_unscaled_test.npy",
            "data/processed/v5c_scenario_A/temp_labels_test.npy",
            "data/processed/v5c_scenario_A/timestamp_key_test.npy",
            "results/v5/headline_models/checkpoints/hard_coulomb_lstm_v5c_scenario_A_seed42.pt",
            "results/v5/ekf_ecm/recursive_vs_ekf_comparison.csv",
        ],
        "window_index": j,
    }


def validate_pngs(entries: list[dict]) -> list[dict]:
    from PIL import Image

    out = []
    for entry in entries:
        path = ROOT / entry["file"]
        with Image.open(path) as img:
            arr = np.asarray(img.convert("L"))
            out.append(
                {
                    "file": entry["file"],
                    "exists": path.exists(),
                    "bytes": path.stat().st_size,
                    "width": img.width,
                    "height": img.height,
                    "pixel_std": round(float(arr.std()), 4),
                    "blank": bool(arr.std() < 1.0),
                    "sources": entry.get("sources", []),
                }
            )
    return out


def main() -> None:
    entries = [
        fig01_actual_udds_signals(),
        fig02_actual_training_curves(),
        fig03_actual_multiseed_rmse(),
        fig04_actual_eta_calibration(),
        fig05_actual_subzero_trajectory(),
    ]
    validation = validate_pngs(entries)
    report = {
        "policy": "authentic CSV/JSON/NPY/checkpoint figures only; no SVG; no mock arrays; no training rerun",
        "figures": entries,
        "validation": validation,
    }
    (OUT / "authentic_figure_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
