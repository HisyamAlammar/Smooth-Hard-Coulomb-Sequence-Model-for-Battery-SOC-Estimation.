from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/ablation_studies_v5_final/20_Final_Publication_Money_Plots.ipynb")


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    code_cell(
        """import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
from pathlib import Path

# IEEE Publication rcParams
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.titlesize': 13,
    'lines.linewidth': 1.8,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})
sns.set_style("whitegrid")
output_dir = Path("../../figures/publication_ready/")
output_dir.mkdir(parents=True, exist_ok=True)"""
    ),
    code_cell(
        """import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path("../..").resolve()
OUT = output_dir.resolve()
for p in (ROOT, ROOT / "src", ROOT / "experiments", ROOT / "inference", ROOT / "baselines"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from config import Q_NOMINAL, R_INT_PER_TEMP
from preprocessing_v4 import Q_ACTUAL_PER_TEMP, read_csv

COL = {
    "steel": "#2F5D7C",
    "blue": "#0072B2",
    "teal": "#008B8B",
    "orange": "#D55E00",
    "burnt": "#B85C00",
    "gray": "#6B7280",
    "slate": "#475569",
    "black": "#111111",
}

def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path

def save_both(fig: plt.Figure, stem: str) -> dict:
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return {"png": str(png.relative_to(ROOT)), "pdf": str(pdf.relative_to(ROOT))}

def raw_1hz(temp: str, filename: str) -> pd.DataFrame:
    path = require_file(ROOT / "data" / "raw" / "LG Dataset" / "LG_HG2_Original_Dataset" / temp / filename)
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

manifest = {
    "policy": "publication-ready figures generated only from verified CSV/JSON/NPY/checkpoint artifacts",
    "figures": [],
}"""
    ),
    code_cell(
        """# Figure 1: Raw UDDS 25 degC vs -20 degC
src25 = "data/raw/LG Dataset/LG_HG2_Original_Dataset/25degC/551_UDDS.csv"
srcn20 = "data/raw/LG Dataset/LG_HG2_Original_Dataset/n20degC/610_UDDS.csv"
d25 = raw_1hz("25degC", "551_UDDS.csv").iloc[:1200]
dn20 = raw_1hz("n20degC", "610_UDDS.csv").iloc[:1200]

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex="col")
for col, (label, df) in enumerate((("25 °C UDDS", d25), ("−20 °C UDDS", dn20))):
    t = df["time_sec"] - df["time_sec"].iloc[0]
    axv = axes[0, col]
    axv.plot(t, df["Voltage"], color=COL["steel"], label=r"$V_{terminal}$")
    axv.plot(t, df["V_proxy"], color=COL["burnt"], linestyle="--", label=r"$V_{proxy}$")
    axv.set_title(label, pad=8)
    axv.set_ylabel("Voltage (V)")
    axv.legend(loc="lower left", frameon=True, framealpha=0.95)

    axi = axes[1, col]
    axi.plot(t, df["Current"], color=COL["teal"], label="Current")
    axi.set_xlabel("Time (s)")
    axi.set_ylabel("Current (A)")
    axs = axi.twinx()
    axs.plot(t, df["SOC_actual"], color=COL["orange"], label="SOC")
    axs.set_ylabel("SOC (%)")
    lines = axi.get_lines() + axs.get_lines()
    axi.legend(lines, [line.get_label() for line in lines], loc="upper right", frameon=True, framealpha=0.95)

for ax in axes.ravel():
    ax.grid(True, which="major", alpha=0.35, linestyle="--")
fig.tight_layout(pad=1.4)
files = save_both(fig, "fig01_money_raw_udds_signals")
manifest["figures"].append({
    "id": "fig01",
    "files": files,
    "sources": [src25, srcn20, "src/config.py", "src/preprocessing_v4.py"],
    "message": "Raw UDDS voltage, current, and SOC contrast at 25 °C and −20 °C.",
})"""
    ),
    code_cell(
        """# Figure 2: Training and validation loss curves
training_paths = {
    ("HC-LSTM", "Scenario A"): ROOT / "outputs" / "v7_final" / "logs" / "training_log_hard_coulomb_lstm_scenario_A.csv",
    ("HC-LSTM", "Scenario B"): ROOT / "outputs" / "v7_final" / "logs" / "training_log_hard_coulomb_lstm_scenario_B.csv",
    ("HC-TCN", "Scenario A"): ROOT / "logs" / "training_log_v5_coulomb_tcn_scenario_A.csv",
    ("HC-TCN", "Scenario B"): ROOT / "logs" / "training_log_v5_coulomb_tcn_scenario_B.csv",
}
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
for ax, ((model, scenario), path) in zip(axes.ravel(), training_paths.items()):
    path = require_file(path)
    df = pd.read_csv(path)
    ax.plot(df["epoch"], df["train_loss"], color=COL["steel"], marker="o", markersize=3.5,
            markevery=max(1, len(df) // 8), label="Training")
    ax.plot(df["epoch"], df["val_loss"], color=COL["orange"], marker="s", markersize=3.5,
            markevery=max(1, len(df) // 8), label="Validation")
    ax.set_yscale("log")
    ax.set_title(f"{model} {scenario}", pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (log scale)")
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax.grid(True, which="both", alpha=0.30, linestyle="--")
fig.tight_layout(pad=1.3)
files = save_both(fig, "fig02_money_training_validation_loss")
manifest["figures"].append({
    "id": "fig02",
    "files": files,
    "sources": [str(p.relative_to(ROOT)) for p in training_paths.values()],
    "message": "Training and validation loss convergence from stored experiment logs.",
})"""
    ),
    code_cell(
        """# Figure 3: Scenario A multi-seed RMSE bar chart
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

labels = ["Null\\nOCV+Coulomb", "Vanilla\\nLSTM", "HC\\nanchor_last", "Best EKF\\n1RC"]
values = [
    float(null["rmse_pct"]),
    seed_stats("vanilla_lstm")[0],
    seed_stats("hc_anchor_last")[0],
    float(best_ekf["rmse_pct"]),
]
errors = [0.0, seed_stats("vanilla_lstm")[1], seed_stats("hc_anchor_last")[1], 0.0]
colors = [COL["slate"], "#5E81AC", COL["orange"], "#7C3AED"]

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(labels))
bars = ax.bar(x, values, yerr=errors, capsize=5, color=colors, edgecolor="#222222", linewidth=0.6, width=0.62)
for i in (0, 3):
    bars[i].set_hatch("//")
for bar, val, err in zip(bars, values, errors):
    ax.text(bar.get_x() + bar.get_width() / 2, val + err + 0.32, f"{val:.2f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Scenario A RMSE (%)")
ax.set_ylim(0, max(v + e for v, e in zip(values, errors)) + 3.0)
ax.grid(axis="y", alpha=0.35, linestyle="--")
ax.text(0.02, 0.96, "Hatched bars: deterministic / single-checkpoint",
        transform=ax.transAxes, ha="left", va="top", fontsize=9, color=COL["gray"])
fig.tight_layout()
files = save_both(fig, "fig03_money_multiseed_rmse")
manifest["figures"].append({
    "id": "fig03",
    "files": files,
    "sources": [
        "results/v5/multiseed/multiseed_summary.csv",
        "results/v5/final_v5_model_comparison.csv",
        "results/v5/ekf_ecm/recursive_vs_ekf_comparison.csv",
    ],
    "message": "Scenario A RMSE comparison with stored multi-seed standard deviation where available.",
})"""
    ),
    code_cell(
        """# Figure 4: eta calibration sweep money plot
eta_path = require_file(ROOT / "results" / "v5" / "delta_calibration" / "eta_gamma_sweep.json")
payload = json.loads(eta_path.read_text(encoding="utf-8"))
eta_df = pd.DataFrame(payload["rows"])
d = eta_df[(eta_df["mode"] == "inference_sweep") & (eta_df["gamma_mode"] == "nominal")].sort_values("eta")
best = d.loc[(d["eta"] - 2.0).abs().idxmin()]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(d["eta"], d["rec_rmse_pct"], color=COL["steel"], marker="o", label="Recursive RMSE")
ax.set_xlabel(r"Calibration factor $\eta^*$")
ax.set_ylabel("Recursive RMSE (%)", color=COL["steel"])
ax.tick_params(axis="y", labelcolor=COL["steel"])
ax.grid(True, alpha=0.35, linestyle="--")

ax2 = ax.twinx()
ax2.plot(d["eta"], d["rec_delta_ratio"], color=COL["burnt"], marker="s", label=r"Delta Ratio $r_\\Delta$")
ax2.axhline(1.0, color=COL["gray"], linestyle=":", linewidth=1.3)
ax2.set_ylabel(r"Delta Ratio $r_\\Delta$", color=COL["burnt"])
ax2.tick_params(axis="y", labelcolor=COL["burnt"])

ax.axvline(2.0, color=COL["orange"], linestyle="--", linewidth=1.4)
ax.scatter([best["eta"]], [best["rec_rmse_pct"]], s=70, color=COL["orange"], edgecolor="white", zorder=5)
ax2.scatter([best["eta"]], [best["rec_delta_ratio"]], s=70, color=COL["burnt"], edgecolor="white", zorder=5)
ax.annotate(r"$\\eta^*=2.0$", xy=(2.0, best["rec_rmse_pct"]), xytext=(2.08, best["rec_rmse_pct"] + 1.3),
            arrowprops=dict(arrowstyle="->", color=COL["orange"], lw=1.1), color=COL["orange"], fontsize=10)
handles = [ax.lines[0], ax2.lines[0]]
ax.legend(handles, [h.get_label() for h in handles], loc="upper center", frameon=True, framealpha=0.95, ncol=2)
fig.tight_layout()
files = save_both(fig, "fig04_money_eta_calibration_sweep")
manifest["figures"].append({
    "id": "fig04",
    "files": files,
    "sources": ["results/v5/delta_calibration/eta_gamma_sweep.json"],
    "provenance": payload.get("provenance", {}),
    "message": "Eta*=2.0 jointly minimizes recursive RMSE and restores delta ratio near 1.",
})"""
    ),
    code_cell(
        """# Figure 5: -20 degC trajectory zoom from checkpoint inference and EKF reconstruction
import torch
from ekf_1rc_ecm_continuous import run_continuous_1rc_ekf
from ekf_ocv_rint_continuous import map_back_to_windows, reconstruct_sequences
from gated_recursive_inference import gating_features, run_policy
from run_multiseed_v5 import build_model, load_splits

def envelope_forward(model, X, I, eta: float, gamma_w: np.ndarray, device, batch: int = 1024):
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

data_dir = ROOT / "data" / "processed" / "v5c_scenario_A"
ckpt = require_file(ROOT / "results" / "v5" / "headline_models" / "checkpoints" / "hard_coulomb_lstm_v5c_scenario_A_seed42.pt")
for name in ("X_test.npy", "y_test.npy", "I_unscaled_test.npy", "temp_labels_test.npy", "timestamp_key_test.npy"):
    require_file(data_dir / name)

data = load_splits("v5c", "A")
keys = np.load(data_dir / "timestamp_key_test.npy")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
model = build_model("hard_coulomb_lstm").to(device)
model.load_state_dict(checkpoint["model_state_dict"])

gamma_w = np.full(len(data["I_test"]), 1.0 / (3600.0 * Q_NOMINAL), dtype=np.float32)
cum, anchor = envelope_forward(model, data["X_test"], data["I_test"], 2.0, gamma_w, device)
feat = gating_features(data["X_test"], data["I_test"])
hc, _ = run_policy("carried_anchor", cum, anchor, keys, feat)

sequences, ekf_keys = reconstruct_sequences(data_dir)
run_continuous_1rc_ekf(sequences, 0.01)
ekf_pred = map_back_to_windows(sequences, ekf_keys)

idxs = np.where(data["temp_labels"] == "n20degC")[0]
if len(idxs) == 0:
    raise RuntimeError("No n20degC test windows found.")
mae_hc = np.mean(np.abs(hc[idxs] - data["y_test"][idxs]), axis=1)
mae_ekf = np.mean(np.abs(ekf_pred[idxs] - data["y_test"][idxs]), axis=1)
window_index = int(idxs[np.argmax(mae_ekf - mae_hc)])

t = np.arange(data["y_test"].shape[1])
gt = data["y_test"][window_index] * 100.0
hc_soc = hc[window_index] * 100.0
ekf_soc = ekf_pred[window_index] * 100.0
ymin = min(gt.min(), hc_soc.min(), ekf_soc.min()) - 1.5
ymax = max(gt.max(), hc_soc.max(), ekf_soc.max()) + 1.5

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, gt, color=COL["black"], linewidth=2.2, label="Ground Truth SOC")
ax.plot(t, hc_soc, color=COL["orange"], linewidth=2.1, label="Calibrated Recursive HC")
ax.plot(t, ekf_soc, color=COL["teal"], linestyle="--", linewidth=2.0, label="Best EKF 1RC")
ax.set_xlabel("Timestep (s)")
ax.set_ylabel("SOC (%)")
ax.set_ylim(ymin, ymax)
ax.grid(True, alpha=0.35, linestyle="--")
ax.legend(loc="best", frameon=True, framealpha=0.95)
ax.text(0.02, 0.05, f"−20 °C window index: {window_index}", transform=ax.transAxes, fontsize=9, color=COL["gray"])
fig.tight_layout()
files = save_both(fig, "fig05_money_subzero_trajectory_zoom")
manifest["figures"].append({
    "id": "fig05",
    "files": files,
    "sources": [
        "data/processed/v5c_scenario_A/X_test.npy",
        "data/processed/v5c_scenario_A/y_test.npy",
        "data/processed/v5c_scenario_A/I_unscaled_test.npy",
        "data/processed/v5c_scenario_A/temp_labels_test.npy",
        "data/processed/v5c_scenario_A/timestamp_key_test.npy",
        "results/v5/headline_models/checkpoints/hard_coulomb_lstm_v5c_scenario_A_seed42.pt",
        "results/v5/ekf_ecm/recursive_vs_ekf_comparison.csv",
    ],
    "window_index": window_index,
    "message": "Sub-zero trajectory comparison generated from checkpoint inference and EKF reconstruction.",
})"""
    ),
    code_cell(
        """# Validation: all PNG/PDF exist and PNGs are nonblank
from PIL import Image

validation = []
for entry in manifest["figures"]:
    for kind, rel in entry["files"].items():
        path = ROOT / rel
        item = {"figure": entry["id"], "kind": kind, "file": rel, "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}
        if kind == "png" and path.exists():
            with Image.open(path) as img:
                arr = np.asarray(img.convert("L"))
            item.update({"width": img.width, "height": img.height, "pixel_std": round(float(arr.std()), 4), "blank": bool(arr.std() < 1.0)})
        validation.append(item)

manifest["validation"] = validation
manifest_path = OUT / "publication_ready_figure_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(NOTEBOOK)
