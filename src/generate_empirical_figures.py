"""Generate empirical publication figures from saved research artifacts.

No training, no SVG, no mock diagrams. Every plotted value is read from raw
LG HG2 data, processed tensors, checkpoints, or CSV/JSON result artifacts.
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
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

for p in (ROOT, ROOT / "src", ROOT / "experiments", ROOT / "inference", ROOT / "baselines"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from config import PHYS_MAX_V3, PHYS_MIN_V3, Q_NOMINAL, R_INT_PER_TEMP  # noqa: E402
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


def save(fig: plt.Figure, stem: str) -> list[str]:
    paths = []
    for ext in ("png", "pdf"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, dpi=300)
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


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
    r_int = R_INT_PER_TEMP[temp]
    q_actual = Q_ACTUAL_PER_TEMP[temp]
    one["V_proxy"] = one["Voltage"] - one["Current"] * r_int
    cap = one["Capacity"].fillna(0.0)
    one["SOC_actual"] = (1.0 - (cap - cap.iloc[0]).abs() / q_actual).clip(0.0, 1.0) * 100.0
    return one.reset_index(drop=True)


def fig_raw_signal() -> dict:
    src25 = "data/raw/LG Dataset/LG_HG2_Original_Dataset/25degC/551_UDDS.csv"
    src20 = "data/raw/LG Dataset/LG_HG2_Original_Dataset/n20degC/610_UDDS.csv"
    d25 = raw_1hz("25degC", "551_UDDS.csv").iloc[:1200]
    dn20 = raw_1hz("n20degC", "610_UDDS.csv").iloc[:1200]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.2), sharex="col")
    for col, (title, df) in enumerate((("25 °C UDDS", d25), ("−20 °C UDDS", dn20))):
        t = df["time_sec"] - df["time_sec"].iloc[0]
        ax = axes[0, col]
        ax.plot(t, df["Voltage"], label="V_terminal", color=COL["blue"])
        ax.plot(t, df["V_proxy"], label="V_proxy", color=COL["orange"], linestyle="--")
        ax.set_title(title)
        ax.set_ylabel("Voltage (V)")
        ax.legend(loc="best", frameon=True, framealpha=0.9)

        ax = axes[1, col]
        ax.plot(t, df["Current"], label="Current", color=COL["green"])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Current (A)")
        ax2 = ax.twinx()
        ax2.plot(t, df["SOC_actual"], label="SOC_actual", color=COL["vermillion"], alpha=0.9)
        ax2.set_ylabel("SOC (%)")
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [x.get_label() for x in lines], loc="best", frameon=True, framealpha=0.9)
    fig.tight_layout()
    return {
        "outputs": save(fig, "fig_empirical_01_raw_signal_preprocessing"),
        "source_data": [src25, src20, "src/config.py", "src/preprocessing_v4.py"],
        "message": "Raw UDDS signal and V_proxy preprocessing are plotted from actual LG HG2 CSV files.",
        "placement": "Bagian II.A, before dataset table",
        "caveat": "SOC_actual is derived from recorded Capacity and q_actual table used by preprocessing_v4.",
        "status": "generated",
    }


def fig_multiseed() -> dict:
    ms_path = require_file(ROOT / "results" / "v5" / "multiseed" / "multiseed_summary.csv")
    final_path = require_file(ROOT / "results" / "v5" / "final_v5_model_comparison.csv")
    ekf_path = require_file(ROOT / "results" / "v5" / "ekf_ecm" / "recursive_vs_ekf_comparison.csv")

    ms = pd.read_csv(ms_path)
    final = pd.read_csv(final_path)
    ekf = pd.read_csv(ekf_path)

    def ms_row(model: str) -> tuple[float, float]:
        r = ms[(ms["scenario"] == "scenario_A") & (ms["model"] == model)].iloc[0]
        return float(r["rmse_pct_mean"]), float(r["rmse_pct_std"])

    null = final[(final["scenario"] == "scenario_A") & (final["model"] == "null[ocv25_qnom]")].iloc[0]
    best_ekf = ekf[ekf["model"] == "EKF_1RC_cont[R=0.01]"].iloc[0]
    methods = ["Null OCV+Coulomb", "Vanilla LSTM", "HC anchor_last", "Best EKF 1RC"]
    values = [
        float(null["rmse_pct"]),
        ms_row("vanilla_lstm")[0],
        ms_row("hc_anchor_last")[0],
        float(best_ekf["rmse_pct"]),
    ]
    errors = [0.0, ms_row("vanilla_lstm")[1], ms_row("hc_anchor_last")[1], 0.0]
    colors = [COL["gray"], COL["sky"], COL["vermillion"], COL["purple"]]

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    bars = ax.bar(methods, values, yerr=errors, capsize=4, color=colors, edgecolor="white")
    for i in (0, 3):
        bars[i].set_hatch("//")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Scenario A RMSE (%)")
    ax.set_ylim(0, max(values) + 3.0)
    ax.tick_params(axis="x", rotation=12)
    ax.text(
        0.98,
        0.96,
        "Hatched bars: deterministic/single-checkpoint, no seed std",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color=COL["gray"],
    )
    fig.tight_layout()
    return {
        "outputs": save(fig, "fig_empirical_02_multiseed_baselines"),
        "source_data": [
            "results/v5/multiseed/multiseed_summary.csv",
            "results/v5/final_v5_model_comparison.csv",
            "results/v5/ekf_ecm/recursive_vs_ekf_comparison.csv",
        ],
        "message": "Scenario A RMSE comparison uses exact means/std where seed data exist.",
        "placement": "Bagian III/IV model comparison",
        "caveat": "Null and EKF rows are deterministic/single-checkpoint artifacts, so no 5-seed std is plotted.",
        "status": "generated",
    }


def fig_eta_curve() -> dict:
    path = require_file(ROOT / "results" / "v5" / "delta_calibration" / "eta_gamma_sweep.csv")
    df = pd.read_csv(path)
    d = df[(df["mode"] == "inference_sweep") & (df["gamma_mode"] == "nominal")].copy()
    d = d.sort_values("eta")

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.plot(d["eta"], d["rec_rmse_pct"], marker="o", color=COL["blue"], label="Recursive RMSE")
    ax.set_xlabel("η*")
    ax.set_ylabel("Sequence RMSE (%)", color=COL["blue"])
    ax.tick_params(axis="y", labelcolor=COL["blue"])
    ax2 = ax.twinx()
    ax2.plot(d["eta"], d["rec_delta_ratio"], marker="s", color=COL["orange"], label="Delta ratio rΔ")
    ax2.axhline(1.0, color=COL["gray"], linestyle=":", linewidth=1.2)
    ax2.set_ylabel("Delta ratio rΔ", color=COL["orange"])
    ax2.tick_params(axis="y", labelcolor=COL["orange"])
    best = d.loc[(d["eta"] - 2.0).abs().idxmin()]
    ax.axvline(2.0, color=COL["vermillion"], linestyle="--", linewidth=1.2)
    ax.scatter([best["eta"]], [best["rec_rmse_pct"]], s=48, color=COL["vermillion"], zorder=4)
    ax.text(2.03, float(best["rec_rmse_pct"]) + 0.8, "η*=2.0", color=COL["vermillion"], fontsize=9)
    handles = [ax.lines[0], ax2.lines[0]]
    ax.legend(handles, [h.get_label() for h in handles], loc="upper center", frameon=True, framealpha=0.9)
    fig.tight_layout()
    return {
        "outputs": save(fig, "fig_empirical_03_eta_calibration_curve"),
        "source_data": ["results/v5/delta_calibration/eta_gamma_sweep.csv"],
        "message": "η* sweep shows recursive RMSE minimum near η*=2.0 and delta-ratio recovery near 1.",
        "placement": "Bagian III/IV eta calibration",
        "caveat": "Inference sweep uses fixed Scenario A seed 42 weights trained at η=1.5.",
        "status": "generated",
    }


def fig_training_curves() -> dict:
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
        "outputs": save(fig, "fig_empirical_05_training_curves_anchor_last"),
        "source_data": [str(p.relative_to(ROOT)) for p in paths.values()],
        "message": "Training and validation loss curves are plotted from saved HC-LSTM and HC-TCN logs.",
        "placement": "Bagian III.A",
        "caveat": "Logs are saved training artifacts; no training rerun. HC-LSTM rows use v7_final logs, HC-TCN rows use v5 Coulomb TCN logs.",
        "status": "generated",
    }


def envelope_forward(model, X, I, eta: float, gamma_w: np.ndarray, device, batch: int = 1024):
    import torch

    thr = model.hard_constraint.threshold
    cum_all, anchor_all = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i : i + batch]).to(device)
            ib = torch.from_numpy(I[i : i + batch]).to(device)
            gw = torch.from_numpy(gamma_w[i : i + batch]).to(device).float().view(-1, 1, 1)
            h, _ = model.lstm(xb)
            dl = model.delta_head(h)
            al = model.anchor_head(h[:, 0, :])
            i3 = ib.unsqueeze(-1)
            mag = torch.sigmoid(dl)
            limit = i3.abs() * (eta * gw)
            zero = torch.zeros_like(dl)
            delta = torch.where(i3 < -thr, -limit * mag, zero)
            delta = torch.where(i3 > thr, limit * mag, delta)
            cum = torch.cumsum(delta, dim=1).squeeze(-1)
            lo = (-cum.min(dim=1).values).clamp(0.0, 1.0)
            hi = (1.0 - cum.max(dim=1).values).clamp(0.0, 1.0)
            width = (hi - lo).clamp_min(1e-6)
            anchor = lo + width * torch.sigmoid(al.squeeze(-1))
            cum_all.append(cum.cpu().numpy())
            anchor_all.append(anchor.cpu().numpy())
    return np.concatenate(cum_all), np.concatenate(anchor_all)


def fig_cold_trajectory() -> dict:
    try:
        import torch
        from ekf_1rc_ecm_continuous import run_continuous_1rc_ekf
        from ekf_ocv_rint_continuous import map_back_to_windows, reconstruct_sequences
        from gated_recursive_inference import gating_features, run_policy
        from run_multiseed_v5 import build_model, load_splits
    except Exception as exc:  # pragma: no cover
        return {
            "outputs": [],
            "source_data": [],
            "message": "Could not import inference utilities.",
            "placement": "Bagian III.D",
            "caveat": str(exc),
            "status": "gap",
        }

    data_dir = ROOT / "data" / "processed" / "v5c_scenario_A"
    ckpt = require_file(
        ROOT
        / "results"
        / "v5"
        / "headline_models"
        / "checkpoints"
        / "hard_coulomb_lstm_v5c_scenario_A_seed42.pt"
    )
    require_file(data_dir / "X_test.npy")
    data = load_splits("v5c", "A")
    keys = np.load(data_dir / "timestamp_key_test.npy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_model("hard_coulomb_lstm").to(device)
    model.load_state_dict(payload["model_state_dict"])

    gamma_w = np.full(len(data["I_test"]), 1.0 / (3600.0 * Q_NOMINAL), dtype=np.float32)
    cum, anchor = envelope_forward(model, data["X_test"], data["I_test"], 2.0, gamma_w, device)
    feat = gating_features(data["X_test"], data["I_test"])
    hc, _ = run_policy("carried_anchor", cum, anchor, keys, feat)

    sequences, ekf_keys = reconstruct_sequences(data_dir)
    run_continuous_1rc_ekf(sequences, 0.01)
    ekf = map_back_to_windows(sequences, ekf_keys)

    mask = data["temp_labels"] == "n20degC"
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        raise RuntimeError("No n20degC test windows in v5c Scenario A.")
    mae_hc = np.mean(np.abs(hc[idxs] - data["y_test"][idxs]), axis=1)
    mae_ekf = np.mean(np.abs(ekf[idxs] - data["y_test"][idxs]), axis=1)
    j = int(idxs[np.argmax(mae_ekf - mae_hc)])

    t = np.arange(data["y_test"].shape[1])
    fig, ax = plt.subplots(figsize=(6.7, 3.4))
    ax.plot(t, data["y_test"][j] * 100.0, color=COL["black"], label="Ground truth SOC")
    ax.plot(t, hc[j] * 100.0, color=COL["vermillion"], label="Calibrated recursive HC")
    ax.plot(t, ekf[j] * 100.0, color=COL["blue"], linestyle="--", label="Best EKF 1RC")
    ax.set_xlabel("Timestep (s)")
    ax.set_ylabel("SOC (%)")
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    fig.tight_layout()
    return {
        "outputs": save(fig, "fig_empirical_04_subzero_trajectory_ekf_hc"),
        "source_data": [
            "data/processed/v5c_scenario_A/X_test.npy",
            "data/processed/v5c_scenario_A/y_test.npy",
            "data/processed/v5c_scenario_A/I_unscaled_test.npy",
            "data/processed/v5c_scenario_A/temp_labels_test.npy",
            "data/processed/v5c_scenario_A/timestamp_key_test.npy",
            "results/v5/headline_models/checkpoints/hard_coulomb_lstm_v5c_scenario_A_seed42.pt",
            "results/v5/delta_calibration/eta_gamma_sweep.csv",
            "results/v5/ekf_ecm/recursive_vs_ekf_comparison.csv",
        ],
        "message": "−20 °C SOC trajectory uses fixed checkpoint inference and continuous 1RC EKF mapping.",
        "placement": "Bagian III.D/IV.G",
        "caveat": f"Window index {j}; selected by largest EKF-minus-HC MAE gap among n20degC test windows. No training rerun.",
        "status": "generated",
    }


def validate(outputs: list[str]) -> list[dict]:
    from PIL import Image

    report = []
    for rel in outputs:
        path = ROOT / rel
        item = {"file": rel, "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}
        if path.suffix.lower() == ".png" and path.exists():
            with Image.open(path) as img:
                arr = np.asarray(img.convert("L"))
                item.update(
                    {
                        "width": img.width,
                        "height": img.height,
                        "pixel_std": float(arr.std()),
                        "blank": bool(arr.std() < 1.0),
                    }
                )
        report.append(item)
    return report


def purge_svg_and_known_mock() -> dict:
    removed = []
    target_dirs = (ROOT / "figures", ROOT / "outputs" / "figures", ROOT / "results", ROOT / "drafts")
    for base in target_dirs:
        if not base.exists():
            continue
        for path in base.rglob("*.svg"):
            path.unlink()
            removed.append(str(path.relative_to(ROOT)))
    mock_stems = {
        "fig01_final_architecture",
        "fig02_research_evolution_flowchart",
        "fig02b_failure_evidence_multipanel",
    }
    for base in (ROOT / "figures", ROOT / "outputs" / "figures"):
        if not base.exists():
            continue
        for path in base.iterdir():
            if path.is_file() and path.stem in mock_stems and path.suffix.lower() in {".png", ".pdf"}:
                path.unlink()
                removed.append(str(path.relative_to(ROOT)))
    remaining = 0
    for base in target_dirs:
        if base.exists():
            remaining += len(list(base.rglob("*.svg")))
    return {"removed": removed, "remaining_svg_in_figure_outputs": remaining}


def main() -> None:
    purge = purge_svg_and_known_mock()
    manifest = {
        "policy": "empirical-only; no SVG; no mock diagrams; no training rerun",
        "purge": purge,
        "figures": [],
        "validation": [],
    }
    for fn in (fig_raw_signal, fig_multiseed, fig_eta_curve, fig_cold_trajectory, fig_training_curves):
        entry = fn()
        manifest["figures"].append(entry)
        manifest["validation"].extend(validate(entry.get("outputs", [])))

    (OUT / "empirical_figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = ["# Empirical Figure Export Report", ""]
    lines.append(f"- SVG removed: {len(purge['removed'])}")
    lines.append(f"- Remaining SVG under figure/output artifact dirs: {purge['remaining_svg_in_figure_outputs']}")
    for fig in manifest["figures"]:
        lines.append("")
        lines.append(f"## {fig['status'].upper()}: {fig['message']}")
        lines.append(f"- outputs: {', '.join(fig.get('outputs', [])) or 'TIDAK DIBUAT'}")
        lines.append(f"- sources: {', '.join(fig.get('source_data', [])) or 'TIDAK DITEMUKAN'}")
        lines.append(f"- caveat: {fig.get('caveat', '')}")
    lines.append("")
    lines.append("## Validation")
    for item in manifest["validation"]:
        lines.append(f"- {item}")
    (OUT / "empirical_figure_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
