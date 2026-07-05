from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
REPORT = ROOT / "outputs" / "manuscript_assets" / "publication_ready_additions_report.json"

RAW_25 = ROOT / "data/raw/LG Dataset/LG_HG2_Original_Dataset/25degC/551_UDDS.csv"
RAW_N20 = ROOT / "data/raw/LG Dataset/LG_HG2_Original_Dataset/n20degC/610_UDDS.csv"
RAW_ROOT = ROOT / "data/raw/LG Dataset/LG_HG2_Original_Dataset"

MODEL_TABLE = ROOT / "outputs/tables/table03_main_model_comparison.csv"
SEED_TABLE = ROOT / "results/v5/multiseed/seed_level_results.csv"
VANILLA_AGG = ROOT / "results/v5/multiseed/runs_A.json"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.5,
})

COLORS = {
    "black": "#222222",
    "blue": "#0072B2",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#6D597A",
    "gray": "#8A9BA8",
}


def load_lg_csv(path: Path) -> pd.DataFrame:
    lines = path.read_text(errors="ignore").splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("Time Stamp,Step,Status"))
    df = pd.read_csv(path, skiprows=header_idx)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    for col in ["Prog Time", "Voltage", "Current", "Temperature", "Capacity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Voltage", "Current", "Temperature", "Capacity"]).reset_index(drop=True)


def add_soc(df: pd.DataFrame, q_actual: float) -> pd.DataFrame:
    out = df.copy()
    out["SOC_actual"] = np.clip(1.0 + out["Capacity"].to_numpy() / q_actual, 0, 1) * 100
    if out["Prog Time"].max() > 0:
        out["t_s"] = out["Prog Time"] - out["Prog Time"].iloc[0]
    else:
        out["t_s"] = np.arange(len(out)) * 0.1
    return out


def normalize(series: pd.Series) -> np.ndarray:
    arr = series.to_numpy(dtype=float)
    lo, hi = np.nanpercentile(arr, [1, 99])
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def save(fig: plt.Figure, stem: str) -> None:
    # Empirical remediation policy: never emit SVG for manuscript figures.
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{stem}.{ext}")
    plt.close(fig)


def make_eda() -> dict:
    d25 = add_soc(load_lg_csv(RAW_25), 2.7744)
    dn20 = add_soc(load_lg_csv(RAW_N20), 2.3304)

    # Keep same 900 s slice for print readability.
    d25s = d25[(d25["t_s"] >= 0) & (d25["t_s"] <= 900)].iloc[::10].copy()
    dn20s = dn20[(dn20["t_s"] >= 0) & (dn20["t_s"] <= 900)].iloc[::10].copy()

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), gridspec_kw={"width_ratios": [1.25, 1.25, 0.9]})
    for ax, df, title in [(axes[0], d25s, "(a) UDDS 25 °C"), (axes[1], dn20s, "(b) UDDS −20 °C")]:
        t = df["t_s"] / 60.0
        ax.plot(t, normalize(df["Voltage"]), color=COLORS["blue"], label="V_terminal")
        ax.plot(t, normalize(df["Current"]), color=COLORS["orange"], label="I")
        ax.plot(t, normalize(df["Temperature"]), color=COLORS["green"], label="T")
        ax.plot(t, normalize(df["SOC_actual"]), color=COLORS["black"], label="SOC_actual")
        ax.set_title(title)
        ax.set_xlabel("Waktu (min)")
        ax.set_ylabel("Nilai ternormalisasi")
        ax.legend(loc="lower left", frameon=True, framealpha=0.88)

    currents = []
    for cycle in ("*UDDS.csv", "*LA92.csv", "*US06.csv"):
        for p in RAW_ROOT.glob(f"*degC/{cycle}"):
            try:
                currents.append(load_lg_csv(p)["Current"].dropna().to_numpy())
            except Exception:
                pass
    all_i = np.concatenate(currents)
    axes[2].hist(all_i, bins=80, color=COLORS["purple"], alpha=0.88)
    axes[2].set_title("(c) Distribusi arus")
    axes[2].set_xlabel("I (A)")
    axes[2].set_ylabel("Frekuensi")
    axes[2].legend(["UDDS/LA92/US06"], loc="upper left", frameon=True, framealpha=0.88)
    save(fig, "fig01a_raw_signal_eda")
    return {
        "asset": "outputs/figures/fig01a_raw_signal_eda.png",
        "sources": [str(RAW_25.relative_to(ROOT)), str(RAW_N20.relative_to(ROOT)), str(RAW_ROOT.relative_to(ROOT))],
        "message": "Raw voltage/current/temperature/SOC contrast at 25 °C and −20 °C plus drive-cycle current diversity.",
    }


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def vanilla_pvr_from_runs(path: Path) -> float | None:
    runs = load_json(path)
    vals = [
        r["metrics"]["pvr"]["discharge"]["by_epsilon"]["0"]["rate_pct"]
        for r in runs
        if r.get("model") == "vanilla_lstm"
    ]
    return float(np.mean(vals)) if vals else None


def make_failure_panel() -> dict:
    comp = pd.read_csv(MODEL_TABLE)
    seed = pd.read_csv(SEED_TABLE)
    vanilla_pvr = vanilla_pvr_from_runs(VANILLA_AGG)

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.7))

    ax = axes[0, 0]
    if vanilla_pvr is not None:
        ax.bar(["Vanilla\naggregate"], [vanilla_pvr], color=COLORS["vermillion"])
        ax.set_ylim(0, 60)
        ax.set_ylabel("PVR discharge (%)")
        ax.set_title("(a) Physics-blind aggregate")
        ax.text(0, vanilla_pvr + 2, f"{vanilla_pvr:.1f}%", ha="center", fontsize=8)
        ax.text(0, 8, "GAP: prediksi per-sequence\nvanilla tidak tersimpan", ha="center", fontsize=7.2)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "GAP: vanilla trajectory\nraw prediction not found", ha="center", va="center")

    ax = axes[0, 1]
    ax.axis("off")
    ax.text(
        0.5,
        0.58,
        "GAP: log Soft-PINN\nMSE vs penalty tidak tersimpan",
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF4E6", ec="#D55E00", lw=0.8),
    )
    ax.text(0.5, 0.32, "Notebook/legacy figure ada,\nraw loss-component CSV tidak ditemukan", ha="center", va="center", fontsize=7.4)
    ax.set_title("(b) Soft-PINN gradient collision")

    ax = axes[1, 0]
    rows = comp[comp["method"].isin(["Vanilla LSTM", "Post-hoc clamp", "anchor_last"])].copy()
    labels = rows["method"].replace({"Post-hoc clamp": "Post-hoc\nclamp", "Vanilla LSTM": "Vanilla", "anchor_last": "anchor_last"})
    vals = rows["scenario_A_RMSE"].astype(str).str.extract(r"([0-9.]+)").astype(float)[0]
    bars = ax.bar(labels, vals, color=[COLORS["gray"], COLORS["vermillion"], COLORS["green"]])
    ax.set_ylabel("RMSE Skenario A (%)")
    ax.set_title("(c) Post-hoc clamp collapse")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=7.5)

    ax = axes[1, 1]
    sub = seed[seed["model"].isin(["hard_coulomb_lstm", "hc_anchor_last"])].copy()
    pivot = sub.pivot_table(index="seed", columns="model", values="rmse_pct", aggfunc="first")
    x = np.arange(len(pivot.index))
    ax.bar(x - 0.18, pivot["hard_coulomb_lstm"], 0.36, label="HC anchor-first", color=COLORS["gray"])
    ax.bar(x + 0.18, pivot["hc_anchor_last"], 0.36, label="anchor_last", color=COLORS["green"])
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in pivot.index])
    ax.set_xlabel("Seed")
    ax.set_ylabel("RMSE (%)")
    ax.set_title("(d) Anchor-first bottleneck")
    ax.legend(loc="upper right", frameon=True, framealpha=0.88)

    fig.tight_layout()
    save(fig, "fig02b_failure_evidence_multipanel")
    return {
        "asset": "outputs/figures/fig02b_failure_evidence_multipanel.png",
        "sources": [str(MODEL_TABLE.relative_to(ROOT)), str(SEED_TABLE.relative_to(ROOT)), str(VANILLA_AGG.relative_to(ROOT))],
        "message": "Partial failure-evidence visualization; raw vanilla trajectory and Soft-PINN loss-component logs are explicit gaps.",
        "caveat": "Panels (a) and (b) do not replace missing raw per-sequence/loss-component artifacts.",
    }


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    entries = {
        "fig01a_raw_signal_eda": make_eda(),
        "fig02b_failure_evidence_multipanel": make_failure_panel(),
        "gaps": {
            "anchor_last_training_curves": "No anchor_last history in logs or checkpoint; checkpoints store best_val_loss and epochs only.",
            "soc_trajectory_hc_vs_ekf": "No persisted calibrated HC and best-EKF per-timestep predictions in one comparable file.",
            "soft_pinn_loss_components": "No raw MSE-vs-penalty training log found; only legacy notebook/figure artifact.",
            "vanilla_discharge_sequence_prediction": "No per-sequence vanilla prediction artifact found; aggregate PVR metrics exist.",
        },
    }
    REPORT.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(entries, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
