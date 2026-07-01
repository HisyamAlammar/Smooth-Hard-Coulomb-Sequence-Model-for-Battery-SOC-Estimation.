"""Generate Q1 EDA evidence plots for observability and transient dynamics.

Outputs are written directly to outputs/figures/:
  fig_q1_observability_collapse.{png,pdf}
  fig_q1_transient_dynamic_profile.{png,pdf}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.preprocessing_v4 import (  # noqa: E402
    DATA_RAW,
    R_INT_PER_TEMP,
    build_ocv_soc_lookup,
    engineer_features_v4,
    read_csv,
    to_strict_1hz_segments,
)

FIG_DIR = BASE_DIR / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_KEYS = ("udds", "la92", "hwfet", "us06", "mixed")
COL_25 = "#0072B2"
COL_N20 = "#D55E00"
COL_DARK = "#264653"
COL_CHARGE = "#009E73"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
})


def drive_files(temp: str) -> list[Path]:
    root = Path(DATA_RAW) / temp
    return sorted(path for path in root.glob("*.csv") if any(key in path.name.lower() for key in DRIVE_KEYS))


def load_engineered_drive_data(temp: str, min_len: int = 600) -> pd.DataFrame:
    ocv_lookup, q_actual = build_ocv_soc_lookup(temp)
    frames: list[pd.DataFrame] = []
    profile_code = 1
    for path in drive_files(temp):
        raw = read_csv(str(path))
        segments, profile_code, _stats = to_strict_1hz_segments(
            raw,
            source_id=f"{temp}:{path.name}",
            profile_code_start=profile_code,
            min_len=min_len,
        )
        for seg in segments:
            engineered, _soc0 = engineer_features_v4(
                seg,
                q_actual=q_actual,
                r_int=R_INT_PER_TEMP[temp],
                ocv_lookup=ocv_lookup,
            )
            engineered["source_file"] = path.name
            engineered["temp_name"] = temp
            frames.append(engineered)
    if not frames:
        raise RuntimeError(f"No usable drive segments for {temp}")
    return pd.concat(frames, ignore_index=True)


def binned_voltage_stats(df: pd.DataFrame, soc_bins: np.ndarray) -> pd.DataFrame:
    rows = []
    for lo, hi in zip(soc_bins[:-1], soc_bins[1:]):
        mask = (df["SOC_cc"] >= lo) & (df["SOC_cc"] < hi)
        values = df.loc[mask, "Voltage"].dropna().to_numpy(dtype=np.float64)
        if len(values) >= 30:
            rows.append({
                "soc": 0.5 * (lo + hi),
                "v_med": float(np.median(values)),
                "v_p25": float(np.percentile(values, 25)),
                "v_p75": float(np.percentile(values, 75)),
                "n": int(len(values)),
            })
    return pd.DataFrame(rows)


def save_both(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_observability_collapse(df25: pd.DataFrame, dfn20: pd.DataFrame) -> None:
    soc_bins = np.linspace(0.05, 1.00, 20)
    stats25 = binned_voltage_stats(df25, soc_bins)
    stats20 = binned_voltage_stats(dfn20, soc_bins)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for df, color in [(df25, COL_25), (dfn20, COL_N20)]:
        step = max(1, len(df) // 6000)
        ax.scatter(
            df["SOC_cc"].iloc[::step],
            df["Voltage"].iloc[::step],
            s=3,
            alpha=0.10,
            color=color,
            linewidths=0,
        )

    ax.plot(stats25["soc"], stats25["v_med"], color=COL_25, lw=2.0, label="25 C median")
    ax.fill_between(stats25["soc"], stats25["v_p25"], stats25["v_p75"], color=COL_25, alpha=0.16)
    ax.plot(stats20["soc"], stats20["v_med"], color=COL_N20, lw=2.0, label="-20 C median")
    ax.fill_between(stats20["soc"], stats20["v_p25"], stats20["v_p75"], color=COL_N20, alpha=0.16)

    rho25 = df25[["SOC_cc", "Voltage"]].corr().iloc[0, 1]
    rho20 = dfn20[["SOC_cc", "Voltage"]].corr().iloc[0, 1]
    ax.set_xlabel("SOC from Coulomb counting")
    ax.set_ylabel("Terminal voltage (V)")
    ax.set_title("Observability Collapse: Voltage-SOC Mapping Shifts at -20 C")
    ax.text(
        0.03,
        0.04,
        f"Pearson r: 25 C = {rho25:.3f}, -20 C = {rho20:.3f}",
        transform=ax.transAxes,
        fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.92),
    )
    ax.legend(frameon=False, loc="upper left")
    save_both(fig, "fig_q1_observability_collapse")


def most_dynamic_500s(df: pd.DataFrame, seconds: int = 500) -> pd.DataFrame:
    best_score = -np.inf
    best_segment: pd.DataFrame | None = None
    for _source, group in df.groupby("source_file"):
        group = group.sort_values("time_sec").reset_index(drop=True)
        if len(group) < seconds:
            continue
        score = group["Current"].diff().abs().rolling(seconds, min_periods=seconds).mean()
        if not score.notna().any():
            continue
        end = int(score.idxmax())
        value = float(score.iloc[end])
        start = max(0, end - seconds + 1)
        segment = group.iloc[start:start + seconds].copy()
        if len(segment) == seconds and value > best_score:
            best_score = value
            best_segment = segment
    if best_segment is None:
        raise RuntimeError("No 500 s dynamic segment found")
    return best_segment


def plot_transient_dynamic_profile(df25: pd.DataFrame) -> None:
    seg = most_dynamic_500s(df25, seconds=500)
    t = seg["time_sec"].to_numpy(dtype=np.float64)
    t = t - t[0]
    current = seg["Current"].to_numpy(dtype=np.float64)
    voltage = seg["Voltage"].to_numpy(dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.2), sharex=True)
    axes[0].plot(t, current, color=COL_DARK, lw=1.2)
    axes[0].fill_between(t, current, 0, where=current < -0.05, color=COL_N20, alpha=0.18, label="Discharge")
    axes[0].fill_between(t, current, 0, where=current > 0.05, color=COL_CHARGE, alpha=0.18, label="Regeneration")
    axes[0].axhline(0, color="0.35", lw=0.8)
    axes[0].set_ylabel("Current (A)")
    axes[0].set_title("Transient Dynamic Profile: EV Load Excites Voltage Sag")
    axes[0].legend(frameon=False, loc="best")

    axes[1].plot(t, voltage, color=COL_25, lw=1.2)
    high_load = current < np.percentile(current, 10)
    if high_load.any():
        axes[1].scatter(t[high_load], voltage[high_load], s=8, color=COL_N20, alpha=0.65, label="High discharge load")
        axes[1].legend(frameon=False, loc="best")
    axes[1].set_xlabel("Time within selected segment (s)")
    axes[1].set_ylabel("Voltage (V)")
    fig.text(
        0.01,
        0.01,
        f"Source: {seg['source_file'].iloc[0]} | selected by max rolling mean |dI/dt| over 500 s",
        fontsize=7.8,
    )
    save_both(fig, "fig_q1_transient_dynamic_profile")


def main() -> None:
    print("Loading 25 C drive profiles...")
    df25 = load_engineered_drive_data("25degC")
    print("Loading -20 C drive profiles...")
    dfn20 = load_engineered_drive_data("n20degC")
    plot_observability_collapse(df25, dfn20)
    plot_transient_dynamic_profile(df25)
    print(f"Saved money plots to: {FIG_DIR}")


if __name__ == "__main__":
    main()
