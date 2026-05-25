"""
hppc_rint_extractor.py -- Reproducible HPPC R_int extraction
================================================================

Standalone reviewer-facing proof script for the R_int values used by the
V_proxy feature:

    V_proxy(t) = V_t(t) - I(t) * R_int(T)

The script loads each raw Kollmeyer LG HG2 HPPC profile, detects
Rest-to-Discharge transitions, computes the instantaneous Ohmic resistance
from the first acquired pulse sample, and compares the calculated means with
the paper/config hardcoded values.

Notes for reviewers
-------------------
Ohmic resistance R0 is the immediate voltage step at pulse onset:

    R_int = ΔV / ΔI

where ΔV = V_pulse_first_sample - V_rest_last_sample and
ΔI = I_pulse_first_sample - I_rest_last_sample.

Using a delayed 1 s sample would include polarization/diffusion dynamics and
does not reproduce the paper values. The hardcoded table in config.py is the
instantaneous REST→DCH step average from the raw HPPC files.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = BASE_DIR / "data" / "raw" / "LG Dataset" / "LG_HG2_Original_Dataset"

REST_CURRENT_THRESHOLD_A = 0.05
DISCHARGE_CURRENT_THRESHOLD_A = -0.10
MAX_VALID_RINT_OHM = 0.50

TEMPERATURE_ORDER = ["40degC", "25degC", "10degC", "0degC", "n10degC", "n20degC"]

TEMP_DISPLAY = {
    "40degC": "40°C",
    "25degC": "25°C",
    "10degC": "10°C",
    "0degC": "0°C",
    "n10degC": "-10°C",
    "n20degC": "-20°C",
}

PAPER_RINT_MOHM = {
    "40degC": 16.51,
    "25degC": 19.86,
    "10degC": 28.75,
    "0degC": 40.08,
    "n10degC": 62.19,
    "n20degC": 109.83,
}


@dataclass(frozen=True)
class PulseRint:
    rest_index: int
    pulse_index: int
    rest_voltage: float
    pulse_voltage: float
    rest_current: float
    pulse_current: float
    delta_voltage: float
    delta_current: float
    r_int_ohm: float

    @property
    def r_int_mohm(self) -> float:
        return self.r_int_ohm * 1000.0


def find_header_row(filepath: Path) -> int:
    with filepath.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            if "Voltage" in line and "Current" in line:
                return line_number
    raise ValueError(f"Header row not found in {filepath}")


def load_hppc_csv(filepath: Path) -> pd.DataFrame:
    header_row = find_header_row(filepath)
    df = pd.read_csv(
        filepath,
        skiprows=header_row,
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=False,
    )
    df.columns = df.columns.str.strip()

    required = ["Voltage", "Current"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{filepath} missing required columns: {missing}")

    for column in ["Voltage", "Current", "Step", "Temperature", "Capacity"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Voltage", "Current"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{filepath} contains no valid Voltage/Current rows")

    return df


def locate_hppc_file(data_root: Path, temp_key: str) -> Path:
    temp_dir = data_root / temp_key
    files = sorted(temp_dir.glob("*HPPC*.csv"))
    if not files:
        raise FileNotFoundError(f"No HPPC CSV found under {temp_dir}")
    if len(files) > 1:
        names = ", ".join(file.name for file in files)
        raise RuntimeError(f"Expected one HPPC file for {temp_key}, found: {names}")
    return files[0]


def is_rest(current_a: float, threshold_a: float = REST_CURRENT_THRESHOLD_A) -> bool:
    return abs(current_a) <= threshold_a


def is_discharge_pulse_start(
    previous_current_a: float,
    current_a: float,
    rest_threshold_a: float = REST_CURRENT_THRESHOLD_A,
    discharge_threshold_a: float = DISCHARGE_CURRENT_THRESHOLD_A,
) -> bool:
    return is_rest(previous_current_a, rest_threshold_a) and current_a <= discharge_threshold_a


def compute_transition_rint(
    df: pd.DataFrame,
    pulse_index: int,
    max_valid_rint_ohm: float = MAX_VALID_RINT_OHM,
) -> PulseRint | None:
    rest_index = pulse_index - 1
    rest_voltage = float(df.at[rest_index, "Voltage"])
    pulse_voltage = float(df.at[pulse_index, "Voltage"])
    rest_current = float(df.at[rest_index, "Current"])
    pulse_current = float(df.at[pulse_index, "Current"])

    delta_voltage = pulse_voltage - rest_voltage
    delta_current = pulse_current - rest_current
    if abs(delta_current) <= 1e-9:
        return None

    r_int_ohm = delta_voltage / delta_current
    if not np.isfinite(r_int_ohm):
        return None
    if r_int_ohm <= 0.0 or r_int_ohm > max_valid_rint_ohm:
        return None

    return PulseRint(
        rest_index=rest_index,
        pulse_index=pulse_index,
        rest_voltage=rest_voltage,
        pulse_voltage=pulse_voltage,
        rest_current=rest_current,
        pulse_current=pulse_current,
        delta_voltage=delta_voltage,
        delta_current=delta_current,
        r_int_ohm=float(r_int_ohm),
    )


def extract_rest_to_discharge_rints(df: pd.DataFrame) -> List[PulseRint]:
    current = df["Current"].to_numpy(dtype=np.float64)
    pulses: List[PulseRint] = []

    for index in range(1, len(df)):
        if not is_discharge_pulse_start(current[index - 1], current[index]):
            continue
        pulse = compute_transition_rint(df, index)
        if pulse is not None:
            pulses.append(pulse)

    return pulses


def summarize_pulses(pulses: Iterable[PulseRint]) -> Dict[str, float]:
    values = np.asarray([pulse.r_int_mohm for pulse in pulses], dtype=np.float64)
    if values.size == 0:
        raise ValueError("No valid REST→DCH pulses were detected")
    return {
        "count": int(values.size),
        "mean_mohm": float(values.mean()),
        "std_mohm": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min_mohm": float(values.min()),
        "max_mohm": float(values.max()),
    }


def format_table(rows: List[Dict[str, object]]) -> str:
    headers = [
        "Temp",
        "HPPC File",
        "Pulses",
        "Script Calculated R_int (mΩ)",
        "Paper Hardcoded R_int (mΩ)",
        "Δ (mΩ)",
        "Status",
    ]
    widths = [8, 14, 6, 30, 28, 8, 8]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    sep = "  ".join("-" * width for width in widths)

    lines = [fmt.format(*headers), sep]
    for row in rows:
        lines.append(
            fmt.format(
                row["temp"],
                row["file"],
                row["pulses"],
                f"{row['script_mohm']:.2f}",
                f"{row['paper_mohm']:.2f}",
                f"{row['delta_mohm']:+.2f}",
                row["status"],
            )
        )
    return "\n".join(lines)


def run_extraction(data_root: Path, tolerance_mohm: float) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, float]]]:
    rows: List[Dict[str, object]] = []
    summaries: Dict[str, Dict[str, float]] = {}

    for temp_key in TEMPERATURE_ORDER:
        hppc_file = locate_hppc_file(data_root, temp_key)
        df = load_hppc_csv(hppc_file)
        pulses = extract_rest_to_discharge_rints(df)
        summary = summarize_pulses(pulses)
        paper_mohm = PAPER_RINT_MOHM[temp_key]
        delta_mohm = summary["mean_mohm"] - paper_mohm
        status = "PASS" if abs(delta_mohm) <= tolerance_mohm else "CHECK"

        summaries[temp_key] = {
            **summary,
            "paper_mohm": paper_mohm,
            "delta_mohm": delta_mohm,
        }
        rows.append(
            {
                "temp": TEMP_DISPLAY[temp_key],
                "file": hppc_file.name,
                "pulses": summary["count"],
                "script_mohm": summary["mean_mohm"],
                "paper_mohm": paper_mohm,
                "delta_mohm": delta_mohm,
                "status": status,
            }
        )

    return rows, summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract REST→DCH Ohmic R_int from raw Kollmeyer LG HG2 HPPC CSV files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Raw LG_HG2_Original_Dataset directory. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--tolerance-mohm",
        type=float,
        default=0.01,
        help="Allowed absolute difference versus paper/config values in mΩ. Default: 0.01",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    print("=" * 112)
    print("  HPPC R_int Reproducibility Extractor")
    print("=" * 112)
    print(f"  Dataset root : {data_root}")
    print("  Transition   : REST current |I| ≤ 0.05 A → discharge current I ≤ -0.10 A")
    print("  Formula      : R_int = ΔV / ΔI using first acquired discharge-pulse sample")
    print("  Units        : milliohm (mΩ)")
    print()

    rows, summaries = run_extraction(data_root, tolerance_mohm=args.tolerance_mohm)
    print(format_table(rows))

    failures = [row for row in rows if row["status"] != "PASS"]
    print()
    if failures:
        print("VERIFICATION FAILED: at least one calculated value differs from the paper/config table.")
        raise SystemExit(1)

    print("VERIFICATION PASSED: Script-calculated R_int values reproduce the paper/config table.")
    print()
    print("Detailed pulse statistics:")
    for temp_key in TEMPERATURE_ORDER:
        summary = summaries[temp_key]
        print(
            f"  {TEMP_DISPLAY[temp_key]:>5}: "
            f"n={summary['count']:>2}, "
            f"mean={summary['mean_mohm']:.4f} mΩ, "
            f"std={summary['std_mohm']:.4f} mΩ, "
            f"range=[{summary['min_mohm']:.4f}, {summary['max_mohm']:.4f}] mΩ"
        )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
