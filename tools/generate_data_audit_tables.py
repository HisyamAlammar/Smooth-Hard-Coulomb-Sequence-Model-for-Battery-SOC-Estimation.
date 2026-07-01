"""Generate raw integrity and v4 scenario composition Markdown tables.

Outputs:
  outputs/data_audit_tables.md
  outputs/data_audit_tables.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.preprocessing_v4 import (  # noqa: E402
    DATA_RAW,
    Q_ACTUAL_PER_TEMP,
    build_ocv_soc_lookup,
    find_header_row,
    read_csv,
)

TEMP_ORDER = ["40degC", "25degC", "10degC", "0degC", "n10degC", "n20degC"]
TEMP_LABEL = {
    "40degC": "40 C",
    "25degC": "25 C",
    "10degC": "10 C",
    "0degC": "0 C",
    "n10degC": "-10 C",
    "n20degC": "-20 C",
}
DRIVE_KEYS = ("udds", "la92", "hwfet", "us06", "mixed")
V_MIN = 2.5
V_MAX = 4.25
I_ABS_LIMIT = 20.0


def is_drive_file(path: Path) -> bool:
    name = path.name.lower()
    return any(key in name for key in DRIVE_KEYS)


def raw_csv_with_quality_columns(path: Path) -> pd.DataFrame:
    header = find_header_row(str(path))
    df = pd.read_csv(
        path,
        skiprows=header,
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=False,
    )
    df.columns = df.columns.str.strip()
    for col in ["Voltage", "Current", "Temperature", "Capacity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Time Stamp" in df.columns:
        ts = pd.to_datetime(df["Time Stamp"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        if ts.isna().all():
            ts = pd.to_datetime(df["Time Stamp"], errors="coerce", format="mixed")
        df["_timestamp"] = ts
        valid_ts = ts.notna()
        if valid_ts.any():
            t0 = ts[valid_ts].iloc[0]
            df["time_sec"] = (ts - t0).dt.total_seconds()
        else:
            df["time_sec"] = np.arange(len(df), dtype=np.float64) * 0.1
    else:
        df["time_sec"] = np.arange(len(df), dtype=np.float64) * 0.1
    return df


def unclipped_soc_for_file(temp: str, path: Path, ocv_lookup) -> np.ndarray:
    df = read_csv(str(path))
    if df.empty:
        return np.empty((0,), dtype=np.float64)
    q_actual = Q_ACTUAL_PER_TEMP[temp]
    soc_initial = float(np.clip(ocv_lookup(df["Voltage"].iloc[0]), 0.0, 1.0)) if ocv_lookup else 1.0
    if "Capacity" in df.columns and df["Capacity"].notna().sum() > 0:
        cap = df["Capacity"].fillna(0.0)
        cap_used = (cap - cap.iloc[0]).abs()
        return (soc_initial - cap_used / q_actual).to_numpy(dtype=np.float64)
    dt = df["time_sec"].diff().fillna(0).abs().clip(upper=10.0)
    delta_ah = (df["Current"] * dt) / 3600.0
    return (soc_initial + delta_ah.cumsum() / q_actual).to_numpy(dtype=np.float64)


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    columns = list(rows[0].keys())
    body = []
    body.append("| " + " | ".join(columns) + " |")
    body.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        body.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(body)


def build_raw_integrity_rows() -> list[dict]:
    rows = []
    for temp in TEMP_ORDER:
        temp_dir = Path(DATA_RAW) / temp
        files = [path for path in sorted(temp_dir.glob("*.csv")) if is_drive_file(path)]
        ocv_lookup, _q_actual = build_ocv_soc_lookup(temp)

        total_rows = 0
        dropped_nan_rows = 0
        voltage_violations = 0
        current_spikes = 0
        soc_anomalies = 0
        parsed_soc_rows = 0

        for path in files:
            raw = raw_csv_with_quality_columns(path)
            total_rows += int(len(raw))
            critical_missing = raw[["Voltage", "Current", "time_sec"]].isna().any(axis=1)
            dropped_nan_rows += int(critical_missing.sum())
            voltage = raw["Voltage"]
            current = raw["Current"]
            voltage_violations += int(((voltage < V_MIN) | (voltage > V_MAX)).fillna(False).sum())
            current_spikes += int((current.abs() > I_ABS_LIMIT).fillna(False).sum())

            soc_unclipped = unclipped_soc_for_file(temp, path, ocv_lookup)
            parsed_soc_rows += int(len(soc_unclipped))
            soc_anomalies += int(((soc_unclipped < 0.0) | (soc_unclipped > 1.0)).sum())

        rows.append(
            {
                "Temperature": TEMP_LABEL[temp],
                "Drive files": len(files),
                "Raw rows": total_rows,
                "Dropped/NaN rows": dropped_nan_rows,
                "Voltage violations (<2.5 or >4.25 V)": voltage_violations,
                "Current spikes (|I|>20 A)": current_spikes,
                "SOC anomalies before clipping": soc_anomalies,
                "SOC rows audited": parsed_soc_rows,
            }
        )
    return rows


def build_scenario_composition_rows() -> list[dict]:
    rows = []
    for scenario in ["A", "B"]:
        metadata_path = BASE_DIR / "data" / "processed" / f"v4_scenario_{scenario}" / "metadata_v4.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        counts = metadata["split_window_counts"]
        for temp in TEMP_ORDER:
            rows.append(
                {
                    "Scenario": scenario,
                    "Temperature": TEMP_LABEL[temp],
                    "Train windows": int(counts["train"].get(temp, 0)),
                    "Validation windows": int(counts["val"].get(temp, 0)),
                    "Test windows": int(counts["test"].get(temp, 0)),
                }
            )
    return rows


def main() -> None:
    out_dir = BASE_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = build_raw_integrity_rows()
    comp_rows = build_scenario_composition_rows()

    markdown = "\n".join(
        [
            "**Table 1. Raw Data Integrity Audit**",
            "",
            markdown_table(raw_rows),
            "",
            "**Table 2. Scenario Composition After v4 Split-Before-Windowing**",
            "",
            markdown_table(comp_rows),
            "",
        ]
    )
    (out_dir / "data_audit_tables.md").write_text(markdown, encoding="utf-8")
    (out_dir / "data_audit_tables.json").write_text(
        json.dumps({"raw_integrity": raw_rows, "scenario_composition": comp_rows}, indent=2),
        encoding="utf-8",
    )
    print(markdown)


if __name__ == "__main__":
    main()
