"""
preprocessing_audit.py -- Phase 5: label and decimation audit (read-only).

Audits three suspected label-contamination mechanisms in preprocessing_v4.py
WITHOUT changing any labels:

  A. 1 Hz decimation keeps the FIRST sample of each second while the ground
     truth (cycler Capacity) integrates at full rate. Measures how far the
     kept sample deviates from the intra-second mean current, how often its
     sign disagrees, and how often the true per-second |dSOC| exceeds the
     model's envelope eta*gamma*|I_kept|.

  B. Per-segment soc_initial = OCV_lookup(V of first sample). Segments split
     at time gaps can start under load, where terminal voltage is depressed
     by polarization -> biased initial SOC for the WHOLE segment's labels.

  C. cap_used = (cap - cap0).abs(): any net-charge excursion above the segment
     start makes the label DECREASE during charging. Measures max positive
     net-capacity excursion per segment.

Outputs:
  results/diagnostics/preprocessing_audit.json
  results/diagnostics/soc_initial_stats_by_temperature.csv
  results/diagnostics/segment_start_condition_report.csv
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "src"), str(BASE_DIR / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import CURRENT_THRESHOLD_A, Q_NOMINAL  # noqa: E402
from predict_utils import provenance  # noqa: E402
from preprocessing_v4 import (  # noqa: E402
    DATA_RAW,
    Q_ACTUAL_PER_TEMP,
    WINDOW,
    build_ocv_soc_lookup,
    read_csv,
    to_strict_1hz_segments,
)

OUT_DIR = BASE_DIR / "results" / "diagnostics"
TEMPS = ["0degC", "10degC", "25degC", "40degC", "n10degC", "n20degC"]
DRIVE_KEYS = ("udds", "la92", "hwfet", "us06", "mixed")
ETA = 1.5
GAMMA = 1.0 / (Q_NOMINAL * 3600.0)


def drive_csvs(temp: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(DATA_RAW, temp, "*.csv")))
    return [f for f in files if any(k in os.path.basename(f).lower() for k in DRIVE_KEYS)]


def audit_decimation(df) -> dict:
    """Compare kept (first) sample per second vs intra-second mean current."""
    sec = np.floor(df["time_sec"].to_numpy(np.float64) + 1e-9).astype(np.int64)
    I = df["Current"].to_numpy(np.float64)
    order = np.argsort(sec, kind="stable")
    sec, I = sec[order], I[order]
    uniq, first_idx, counts = np.unique(sec, return_index=True, return_counts=True)
    multi = counts > 1
    if not multi.any():
        return {"n_seconds": int(len(uniq)), "n_multi_sample_seconds": 0}
    sums = np.add.reduceat(I, first_idx)
    I_mean = sums / counts
    I_first = I[first_idx]
    dev = np.abs(I_first - I_mean)[multi]
    im, ifst = I_mean[multi], I_first[multi]
    # routing-sign disagreement (both outside the rest dead-band, opposite sign)
    sign_conflict = ((ifst > CURRENT_THRESHOLD_A) & (im < -CURRENT_THRESHOLD_A)) | (
        (ifst < -CURRENT_THRESHOLD_A) & (im > CURRENT_THRESHOLD_A)
    )
    # true per-second |dSOC| (from mean current) exceeding the model envelope
    envelope_exceeded = np.abs(im) * GAMMA > ETA * np.abs(ifst) * GAMMA
    # rest-labelled seconds that actually carried charge
    rest_but_flowing = (np.abs(ifst) <= CURRENT_THRESHOLD_A) & (np.abs(im) > CURRENT_THRESHOLD_A)
    return {
        "n_seconds": int(len(uniq)),
        "n_multi_sample_seconds": int(multi.sum()),
        "abs_dev_A": {"mean": float(dev.mean()), "p95": float(np.percentile(dev, 95)),
                      "max": float(dev.max())},
        "sign_conflict_pct": float(100.0 * sign_conflict.mean()),
        "envelope_exceeded_pct": float(100.0 * envelope_exceeded.mean()),
        "rest_but_flowing_pct": float(100.0 * rest_but_flowing.mean()),
    }


def main() -> None:
    audit: dict = {"provenance": provenance("scenario_A", None, {
        "experiment": "preprocessing_audit", "eta": ETA,
        "note": "read-only audit of preprocessing_v4 label pipeline; raw 10 Hz CSVs",
    })}
    soc_rows, seg_rows = [], []

    for temp in TEMPS:
        files = drive_csvs(temp)
        if not files:
            audit[temp] = {"blocker": "no drive-cycle CSVs found"}
            continue
        ocv_lookup, q_actual = build_ocv_soc_lookup(temp)
        dec_stats, soc_inits = [], []
        n_seg = n_load_start = n_postgap = 0
        cap_pos_excursions = []

        for fpath in files:
            df = read_csv(fpath)
            dec_stats.append(audit_decimation(df))
            segments, _, _ = to_strict_1hz_segments(
                df, source_id=f"{temp}/{os.path.basename(fpath)}",
                profile_code_start=1, min_len=WINDOW,
            )
            for k, seg in enumerate(segments):
                v0 = float(seg["Voltage"].iloc[0])
                i0 = float(seg["Current"].iloc[0])
                loaded = abs(i0) > CURRENT_THRESHOLD_A
                soc_init = float(np.clip(ocv_lookup(v0), 0.0, 1.0)) if ocv_lookup else 1.0
                # what the anchor would be if the Ohmic-corrected voltage were used
                soc_init_vproxy = (
                    float(np.clip(ocv_lookup(v0 - i0 * 0.03), 0.0, 1.0)) if ocv_lookup else 1.0
                )
                soc_inits.append(soc_init)
                n_seg += 1
                n_load_start += int(loaded)
                n_postgap += int(k > 0)
                # capacity abs() audit
                cap_excursion = 0.0
                if "Capacity" in seg.columns and seg["Capacity"].notna().any():
                    net = seg["Capacity"].fillna(0.0) - float(seg["Capacity"].fillna(0.0).iloc[0])
                    cap_excursion = float(net.max())
                cap_pos_excursions.append(cap_excursion)
                seg_rows.append({
                    "temp": temp, "file": os.path.basename(fpath), "segment_idx": k,
                    "n_rows": len(seg), "post_gap": int(k > 0),
                    "I_start_A": round(i0, 4), "V_start_V": round(v0, 4),
                    "loaded_start": int(loaded),
                    "soc_initial_ocv": round(soc_init, 4),
                    "soc_initial_ocv_ohmic_corrected": round(soc_init_vproxy, 4),
                    "max_pos_cap_excursion_Ah": round(cap_excursion, 5),
                    "max_label_flip_pct_soc": round(100.0 * cap_excursion / q_actual, 3),
                })

        def agg(key, sub=None):
            vals = [d[key][sub] if sub else d[key] for d in dec_stats if key in d]
            return float(np.mean(vals)) if vals else None

        pos_exc = np.array(cap_pos_excursions)
        audit[temp] = {
            "files": len(files),
            "decimation": {
                "mean_abs_dev_A": round(agg("abs_dev_A", "mean") or 0, 4),
                "p95_abs_dev_A": round(agg("abs_dev_A", "p95") or 0, 4),
                "max_abs_dev_A": round(max((d["abs_dev_A"]["max"] for d in dec_stats if "abs_dev_A" in d), default=0), 4),
                "sign_conflict_pct": round(agg("sign_conflict_pct") or 0, 4),
                "envelope_exceeded_pct": round(agg("envelope_exceeded_pct") or 0, 4),
                "rest_but_flowing_pct": round(agg("rest_but_flowing_pct") or 0, 4),
            },
            "segments": {
                "total": n_seg, "post_gap": n_postgap,
                "loaded_start": n_load_start,
                "loaded_start_pct": round(100.0 * n_load_start / max(n_seg, 1), 2),
                "cap_positive_excursion_segments": int((pos_exc > 1e-6).sum()),
                "max_label_flip_pct_soc": round(100.0 * float(pos_exc.max()) / q_actual, 3) if len(pos_exc) else 0.0,
            },
            "q_actual_Ah": round(float(q_actual), 4),
        }
        s = np.array(soc_inits)
        soc_rows.append({
            "temp": temp, "n_segments": len(s),
            "soc_initial_mean": round(float(s.mean()), 4),
            "soc_initial_min": round(float(s.min()), 4),
            "soc_initial_max": round(float(s.max()), 4),
            "soc_initial_std": round(float(s.std()), 4),
            "loaded_start_pct": audit[temp]["segments"]["loaded_start_pct"],
        })
        print(f"{temp}: {audit[temp]['decimation']} | segments {audit[temp]['segments']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "preprocessing_audit.json").write_text(json.dumps(audit, indent=2, default=float))
    for name, rows in (("soc_initial_stats_by_temperature.csv", soc_rows),
                       ("segment_start_condition_report.csv", seg_rows)):
        with (OUT_DIR / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Saved audit to {OUT_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
