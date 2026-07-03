"""
ekf_ocv_rint_continuous.py -- Phase 6: continuous scalar OCV-Rint EKF.

Unlike the v4 per-window EKF, this filter runs ONCE per contiguous profile
chain (reconstructed from overlapping test windows via timestamp keys) and is
therefore the fair classical counterpart to recursive carried-anchor HC
inference.

Measurement model: V_proxy = OCV(SOC) + v (Ohmic drop already removed).
Assumptions identical to the v4 EKF (25 degC OCV inversion, Q(T) calibration
table, no test-set tuning; R sensitivity set fully reported).

Also exposes `reconstruct_sequences` and `map_back_to_windows` used by the
1RC variant and the comparison script.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "src"), str(BASE_DIR / "analysis"),
          str(BASE_DIR / "baselines"), str(BASE_DIR / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import PHYS_MAX_V3, PHYS_MIN_V3  # noqa: E402
from ekf_ocv_rint import P0, Q_PROC_STD, build_soc_to_ocv  # noqa: E402
from preprocessing_v4 import PROFILE_KEY_SCALE, Q_ACTUAL_PER_TEMP  # noqa: E402

STRIDE = 10


def reconstruct_sequences(data_dir: Path) -> Tuple[List[Dict], np.ndarray]:
    """
    Rebuild contiguous 1 Hz sequences from overlapping test windows.
    Returns (sequences, timestamp_keys). Each sequence dict has arrays
    v_proxy, I, y_true, temp (str), keys; first-occurrence wins on overlaps.
    """
    X = np.load(data_dir / "X_test.npy")
    y = np.load(data_dir / "y_test.npy")
    I = np.load(data_dir / "I_unscaled_test.npy")
    keys = np.load(data_dir / "timestamp_key_test.npy")
    temps = np.load(data_dir / "temp_labels_test.npy", allow_pickle=True)
    v_proxy = X[:, :, 0] * (PHYS_MAX_V3[0] - PHYS_MIN_V3[0]) + PHYS_MIN_V3[0]

    order = np.argsort(keys[:, 0], kind="stable")
    sequences: List[Dict] = []
    cur = None
    for j in order:
        k0 = int(keys[j, 0])
        if cur is not None and k0 == cur["next_expected"] and (
            k0 // PROFILE_KEY_SCALE == cur["profile"]
        ):
            # contiguous window: append only the STRIDE new trailing steps
            cur["v_proxy"].extend(v_proxy[j][-STRIDE:])
            cur["I"].extend(I[j][-STRIDE:])
            cur["y_true"].extend(y[j][-STRIDE:])
            cur["keys"].extend(keys[j][-STRIDE:])
            cur["window_index"].append(j)
            cur["next_expected"] = k0 + STRIDE
        else:
            if cur is not None:
                sequences.append(cur)
            cur = {"profile": k0 // PROFILE_KEY_SCALE, "temp": str(temps[j]),
                   "v_proxy": list(v_proxy[j]), "I": list(I[j]), "y_true": list(y[j]),
                   "keys": list(keys[j]), "window_index": [j], "next_expected": k0 + STRIDE}
    if cur is not None:
        sequences.append(cur)
    for s in sequences:
        for name in ("v_proxy", "I", "y_true"):
            s[name] = np.asarray(s[name], dtype=np.float64)
        s["keys"] = np.asarray(s["keys"], dtype=np.int64)
    return sequences, keys


def map_back_to_windows(sequences: List[Dict], keys: np.ndarray) -> np.ndarray:
    """Scatter per-sequence estimates back onto the (N, T) window grid."""
    est: Dict[int, float] = {}
    for s in sequences:
        for k, v in zip(s["keys"], s["soc_est"]):
            est.setdefault(int(k), float(v))
    out = np.empty(keys.shape, dtype=np.float32)
    flat_keys = keys.reshape(-1)
    out_flat = out.reshape(-1)
    for i, k in enumerate(flat_keys):
        out_flat[i] = est[int(k)]
    return out


def run_continuous_scalar_ekf(sequences: List[Dict], R: float) -> None:
    """In-place: adds s['soc_est'] to each sequence."""
    h, dh, ocv_to_soc = build_soc_to_ocv()
    for s in sequences:
        q_ah = Q_ACTUAL_PER_TEMP.get(s["temp"], 3.0)
        coeff = 1.0 / (3600.0 * q_ah)
        v, I = s["v_proxy"], s["I"]
        n = len(v)
        soc = float(np.clip(ocv_to_soc(v[0]), 0.0, 1.0))
        P = P0
        out = np.empty(n)
        out[0] = soc
        for t in range(1, n):
            soc = soc + I[t] * coeff
            P = P + Q_PROC_STD**2
            s_cl = min(max(soc, 0.0), 1.0)
            H = float(dh(s_cl))
            resid = v[t] - float(h(s_cl))
            S = H * P * H + R
            K = P * H / S
            soc = min(max(soc + K * resid, 0.0), 1.0)
            P = (1.0 - K * H) * P
            out[t] = soc
        s["soc_est"] = out
