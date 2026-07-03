"""
ekf_1rc_ecm_continuous.py -- Phase 6: continuous 1RC-ECM EKF.

State x = [SOC, V_rc] over contiguous profile chains.
  SOC_t  = SOC_{t-1} + I_t*dt/(3600*Q(T))
  Vrc_t  = Vrc_{t-1}*exp(-dt/tau) + R1*(1-exp(-dt/tau))*I_t
Measurement (Ohmic drop already removed by preprocessing):
  V_proxy = OCV(SOC) + V_rc + v,   H = [dOCV/dSOC, 1]

Because V_rc models the diffusion/polarization overpotential the scalar EKF
had to treat as noise, this filter CAN separate "cold voltage sag" from
"low SOC" -- the fair classical answer to the anchor-observability problem.

Parameter assumptions (NOT identified from HPPC, NOT tuned on test;
documented as literature-like priors):
  tau = 50 s;  R1(T) = 0.5 * R_int(T)  (polarization resistance comparable to
  ohmic share, growing at cold);  Q(T) from the calibration table.
Process noise: SOC (5e-5)^2, Vrc (1e-3 V)^2. P0 = diag(0.2^2, 0.02^2).
R sensitivity set shared with the scalar EKF.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "src"), str(BASE_DIR / "baselines")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import R_INT_PER_TEMP  # noqa: E402
from ekf_ocv_rint import build_soc_to_ocv  # noqa: E402
from preprocessing_v4 import Q_ACTUAL_PER_TEMP  # noqa: E402

TAU_S = 50.0
R1_FRACTION_OF_RINT = 0.5
Q_SOC = (5e-5) ** 2
Q_VRC = (1e-3) ** 2
P0 = np.diag([0.2**2, 0.02**2])


def run_continuous_1rc_ekf(sequences: List[Dict], R: float) -> None:
    """In-place: adds s['soc_est'] (and s['vrc_est']) to each sequence."""
    h, dh, ocv_to_soc = build_soc_to_ocv()
    a = float(np.exp(-1.0 / TAU_S))
    Qp = np.diag([Q_SOC, Q_VRC])
    for s in sequences:
        q_ah = Q_ACTUAL_PER_TEMP.get(s["temp"], 3.0)
        r1 = R1_FRACTION_OF_RINT * R_INT_PER_TEMP.get(s["temp"], 0.03)
        b = r1 * (1.0 - a)
        coeff = 1.0 / (3600.0 * q_ah)
        v, I = s["v_proxy"], s["I"]
        n = len(v)

        x = np.array([float(np.clip(ocv_to_soc(v[0]), 0.0, 1.0)), 0.0])
        P = P0.copy()
        F = np.array([[1.0, 0.0], [0.0, a]])
        soc_out, vrc_out = np.empty(n), np.empty(n)
        soc_out[0], vrc_out[0] = x[0], x[1]
        for t in range(1, n):
            # predict
            x = np.array([x[0] + I[t] * coeff, a * x[1] + b * I[t]])
            P = F @ P @ F.T + Qp
            # update
            s_cl = min(max(x[0], 0.0), 1.0)
            H = np.array([float(dh(s_cl)), 1.0])
            resid = v[t] - (float(h(s_cl)) + x[1])
            S = float(H @ P @ H) + R
            K = (P @ H) / S
            x = x + K * resid
            x[0] = min(max(x[0], 0.0), 1.0)
            P = (np.eye(2) - np.outer(K, H)) @ P
            soc_out[t], vrc_out[t] = x[0], x[1]
        s["soc_est"] = soc_out
        s["vrc_est"] = vrc_out
