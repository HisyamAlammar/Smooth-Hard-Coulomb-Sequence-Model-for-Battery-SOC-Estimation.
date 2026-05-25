"""
config.py — Project-wide hyperparameters and directory paths
=============================================================
All values locked per Project Brief Section 2.3.
Do NOT modify without explicit researcher approval.

Created : 2026-04-07
Updated : 2026-04-08 — Sprint 2/3: added model hyperparams
Updated : 2026-05-16 — v3 Hybrid Physics-ML: R_int + Hard Constraint config
"""

import os

# ── Directory Paths ──────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW    = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC   = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_FIG  = os.path.join(BASE_DIR, "outputs", "figures")
OUTPUT_MOD  = os.path.join(BASE_DIR, "outputs", "models")
OUTPUT_SCAL = os.path.join(BASE_DIR, "outputs", "scalers")
LOG_DIR     = os.path.join(BASE_DIR, "logs")

# ── Battery Physical Constants ───────────────────────────────────────
Q_NOMINAL   = 3.0          # Ah  — LG HG2 nominal capacity
V_MAX       = 4.2          # V   — upper cut-off voltage
V_MIN       = 2.5          # V   — lower cut-off voltage
SOH_EOL     = 0.80         # End-of-Life threshold

# ── Data Pipeline ────────────────────────────────────────────────────
TARGET_TEMPS  = ["25degC", "40degC"]   # base temperatures
SEQUENCE_LEN  = 100         # timesteps per window
STRIDE        = 10          # sliding-window stride

# ── Model Architecture (Project Brief §4) ────────────────────────────
NUM_INPUTS      = 5                  # [V, I, T, dV/dt, dI/dt]
NUM_FILTERS     = 64                 # channels per TemporalBlock
KERNEL_SIZE     = 7                  # conv kernel size
DROPOUT         = 0.2                # dropout probability
DILATION_RATES  = [1, 2, 4, 8]       # one per TemporalBlock

# ── Training Hyperparameters (Project Brief §2.3) ────────────────────
BATCH_SIZE    = 1024        # GPU-optimized (RTX 4060 8GB)
LEARNING_RATE = 1e-3
EPOCHS        = 100
LAMBDA_PHYS   = 0.1         # weight of physics penalty term (v2 only)
RANDOM_SEED   = 42

# ── v3: Hybrid Physics-ML Pipeline ───────────────────────────────────
# Internal Resistance (Ohms) extracted from HPPC pulse data.
# Method: Average R_int across all SOC-level pulses per temperature.
# Source: REST→DCH transitions in *_HPPC.csv files.
# Verified: 2026-05-16 — 45-60 pulses per temperature.
R_INT_PER_TEMP = {
    '40degC':  0.01651,    # 16.51 mΩ  (55 pulses from 555_HPPC.csv)
    '25degC':  0.01986,    # 19.86 mΩ  (60 pulses from 549_HPPC.csv)
    '10degC':  0.02875,    # 28.75 mΩ  (55 pulses from 575_HPPC.csv)
    '0degC':   0.04008,    # 40.08 mΩ  (50 pulses from 585_HPPC.csv)
    'n10degC': 0.06219,    # 62.19 mΩ  (50 pulses from 593_HPPC.csv)
    'n20degC': 0.10983,    # 109.83 mΩ (45 pulses from 607_HPPC.csv)
}

# Current threshold for 3-way Hard Constraint split (Amperes)
# |I| < THRESHOLD → Rest;  I < -THRESHOLD → Discharge;  I > THRESHOLD → Charge
CURRENT_THRESHOLD = 0.05

# Physics-Informed Scaling bounds for v3 (V_proxy replaces raw V_t)
# V_proxy has a narrower, more stable range since Ohmic drop is removed
PHYS_MIN_V3 = [2.5,  -20.0, -20.0, -2.0, -20.0]  # [V_proxy, I, T, dV_proxy/dt, dI/dt]
PHYS_MAX_V3 = [4.25,  20.0,  50.0,  2.0,  20.0]
