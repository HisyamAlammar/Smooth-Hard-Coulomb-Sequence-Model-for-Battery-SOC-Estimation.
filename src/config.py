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

# ── Evaluation-layer single source of truth (Phase 1 audit fix) ─────
# The SAME threshold gates the model's routing and the PVR audit; any
# divergence between the two silently breaks the structural-PVR argument,
# so evaluation code must import these instead of hardcoding values.
CURRENT_THRESHOLD_A = CURRENT_THRESHOLD          # Amperes, alias for eval code
# Dead-band epsilons (SOC fraction per 1 s step) for epsilon-PVR curves.
# 0.0005 = 0.05 %SOC/step; envelope scale at 1 A is ~1.4e-4 SOC/step.
PVR_EPSILONS = [0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01]

# ── Label generation mode (Phase 5 audit) ───────────────────────────
# "legacy"          : soc_initial = OCV_lookup(raw V of first segment sample),
#                     even when the segment starts under load (48/103 segments;
#                     mean 2.4 %SOC bias, max 33.5 %, ohmic-only lower bound).
# "ohmic_corrected" : soc_initial = OCV_lookup(V0 - I0 * R_int(T)); removes the
#                     Ohmic share of the loaded-start bias (diffusion
#                     overpotential remains uncorrected).
# All published v4 tensors were generated with "legacy"; switching modes
# requires regenerating data/processed and retraining every model.
LABEL_MODE = "legacy"

# ── Dataset versioning (v5 campaign) ─────────────────────────────────
# v4_legacy tensors stay untouched in data/processed/v4_scenario_*.
# v5 variants live in data/processed/<variant>_scenario_* and differ ONLY in
# label_mode / decimation_mode; window, stride, features, and scenario split
# logic are identical to v4.
DATASET_VERSION = "v4_legacy"          # default for legacy code paths
# DECIMATION_MODE options:
#   "first_sample"                : legacy — keep first raw sample per second
#   "mean_per_second"             : V/I/T = intra-second mean (anti-aliased),
#                                   Capacity = last sample (integral state)
#   "integrated_current_per_second": I = intra-second mean (charge-preserving),
#                                   V/T = first sample, Capacity = last
DECIMATION_MODE = "first_sample"
DATASET_VARIANTS = {
    "v4_legacy": {"label_mode": "legacy",          "decimation_mode": "first_sample"},
    "v5a":       {"label_mode": "ohmic_corrected", "decimation_mode": "first_sample"},
    "v5b":       {"label_mode": "legacy",          "decimation_mode": "mean_per_second"},
    "v5c":       {"label_mode": "ohmic_corrected", "decimation_mode": "mean_per_second"},
}

# Physics-Informed Scaling bounds for v3 (V_proxy replaces raw V_t)
# V_proxy has a narrower, more stable range since Ohmic drop is removed
PHYS_MIN_V3 = [2.5,  -20.0, -20.0, -2.0, -20.0]  # [V_proxy, I, T, dV_proxy/dt, dI/dt]
PHYS_MAX_V3 = [4.25,  20.0,  50.0,  2.0,  20.0]
