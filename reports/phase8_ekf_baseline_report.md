# Phase 8 — EKF (OCV–Rint) Baseline Report

Date: 2026-07-03. Artifacts: `baselines/ekf_ocv_rint.py`, `results/baselines/ekf_results.json`, `results/baselines/ekf_comparison_table.csv`.

## Setup and assumptions (no test-set tuning)

Scalar-state EKF per window, same tensors/metrics as every other model. Measurement model `V_proxy = OCV(SOC) + v` (Ohmic drop already removed by preprocessing; un-modeled diffusion overpotential must be absorbed by R). OCV curve: 25 °C HPPC only (train-temperature knowledge, Scenario-A-consistent), numerically inverted. Q(T) calibration table (same as null model). Init = OCV⁻¹(V_proxy(0)), P₀ = 0.2². Q_proc = (5e-5)². R sensitivity set {1e-4, 9e-4, 1e-2} V² — all three reported, none selected on test.

## Results (full-sequence)

| Scenario | R (V²) | RMSE % | MaxE % | −20 °C RMSE % | PVR disch ε=0 % |
|---|---:|---:|---:|---:|---:|
| A | 1e-4 | 13.00 | 99.05 | 18.40 | 30.29 |
| A | 9e-4 | 13.03 | 93.66 | 18.43 | 24.81 |
| A | 1e-2 | 13.02 | 86.98 | 18.36 | 20.27 |
| B | 1e-4 | 7.05 | 94.70 | 12.89 | 26.08 |
| B | 9e-4 | 7.11 | 94.70 | 12.99 | 18.92 |
| B | 1e-2 | 7.14 | 94.70 | 13.03 | 15.03 |

Comparators: HC-LSTM 12.71 % (A) / 8.57 % (B); null 13.24 / 7.56; **recursive HC 9.23 % (A), −20 °C 6.68 %**.

## Answer to the acceptance question

The proposed method has now been compared against a recursive estimator designed for anchor correction. Under this per-window protocol and these documented assumptions:

1. **The EKF does not rescue the cold-start anchor.** At −20 °C its voltage feedback is poisoned by the same polarization that poisons the OCV anchor: correcting toward a depressed voltage *reinforces* the low-SOC misread (−20 °C RMSE 18.4 % ≈ HC's 17.9 %, MaxE up to 99 %). The learned anchor is at least competitive with classical voltage feedback at cold — genuine, previously missing evidence *for* the paper's observability narrative (properly restated as *voltage-path* observability).
2. **In-distribution the EKF beats both Hard-Coulomb networks** (7.05–7.14 % vs 8.57/8.58 % in Scenario B) with zero training, while losing to nothing but vanilla/null. Any accuracy claim must reckon with this.
3. **The trade is explicit now:** EKF buys mid-window correction at the price of sign-consistency (PVR 15–30 % — it raises SOC during discharge whenever voltage disagrees; that *is* its mechanism). Hard-Coulomb buys PVR ≡ 0 at the price of no mid-window correction. These are two points on a correction-vs-consistency frontier; the manuscript should present them as such rather than as strict superiority.

## Limitations of this baseline (blockers for stronger claims)

- Minimal ECM (no RC polarization states, single OCV curve, no hysteresis). A 1RC-ECM EKF with temperature-dependent parameters would be stronger and might close part of the cold gap; left as future work (identification data exists in HPPC files).
- Per-window protocol handicaps the EKF exactly as it handicaps the networks; a continuously running EKF (analogous to Phase 6 recursive inference) is the fairer deployment comparison and was not run here — flagged as the follow-up experiment.
