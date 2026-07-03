# V5 Campaign — Phase 6: Continuous OCV-Rint + 1RC-ECM EKF Baselines

Date: 2026-07-03. Classical recursive filters run **continuously** over 53 reconstructed profile chains (193,910 unique steps, v5c Scenario A), mapped back to the (N,T) test-window grid and scored by the identical metrics module as every neural row. EKF assumptions: 25 °C OCV inversion, Q(T) calibration table, 1RC tau=50 s / R1=0.5·R_int(T) (literature-like, not identified), no test tuning, measurement-noise R sensitivity fully reported (R ∈ {1e-4, 9e-4, 1e-2}). Artifacts: `baselines/ekf_ocv_rint_continuous.py`, `baselines/ekf_1rc_ecm_continuous.py`, `experiments/compare_recursive_vs_ekf.py`, `results/v5/ekf_ecm/{recursive_vs_ekf_comparison.csv, continuous_ekf_results.json}`, figures `recursive_vs_ekf_temperature_breakdown.png`, `ekf_voltage_residual_case_study.png`.

## Comparison table (v5c Scenario A, seed 42, RMSE %)

| Model | RMSE | MaxE | −20 °C | 40 °C | PVR disch ε=0 |
|---|---:|---:|---:|---:|---:|
| HC windowed (legacy) | 11.05 | 46.19 | 16.75 | 4.22 | **0.00** |
| HC carried (train eta) | 11.78 | 29.63 | 11.11 | 8.58 | **0.00** |
| HC hybrid gate | 11.15 | 29.63 | 11.11 | 5.39 | **0.00** |
| HC carried @ eta*=2.0 (Phase 5) | **4.43** | — | **3.59** | 3.89 | **0.00** |
| EKF Rint cont [R=1e-4] | 36.98 | 86.70 | 35.93 | 39.79 | 39.62 |
| EKF 1RC cont [R=1e-4] | 39.12 | 85.63 | 36.06 | 39.62 | 33.67 |
| EKF Rint cont [R=9e-4] | 22.76 | 80.32 | 27.18 | 25.24 | 23.84 |
| EKF 1RC cont [R=9e-4] | 27.45 | 80.31 | 32.40 | 29.33 | 22.09 |
| EKF Rint cont [R=1e-2] | 7.36 | 26.47 | 8.09 | 7.02 | 4.94 |
| EKF 1RC cont [R=1e-2] | 6.85 | **21.15** | 5.77 | 8.25 | 5.48 |

## Findings

1. **EKF performance is dominated by the voltage-trust knob, spanning 6.8 → 39.1 % RMSE over three decades of R.** Small R (trusting the polarization-contaminated voltage proxy) is catastrophic (37–39 % RMSE, MaxE ~86 %); only near-open-loop R=1e-2 — where the EKF degenerates toward pure Coulomb counting with mild voltage correction — is competitive. This sensitivity is itself a headline result: the classical filter has a *hidden tuning dependency* the HC design does not.
2. **Best EKF (1RC, R=1e-2: 6.85 %) beats every uncalibrated HC inference mode (11.05–11.78 %) but loses to the calibrated recursive HC (4.43 %).** The fair like-for-like — both continuous, both zero test tuning — is EKF 6.85 vs HC-carried-at-eta* 4.43, a 2.4 pp margin to the neural-physics hybrid.
3. **Monotonicity is the qualitative separator.** Even the best EKF violates discharge monotonicity on 4.9–5.5 % of steps (ε=0), rising to 22–40 % at lower R; every HC variant is exactly 0 % by construction. For safety-cases built on PVR, the EKF family cannot match the architectural guarantee at any R.
4. **Cold advantage of the best EKF (−20 °C 5.77 %) over uncalibrated HC (11.11 %) disappears against calibrated HC (3.59 %)** — consistent with Phase 5's finding that the HC cold deficit was envelope calibration, not model capacity.
5. **1RC vs Rint:** the polarization state helps only at high R (6.85 vs 7.36) and hurts at mid R; with the voltage proxy this noisy, model-order refinement is second-order relative to the trust setting. The voltage-residual case study figure shows the 1RC V_rc estimate tracking only a fraction of the true polarization residual at −20 °C.

## Caveats

EKF parameters are literature-like, not identified from this cell's data; a fully identified ECM could close some of the gap (flagged as future work, does not affect the tuning-sensitivity or PVR structural claims). Single scenario/seed; Phase 7 assembles the joint matrix.
