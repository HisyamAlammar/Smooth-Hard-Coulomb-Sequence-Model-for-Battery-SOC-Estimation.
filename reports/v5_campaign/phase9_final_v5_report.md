# V5 Campaign — Phase 9: Final Research Report

Date: 2026-07-03. Consolidates Phases 0–8. All numbers v5c (ohmic-corrected labels, mean-per-second decimation) unless marked v4. Full provenance in `results/v5/final_ablation_matrix.json` (62 rows); claims adjudicated in `claims_register_v2.{md,json}`.

## 1. What the v5 campaign was for

The v4 audit left three open threats to the manuscript: (T1) label contamination — 48/103 segments anchored under load (ohmic bias up to 33.5 %SOC) and decimation routing conflicts; (T2) the windowed protocol materially changing conclusions vs recursive deployment; (T3) missing continuous classical baselines (the audit EKF was per-window). V5 rebuilt the dataset (four variants; v5c selected), retrained everything multi-seed, mapped the inference-policy space, calibrated the delta path, and ran continuous EKFs on identical chains.

## 2. Headline results (v5c)

### 2.1 Windowed accuracy, 5 seeds (RMSE % mean ± std)

| Model | Scen A (temp-OOD) | Scen B (in-dist) | A MaxE | Verdict |
|---|---:|---:|---:|---|
| null OCV+Coulomb (0 params) | 13.03 | 7.53 | 60.9 | reference |
| Vanilla LSTM | 11.35 ± 0.18 | 6.20 ± 0.27 | 47.5 ± 1.5 | beats null OOD only |
| HC-LSTM (original anchor) | 10.87 ± 0.19 | 10.63 ± 0.60 | 46.5 ± 0.8 | fails in-dist (systematic, 5/5 seeds) |
| HC-TCN | 11.12 ± 1.71 | 7.91 ± 1.16 | 45.4 ± 4.0 | high variance |
| HC anchor_pooled | 10.75 ± 0.98 | 4.93 ± 0.84 | 40.5 ± 1.4 | runner-up |
| **HC anchor_last** | **9.99 ± 1.09** | **4.74 ± 0.31** | **36.5 ± 3.8** | **selected** |

PVR = 0.00 % (ε=0) for every HC row, every seed — by construction, footnoted not tabulated.

### 2.2 Deployment (continuous chains, Scenario A, seed 42)

| System | RMSE | −20 °C | PVR | Tuning exposure |
|---|---:|---:|---:|---|
| HC windowed (legacy protocol) | 11.05 | 16.75 | 0 | none |
| **HC carried + eta*=2.0 (Phase 5)** | **4.43** | **3.59** | **0** | eta* from rate-fidelity ≈ 1, no test labels needed |
| Best EKF (1RC, R=1e-2) | 6.85 | 5.77 | 5.5 % | RMSE spans 6.8–39.1 over R decades |
| load_gated policy (uncalibrated) | 8.41 | 9.95 | 0 | physics-fixed gates |

## 3. The scientific narrative (what the paper now says)

1. **Constraint-through-training works; post-hoc clamping doesn't.** Clamp collapse (25–30 % RMSE) reproduced on corrected labels. [Claim 5]
2. **The anchor is the bottleneck, proven three ways:** oracle reference (0.03 % at −20 °C), architectural intervention (anchor_last redesign → best model), inference intervention (carrying anchors → −5.6 pp cold, calibrated → 3.59 %). [Claims 6, 14]
3. **The delta path was never wrong — its envelope was mis-scaled.** Inference-time eta* restores exact rate fidelity (ratio 1.002) and makes recursive HC the best measured system (4.43 %). Retraining cannot substitute: the head compensates the envelope, so calibration is inherently post-training. This validates the two-stage (learn-then-calibrate) design. [Claim 15]
4. **Label correction changed magnitudes, not conclusions — and honesty requires saying so.** Every model improved on corrected labels; part of v4's MaxE story (87 %) was artifact (→61 %); the HC advantage survives and strengthens. [Claims 8, 17]
5. **Against classical recursion: better and structurally safer.** Best-case EKF is 1.5× worse, hides a 6× RMSE tuning knob, and violates monotonicity on 5–40 % of steps. [Claim 16]

## 4. Honest limitations (manuscript-binding)

- eta* evidence is single-checkpoint/scenario (multi-seed + Scen B re-run scheduled; windowed numbers unaffected).
- Ohmic-only label correction is a lower bound; polarization unmodeled.
- Winner not strictly seed-stable (anchor_last 4/5 in A); family-level claim only.
- Edge feasibility parameter-level only; no hardware WCET/RAM numbers.
- Sensor faults: characterized failure envelope, not robustness.
- EKF params literature-like, not identified from this cell.

## 5. Figures inventory (results/v5/figures/)

`v4_vs_v5_rmse_by_model.png`, `v5_temperature_breakdown.png`, `multiseed_{rmse,maxe}_boxplot.png`, `recursive_policy_temperature_breakdown.png`, `cold_sequence_recursive_case_study.png`, `hot_sequence_recursive_failure_case.png`, `eta_vs_rmse_by_temperature.png`, `eta_vs_delta_ratio.png`, `eta_vs_recursive_drift.png`, `recursive_vs_ekf_temperature_breakdown.png`, `ekf_voltage_residual_case_study.png`, plus P1 dataset figures.

## 6. Manuscript rewrite brief

**Title/abstract:** lead with anchored-HC + inference-time physics calibration; quote A 9.99 ± 1.09 / B 4.74 ± 0.31 windowed and 4.43 % recursive; PVR "by construction".

**§ Method:** (a) anchor head reads last hidden state, remapped to feasibility interval; (b) envelope `|I|·eta·gamma` with eta a *calibratable* physical safety factor — new subsection "Post-training rate calibration" (Phase 5 math + rate-fidelity criterion); (c) inference policies as a first-class protocol axis.

**§ Experiments:** dataset = v5c with explicit label-correction subsection (v4 deltas disclosed, Claim 17); all learned rows mean ± std over 5 seeds; protocols reported in pairs (windowed + recursive); EKF sensitivity table in main text, not appendix — it carries the "hidden knob" argument.

**§ Results order:** (1) windowed table, (2) anchor-dominance evidence chain, (3) calibration sweep figure trio, (4) deployment table vs EKF, (5) MaxE/PVR safety framing.

**§ Discussion:** correction-vs-consistency frontier replaced by calibrated recursion (the frontier collapsed); label-uncertainty limitation; the five reviewer questions answered with v5 numbers (claims register §Reviewer questions).

**Delete/never write:** "suitable for edge MCUs", "functional safety" (bare), "robust to sensor faults", any single-seed headline number, PVR as empirical achievement.

## 7. Reproduction

```
python experiments/run_multiseed_v5.py --variant v5c --scenarios A B --models all --seeds 1 2 3 4 5
python analysis/multiseed_summary.py
python experiments/compare_recursive_policies_v5.py --variant v5c --scenario A --seed 42
python experiments/delta_path_rate_calibration.py --variant v5c --scenario A --seed 42
python experiments/compare_recursive_vs_ekf.py --variant v5c --scenario A --seed 42
python analysis/build_final_ablation_v5.py
```
Deterministic given seeds; ~35 min GPU total (RTX-class), checkpoints cached under `results/v5/`.
