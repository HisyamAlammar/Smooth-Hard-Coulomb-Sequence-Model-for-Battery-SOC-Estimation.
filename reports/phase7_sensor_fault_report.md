# Phase 7 — Sensor Fault Stress Test Report

Date: 2026-07-03. Artifacts: `experiments/sensor_fault_stress_test.py`, `results/robustness/sensor_fault_results.json`, `results/robustness/sensor_fault_table.csv`. Scenario A test split, production checkpoints. Faults propagated consistently through every sensor-derived input (I, V_proxy = V − I·R_int(T), dV_proxy/dt, dI/dt); noise seed 42.

## Headline numbers (Hard-Coulomb LSTM)

| Fault | RMSE % | PVR vs **measured** I (disch, ε=0) | PVR vs **true** I (disch, ε=0) | Rest drift (%SOC/h at true rest) |
|---|---:|---:|---:|---:|
| clean | 12.71 | 0.00 | 0.00 | 0.00 |
| offset +0.05 A | 12.94 | 0.00 | 0.00 | +0.14 |
| offset +0.10 A | 13.18 | 0.00 | 0.00 | +1.61 |
| offset **+0.50 A** | 15.23 | **0.00** | **25.74** | +7.94 |
| offset **+1.00 A** | 18.01 | **0.00** | **36.59** | +15.55 |
| offset −0.50 A | **10.80** | 0.00 | 0.00 | −8.86 |
| offset −1.00 A | **9.94** | 0.00 | 0.00 | −18.69 |
| Gaussian σ=0.1 A | 12.72 | 0.00 | 1.27 | −0.01 |
| Gaussian σ=0.5 A | 13.16 | 0.00 | 9.89 | −0.01 |
| stuck at 0 A | 22.32 | 0.00 | 0.00 | 0.00 |

Vanilla contrast: clean rest drift −617 %SOC/h (wander); +0.5 A offset RMSE 16.45 %, PVR(true) 48.2 %.

## Findings

1. **The audited guarantee is measured-current consistency, not physical correctness — now quantified.** With a +0.5 A offset (realistic for automotive hall sensors at percent-of-full-scale accuracy), the audited PVR stays exactly 0.00 % while the model *increases* SOC on **25.7 %** of true-discharge steps (36.6 % at +1 A). The constraint converts a detectable anomaly into a physically plausible wrong trajectory. Any safety text must state PVR's frame of reference explicitly.
2. **Forced rest drift matches the structural prediction.** Offsets beyond the ±0.05 A dead-band force routing into the offset's direction; the learned magnitude head suppresses but cannot zero it (sigmoid > 0): measured drift +7.9 / +15.5 %SOC/h at +0.5 / +1.0 A. Below the dead-band (±0.05 A) the model is immune at rest by construction — the dead-band is the real (narrow) robustness margin.
3. **Negative offsets *improve* RMSE (10.80 % at −0.5 A, 9.94 % at −1.0 A vs 12.71 % clean).** Not robustness: the extra apparent discharge current widens the envelope and compensates the delta path's rate undershoot (ratio 0.72, Phase 1) — one error canceling another. Evidence of delta-path miscalibration, and a warning that offset sensitivity is asymmetric.
4. **A dead current sensor is invisible to PVR.** Stuck-at-zero forces every step into rest routing: output freezes at the per-window anchor, RMSE degrades to 22.3 %, and *both* PVR framings read 0.00 % (a frozen trajectory violates nothing). PVR must never be the sole safety monitor; a rate-tracking metric (delta ratio, Phase 1) catches this (ratio → 0).
5. Gaussian noise is comparatively benign for accuracy (σ=0.5 A: +0.45 pp RMSE) because symmetric sign flips near zero mostly cancel, but true-frame PVR rises to 9.9 % — plausibility is lost step-wise even when the average survives.

## Boundary of claims (mandated statement)

No functional-safety conclusion may be drawn from this test. It demonstrates: (a) the structural PVR property holds w.r.t. its own input under all tested faults (as proven — it is a forward-pass identity); (b) that property is silent about physical correctness under sensor faults. A production system would need sensor diagnostics, plausibility monitors in the *true-physics* frame (e.g., voltage-consistency checks), and rate-tracking alarms.
