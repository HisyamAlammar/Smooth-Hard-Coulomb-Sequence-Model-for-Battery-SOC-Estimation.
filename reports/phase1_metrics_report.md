# Phase 1 — Metrics Layer Fix Report

Date: 2026-07-03. Scope: evaluation code only; no model, training, or data changes.

## What was measured / changed

1. **Threshold centralization.** `src/config.py` now defines `CURRENT_THRESHOLD_A` (alias of the model's `CURRENT_THRESHOLD`) and `PVR_EPSILONS = [0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01]`. Both sprint48 evaluation scripts import these; the duplicated hardcoded `-0.05` constants were removed (`sprint48_safety_ablation.py` now derives `DISCHARGE_THRESHOLD_A = -CURRENT_THRESHOLD_A`). Audit finding 12 resolved. Note: the *same* constant intentionally gates the model routing and the audit — that circularity is now documented in code rather than hidden.
2. **Shared metrics module** `analysis/soc_metrics.py`:
   - `pvr_metrics`: region-resolved (discharge / charge / rest / total) epsilon-dead-band PVR + violation-magnitude stats. Fixes findings 1–2 at the *measurement* level (structural circularity itself is a model property and is now visible, not hidden).
   - `delta_magnitude_metrics`: per-region delta MAE/RMSE, mean |Δpred| vs |Δtrue|, and their ratio (1.0 = correct rate, 0 = frozen). Catches the frozen-output loophole.
   - `regression_metrics`, `per_temperature_metrics`, `evaluate_soc_predictions` (one-call bundle), `legacy_pvr` (bit-identical to the pre-audit definition, kept for reproducibility).
3. **Refactor**: `sprint48_evaluate_all.py` and `sprint48_safety_ablation.py` now call the shared module; legacy JSON field names preserved, `pvr_extended` and `delta_magnitude` added alongside.
4. **Sanity tests** `analysis/test_soc_metrics.py` (plain asserts, all pass): frozen output → PVR 0 but ratio 0; sign-flip → discharge and charge violations fire; ε filters wiggles; rest drift detected; empty regions logged not crashed; masks partition all steps; `legacy_pvr` reproduces old definition on random data.
5. **Shared inference/provenance utils** `analysis/predict_utils.py` (used by all later phases; every result JSON carries scenario, split, checkpoint, seed, timestamp, config).

## Verification

- `analysis/test_soc_metrics.py`: 7/7 PASS.
- Rerun of `src/sprint48_evaluate_all.py` reproduces the published legacy numbers exactly (Scen A vanilla RMSE 13.3712 % / PVR 49.9694 %; Scen A HC 12.7107 % / 0.0 %; Scen B vanilla 7.2806 % / 41.0552 %; Scen B HC 8.5667 % / 0.0 %). No silent protocol change.

## Key new numbers (results/metrics/phase1_summary.csv)

| Scenario | Model | RMSE % | PVR disch ε=0 | ε=0.005 | Δ-ratio (discharge) |
|---|---|---:|---:|---:|---:|
| A | vanilla | 13.37 | 49.97 | 8.05 | **20.09** |
| A | hard_coulomb | 12.71 | 0.00 | 0.00 | **0.72** |
| B | vanilla | 7.28 | 41.06 | 2.71 | **16.30** |
| B | hard_coulomb | 8.57 | 0.00 | 0.00 | **0.38** |

Interpretation (honest, both directions):
- Vanilla's headline "≈50 % PVR" shrinks to 8.1 % (A) / 2.7 % (B) at a 0.5 %SOC/step dead-band — but violations are *not* pure float noise (median 0.11 %SOC/step; predicted delta magnitude 16–20× true rate). Vanilla oscillates pathologically; see Phase 3.
- HC passes every ε-PVR trivially (structural), but its delta-rate tracking is poor: it moves SOC at 72 % (A) and **38 % (B)** of the true rate during discharge. Sign-consistency ≠ rate-correctness; the new metrics make this visible in every future table.
- Rest-region PVR at ε=0 is near-100 % for vanilla (any nonzero drift counts) and exactly 0 for HC (structural zero at rest); interpret rest PVR at ε>0.

## Deliverables

- `analysis/soc_metrics.py`, `analysis/test_soc_metrics.py`, `analysis/predict_utils.py`, `analysis/run_phase1_metrics.py`
- `results/metrics/pvr_deadband_results.json`, `results/metrics/delta_magnitude_results.json`, `results/metrics/phase1_summary.csv`
- Refactored `src/sprint48_evaluate_all.py`, `src/sprint48_safety_ablation.py`; extended `src/config.py`.

## Acceptance criteria

- [x] No hardcoded PVR thresholds outside config.
- [x] PVR reported by discharge/charge/rest and epsilon.
- [x] Delta magnitude reported alongside PVR.
- [x] Existing model results still evaluate; legacy numbers reproduced exactly.
