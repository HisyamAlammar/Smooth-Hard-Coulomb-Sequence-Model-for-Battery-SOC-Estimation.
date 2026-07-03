# Final Code Audit Fix Report

Date: 2026-07-03. Campaign: Phases 0–10 executed against the 12 red-team audit findings. All experiments ran to completion; no blockers were faked. Original `outputs/`, published checkpoints, and legacy metric definitions preserved.

## What was changed (existing code)

| File | Change | Audit finding |
|---|---|---|
| `src/config.py` | `CURRENT_THRESHOLD_A`, `PVR_EPSILONS`, `LABEL_MODE` added (single source of truth) | 12, 9 |
| `src/sprint48_evaluate_all.py` | Metrics delegated to `analysis/soc_metrics.py`; legacy fields preserved; `pvr_extended` + `delta_magnitude` added | 1, 2, 12 |
| `src/sprint48_safety_ablation.py` | Same refactor; hardcoded −0.05 removed | 12 |
| `src/preprocessing_v4.py` | `label_mode` parameter (`legacy` default = byte-identical; `ohmic_corrected` fixes loaded-start OCV anchor) | 9 |
| `analysis/predict_utils.py` | (new, shared) inference + provenance for all experiments | reproducibility |

## What was added (new modules, all with JSON/CSV/PNG outputs + per-phase reports)

- **Metrics**: `analysis/soc_metrics.py` (region/ε-PVR, delta-rate tracking, frozen-output detection), `analysis/test_soc_metrics.py` (7 sanity checks, all pass).
- **Baselines**: `baselines/null_ocv_coulomb.py` (0 params), `baselines/posthoc_clamp.py`, `baselines/ekf_ocv_rint.py`, `baselines/build_comparison.py`.
- **Diagnostics**: `analysis/diagnose_vanilla_baseline.py`, `analysis/anchor_error_analysis.py`, `analysis/preprocessing_audit.py`, `analysis/edge_feasibility.py`.
- **Experiments**: `experiments/oracle_anchor.py`, `experiments/compare_anchor_variants.py` (incl. recursive stitched inference), `experiments/sensor_fault_stress_test.py`, `experiments/quantized_pvr_check.py`.
- **Aggregates**: `results/final_model_comparison.{csv,json}`, `reports/claims_register.md`, per-phase reports `phase0`–`phase9`.

## Audit findings: resolved vs remaining

**Resolved (measured/fixed):** duplicated thresholds (12); sign-only/one-sided PVR (2 — metrics now region/ε-resolved with rate tracking); missing baselines (10 — null, clamp, EKF all in); vanilla fairness question (11 — no bug; mechanism identified); anchor-trap hypothesis (6, 7 — proven anchor-dominated; ~96 % of RMSE); windowed-protocol question (8 — recursive inference changes conclusions materially); [0,1] conditionality (4 — precondition documented; quantization pipeline requirement identified); label mechanisms quantified (9 — flag added, default legacy).

**Remaining as documented limitations:** measured-vs-true-current gap is *inherent* to the architecture (Phase 7 characterizes it: 25.7 % true-frame PVR at +0.5 A offset; dead-band immunity only ±0.05 A); label regeneration under `ohmic_corrected` + mean-per-second decimation deferred to a versioned v5 pipeline (invalidates all published tensors); 1RC-ECM EKF and continuously-running EKF not built; hardware WCET/RAM validation absent; functional-safety claims remain out of scope by evidence (5).

## Key results table (full detail in `results/final_model_comparison.csv`)

| Model (Scenario A / OOD) | RMSE % | MaxE % | −20 °C RMSE % | PVR disch ε=0 % |
|---|---:|---:|---:|---:|
| **HC-LSTM, recursive inference** | **9.23** | **31.19** | **6.68** | 0.00 |
| HC-TCN (windowed) | 11.46 | 46.73 | — | 0.00 |
| HC-LSTM anchor_last (3-seed mean) | 11.97 ± 1.00 | 41.9 | 16.4 | 0.00 |
| HC-LSTM (published) | 12.71 | 55.11 | 17.86 | 0.00 |
| EKF OCV–Rint (best R) | 13.00 | 86.98 | 18.36 | 20–30 |
| Null OCV+Coulomb (0 params) | 13.24 | 86.97 | — | 0.00 |
| Vanilla LSTM | 13.37 | 51.02 | — | 49.97 |
| Vanilla + post-hoc clamp | 25.56 | 48.16 | — | 0.00 |
| *HC-LSTM oracle anchor (diagnostic)* | *0.54* | *5.05* | *0.50* | *0.00* |

Key plots: `results/figures/vanilla_prediction_examples.png` (wander mechanism), `anchor_error_by_temperature.png` (5.0→15.8 % cold ramp), `vanilla_delta_histogram.png`.

## Scientific interpretation shift (explicit, per instructions)

1. The contribution is **not** average accuracy — the zero-parameter null model matches or beats the published HC-LSTM. The defensible contributions are: (a) structural sign-consistency that *survives training, faults, and benign quantization*; (b) trainability through the constraint (post-hoc clamping collapses); (c) bounded MaxE vs raw-voltage anchoring.
2. The "−20 °C thermodynamic observability limit" narrative is **replaced** by: single-sample voltage-anchor observability under a windowed protocol. Recursive inference on the *unchanged* checkpoint cuts −20 °C MaxE from 55 % to 17 % — the strongest new result of the campaign, at zero training cost — while exposing a real delta-rate undershoot (ratio 0.38–0.72; partially explained by decimation making the envelope unsatisfiable on 7–13 % of steps).
3. PVR must always be published in both frames (measured and true current) with an ε-curve and the delta-ratio column; alone, it rewards frozen outputs and hides dead sensors.

## Recommended next steps before manuscript rewrite

1. Regenerate a versioned v5 dataset (`LABEL_MODE="ohmic_corrected"` + mean-per-second decimation), retrain, and check which conclusions move (top priority — label validity underpins everything).
2. Multi-seed (≥5) runs for HC-LSTM/vanilla/HC-TCN; anchor_last showed seed sensitivity.
3. Continuous (stitched) EKF and a 1RC-ECM EKF for a fair recursive-vs-recursive comparison against recursive HC.
4. Rate-calibration of the delta path (η justification: η must exceed max_T Q_nom/Q_actual(T) = 1.287; consider supervising delta magnitude against Coulomb deltas).
5. Only then: manuscript revision against `reports/claims_register.md`.
