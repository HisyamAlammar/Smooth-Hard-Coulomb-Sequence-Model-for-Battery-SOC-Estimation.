# v5 Ablation Notebook Screening Report

Generated: 2026-07-03

## Scope

This screening covers the active notebook and artifact directories before creating the final v5 ablation notebook stack. The goal is to preserve the v4 evidence layer, avoid heavy retraining, and create v5 notebooks that load existing result artifacts only.

## Existing Notebook Inventory

Existing notebook directories:

| Directory | Status | Notes |
|---|---|---|
| `notebooks/` | present | Contains EDA scripts and the existing ablation folder. |
| `notebooks/ablation_studies/` | present | Contains the legacy v4/early-hypothesis ablation stack. |
| `notebooks/ablation_studies_v5_final/` | not present before this task | Safe to create as a versioned final v5 analysis layer. |
| `notebooks/ablation_studies_v4_legacy/` | not present before this task | Not required if legacy notebooks remain in place. |

Existing `.ipynb` ablation notebooks:

| Notebook | Legacy classification | v5 status |
|---|---|---|
| `01_Seq2Point_Windowing_Artifact_and_Pseudo_PVR.ipynb` | v4/early evidence | Preserve as historical baseline narrative. |
| `02_Vanilla_LSTM_TCN_Physics_Blindness.ipynb` | v4/early evidence | Preserve; final v5 should use updated multiseed/ablation matrix ledgers. |
| `03_Soft_PINN_Penalty_Gradient_Collision.ipynb` | v4/early evidence | Preserve as negative soft-constraint evidence. |
| `04_Direction_Only_Hard_Constraint_Cumulative_Drift.ipynb` | v4/early evidence | Needs explicit legacy note: v5 Phase 5 shows drift is largely correctable by eta calibration. |
| `05_Hard_Clamp_vs_Smooth_Coulomb_Gradient_Pathology.ipynb` | v4/early evidence | Preserve as architecture-evolution evidence. |
| `06_Vproxy_HPPC_Rint_Feature_Defense.ipynb` | v4/early evidence | Preserve; v5 dataset correction supersedes some label-artifact context. |
| `07_Zero_Leakage_Split_Before_Windowing_Forensics.ipynb` | v4/early evidence | Preserve as leakage defense. |
| `08_Anchor_Trap_and_Safety_Factor_NonCause.ipynb` | v4/early evidence | Needs explicit legacy note: v5 confirms anchor bottleneck but updates final solution to anchor_last + calibrated carried inference. |
| `09_Contextual_Anchor_OCV_Rest_vs_History.ipynb` | v4/early evidence | Preserve; not the final v5 selected system. |
| `10_Gated_Context_Negative_Result_Sparse_Rest_Validity.ipynb` | v4/early evidence | Needs explicit legacy note: v5 Phase 4 later found load-gated recursive inference positive. |
| `11_HardCoulomb_LSTM_vs_TCN_Backbone_Tradeoff.ipynb` | v4/early evidence | Preserve; final v5 ranking is anchor/inference/calibration dominated. |

Decision: do not move or overwrite the old notebooks. Create `notebooks/ablation_studies/README_LEGACY.md` instead so existing links and relative paths remain stable.

## v5 Result Artifact Inventory

Core artifacts required for the new notebook stack are present:

| Artifact | Status | Used by |
|---|---|---|
| `results/v5/dataset_variant_comparison.csv` | present | Notebook 12 |
| `results/v5/dataset_variant_comparison.json` | present | Notebook 12 |
| `results/v5/headline_models/v4_vs_v5_comparison.csv` | present | Notebook 13 |
| `results/v5/headline_models/v5_headline_model_comparison.csv` | present | Notebooks 13, 15 |
| `results/v5/final_v5_model_comparison.csv` | present | Notebooks 13, 16, 17, 18, 19 |
| `results/v5/final_v5_model_comparison.json` | present | Notebook 19 |
| `results/v5/multiseed/seed_level_results.csv` | present | Notebook 14 |
| `results/v5/multiseed/multiseed_summary.csv` | present | Notebook 14 |
| `results/v5/multiseed/ranking_stability.json` | present | Notebook 14 |
| `results/v5/recursive_inference/recursive_policy_comparison.csv` | present | Notebook 16 |
| `results/v5/recursive_inference/recursive_policy_results.json` | present | Notebook 16 |
| `results/v5/delta_calibration/eta_gamma_sweep.csv` | present | Notebook 17 |
| `results/v5/delta_calibration/eta_gamma_sweep.json` | present | Notebook 17 |
| `results/v5/ekf_ecm/recursive_vs_ekf_comparison.csv` | present | Notebook 18 |
| `results/v5/ekf_ecm/continuous_ekf_results.json` | present | Notebook 18 |
| `results/v5/final_ablation_matrix.csv` | present | Notebook 19 |
| `results/v5/final_ablation_matrix.json` | present | Notebook 19 |
| `reports/v5_campaign/claims_register_v2.md` | present | Notebook 19 |
| `reports/v5_campaign/claims_register_v2.json` | present | Notebook 19 |
| `reports/v5_campaign/phase10_readiness_gate.json` | present | Notebook 19 |
| `reports/v5_campaign/phase10_manuscript_readiness_gate.md` | present | Notebook 19 |

Existing v5 figure artifacts:

- `results/v5/figures/soc_initial_bias_by_temperature.png`
- `results/v5/figures/routing_conflict_by_decimation_mode.png`
- `results/v5/figures/v4_vs_v5_rmse_by_model.png`
- `results/v5/figures/v5_temperature_breakdown.png`
- `results/v5/figures/multiseed_rmse_boxplot.png`
- `results/v5/figures/multiseed_maxe_boxplot.png`
- `results/v5/figures/recursive_policy_temperature_breakdown.png`
- `results/v5/figures/recursive_vs_ekf_temperature_breakdown.png`
- `results/v5/figures/eta_vs_rmse_by_temperature.png`
- `results/v5/figures/eta_vs_delta_ratio.png`
- `results/v5/figures/eta_vs_recursive_drift.png`
- `results/v5/figures/cold_sequence_recursive_case_study.png`
- `results/v5/figures/hot_sequence_recursive_failure_case.png`
- `results/v5/figures/ekf_voltage_residual_case_study.png`

## Missing or Renamed Artifacts

No blocker for notebook creation was found. The core CSV/JSON evidence needed for notebooks 12-19 exists.

Path aliases requested in the task but not present exactly:

| Requested path/name | Actual substitute | Severity |
|---|---|---|
| `reports/v5_campaign/final_v5_research_report.md` | `reports/v5_campaign/phase9_final_v5_report.md` | non-blocking alias mismatch |
| `reports/v5_campaign/manuscript_readiness_gate.md` | `reports/v5_campaign/phase10_manuscript_readiness_gate.md` | non-blocking alias mismatch |
| `reports/v5_campaign/manuscript_rewrite_brief.md` | no exact file found | non-blocking; claims register and phase reports cover rewrite evidence |
| `results/v5/final_figures/` | figures are under `results/v5/figures/` | non-blocking alias mismatch |
| `results/v5/final_tables/` | tables are CSV/JSON files under `results/v5/` subfolders | non-blocking alias mismatch |

Notebook 12 requested explicit envelope-unsatisfiable stats. The available dataset artifact exposes `decimation_defect_rates` with `envelope_exceeded_pct_mean` and `sign_conflict_pct_mean`; no separate field named `envelope_unsatisfiable` was found. Notebook 12 should load the available field and display a clear missing-artifact note if a more specific future file appears absent.

## Proposed New Notebook Plan

Create the final v5 notebook layer under `notebooks/ablation_studies_v5_final/`:

| Notebook | Purpose | Primary source artifacts |
|---|---|---|
| `12_Dataset_v5_Label_Decimation_Correction.ipynb` | Explain v5 label/decimation correction. | `dataset_variant_comparison.csv/json`, v5 figure PNGs |
| `13_v4_vs_v5_Result_Shift_and_Label_Artifact.ipynb` | Show conclusion shift from v4 to v5. | `v4_vs_v5_comparison.csv`, `v5_headline_model_comparison.csv`, `final_v5_model_comparison.csv` |
| `14_Multiseed_Stability_and_Model_Ranking.ipynb` | Prove multi-seed stability and systematic original-HC failure. | `seed_level_results.csv`, `multiseed_summary.csv`, `ranking_stability.json` |
| `15_Anchor_Last_vs_Anchor_First_Observability.ipynb` | Explain anchor_last observability gain. | `v5_headline_model_comparison.csv`, `multiseed_summary.csv`, `final_v5_model_comparison.csv` |
| `16_Windowed_vs_Carried_vs_Load_Gated_Recursive_Inference.ipynb` | Show inference protocol effects. | `recursive_policy_comparison.csv/json` |
| `17_Eta_Calibration_Delta_Path_Rate_Recovery.ipynb` | Show eta calibration and delta-rate recovery. | `eta_gamma_sweep.csv/json`, final comparison |
| `18_Continuous_EKF_1RC_Correction_vs_Consistency_Frontier.ipynb` | Compare calibrated HC vs EKF/ECM baselines. | `recursive_vs_ekf_comparison.csv`, `continuous_ekf_results.json` |
| `19_Final_Ablation_Matrix_and_Claims_Register.ipynb` | Consolidate 62-row matrix, 17 claims, and 52/52 gate. | `final_ablation_matrix.csv/json`, `claims_register_v2.json/md`, `phase10_readiness_gate.json` |

## Legacy Notebook Notes Required

Add `notebooks/ablation_studies/README_LEGACY.md` with a global warning:

> This notebook stack belongs to the v4/early-hypothesis evidence layer. It is preserved for traceability. Final conclusions should use `notebooks/ablation_studies_v5_final/` and `reports/v5_campaign/`.

Explicit updates to mention in the README:

- `10_Gated_Context_Negative_Result_Sparse_Rest_Validity.ipynb`: legacy negative result; v5 Phase 4 later found load-gated recursive inference positive.
- `04_Direction_Only_Hard_Constraint_Cumulative_Drift.ipynb`: v5 Phase 5 shows this drift is largely correctable by eta calibration.
- `08_Anchor_Trap_and_Safety_Factor_NonCause.ipynb`: v5 confirms anchor bottleneck but updates final solution to anchor_last + calibrated carried inference.

## Screening Decision

Proceed with notebook creation. Key v5 evidence is present, legacy notebooks can remain untouched, and missing items are alias mismatches or optional detail files rather than blockers.
