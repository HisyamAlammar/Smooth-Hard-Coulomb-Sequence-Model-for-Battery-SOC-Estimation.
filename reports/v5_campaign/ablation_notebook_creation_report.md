# v5 Ablation Notebook Creation Report

Generated: 2026-07-03

## Summary

Created the final v5 ablation notebook stack under:

`notebooks/ablation_studies_v5_final/`

No heavy training was run. The notebooks load completed artifacts from `results/v5/` and `reports/v5_campaign/` and act as analysis, visualization, and manuscript-evidence layers.

## Notebooks Created

| Notebook | Primary claim supported |
|---|---|
| `12_Dataset_v5_Label_Decimation_Correction.ipynb` | v5 separates true model behavior from v4 label/decimation artifacts. |
| `13_v4_vs_v5_Result_Shift_and_Label_Artifact.ipynb` | Corrected labels reduce artifact-driven catastrophic errors while sharpening the OOD HC contribution. |
| `14_Multiseed_Stability_and_Model_Ranking.ipynb` | `anchor_last` is the stable candidate; original HC Scenario-B failure is systematic across 5/5 seeds. |
| `15_Anchor_Last_vs_Anchor_First_Observability.ipynb` | Cold-temperature error is dominated by anchor observability, not delta-path failure. |
| `16_Windowed_vs_Carried_vs_Load_Gated_Recursive_Inference.ipynb` | Independent windowing manufactures repeated cold-starts; load-gated recursive inference reduces this artifact. |
| `17_Eta_Calibration_Delta_Path_Rate_Recovery.ipynb` | Inference envelope calibration fixes delta-path underestimation; eta*=2.0 moves delta ratio from ~0.751 to ~1.002. |
| `18_Continuous_EKF_1RC_Correction_vs_Consistency_Frontier.ipynb` | EKF corrects through voltage feedback but violates consistency; calibrated HC outperforms EKF in this setup. |
| `19_Final_Ablation_Matrix_and_Claims_Register.ipynb` | The manuscript is ready for conservative rewrite using the 62-row ablation matrix, 17-claim register, and 52/52 readiness gate. |

## Artifacts Used by Notebook

| Notebook | Artifacts |
|---|---|
| 12 | `results/v5/dataset_variant_comparison.csv`, `results/v5/dataset_variant_comparison.json`, `reports/v5_campaign/phase1_dataset_v5_report.md`, `results/v5/figures/soc_initial_bias_by_temperature.png`, `results/v5/figures/routing_conflict_by_decimation_mode.png` |
| 13 | `results/v5/headline_models/v4_vs_v5_comparison.csv`, `results/v5/headline_models/v5_headline_model_comparison.csv`, `results/v5/final_v5_model_comparison.csv`, `results/v5/figures/v4_vs_v5_rmse_by_model.png` |
| 14 | `results/v5/multiseed/seed_level_results.csv`, `results/v5/multiseed/multiseed_summary.csv`, `results/v5/multiseed/ranking_stability.json`, `results/v5/figures/multiseed_rmse_boxplot.png`, `results/v5/figures/multiseed_maxe_boxplot.png` |
| 15 | `results/v5/multiseed/multiseed_summary.csv`, `results/v5/headline_models/v5_headline_model_comparison.csv`, `results/v5/final_v5_model_comparison.csv` |
| 16 | `results/v5/recursive_inference/recursive_policy_comparison.csv`, `results/v5/recursive_inference/recursive_policy_results.json`, `results/v5/figures/recursive_policy_temperature_breakdown.png`, `results/v5/figures/cold_sequence_recursive_case_study.png`, `results/v5/figures/hot_sequence_recursive_failure_case.png` |
| 17 | `results/v5/delta_calibration/eta_gamma_sweep.csv`, `results/v5/delta_calibration/eta_gamma_sweep.json`, `results/v5/final_v5_model_comparison.csv`, `results/v5/figures/eta_vs_rmse_by_temperature.png`, `results/v5/figures/eta_vs_delta_ratio.png`, `results/v5/figures/eta_vs_recursive_drift.png` |
| 18 | `results/v5/ekf_ecm/recursive_vs_ekf_comparison.csv`, `results/v5/ekf_ecm/continuous_ekf_results.json`, `results/v5/delta_calibration/eta_gamma_sweep.csv`, `results/v5/figures/recursive_vs_ekf_temperature_breakdown.png`, `results/v5/figures/ekf_voltage_residual_case_study.png` |
| 19 | `results/v5/final_ablation_matrix.csv`, `results/v5/final_ablation_matrix.json`, `results/v5/final_v5_model_comparison.csv`, `reports/v5_campaign/claims_register_v2.json`, `reports/v5_campaign/claims_register_v2.md`, `reports/v5_campaign/phase10_readiness_gate.json`, `reports/v5_campaign/phase10_manuscript_readiness_gate.md`, `reports/v5_campaign/phase7_final_ablation_model_card.md` |

## Missing Artifacts

Blocking missing artifacts: none.

Non-blocking missing or renamed artifacts:

| Item | Handling |
|---|---|
| `results/v5/anchor_error_diagnostics.csv` | Optional diagnostic only. Notebook 15 records it as missing and uses multiseed temperature RMSE plus headline anchor variants as the audited proxy. |
| `results/v5/final_figures/` | Not present. Existing figures are under `results/v5/figures/`. |
| `results/v5/final_tables/` | Not present. Existing tables are CSV/JSON files under `results/v5/` subfolders. |
| `reports/v5_campaign/final_v5_research_report.md` | Renamed/equivalent report exists as `reports/v5_campaign/phase9_final_v5_report.md`. |
| `reports/v5_campaign/manuscript_readiness_gate.md` | Renamed/equivalent report exists as `reports/v5_campaign/phase10_manuscript_readiness_gate.md`. |
| `reports/v5_campaign/manuscript_rewrite_brief.md` | No exact file found; claims register and phase reports supply rewrite evidence. |

## Legacy Notebook Handling

Old notebooks were not moved, deleted, or overwritten.

Created:

`notebooks/ablation_studies/README_LEGACY.md`

This marks the original `notebooks/ablation_studies/` stack as v4/early-hypothesis evidence and redirects final claims to:

- `notebooks/ablation_studies_v5_final/`
- `results/v5/`
- `reports/v5_campaign/`

Explicit legacy warnings were added in the README for:

- `04_Direction_Only_Hard_Constraint_Cumulative_Drift.ipynb`
- `08_Anchor_Trap_and_Safety_Factor_NonCause.ipynb`
- `10_Gated_Context_Negative_Result_Sparse_Rest_Validity.ipynb`

## Validation Performed

- Generated all eight notebooks using `tools/create_v5_ablation_notebooks.py`.
- Parsed all `.ipynb` files as JSON.
- Compiled every code cell with Python `compile()`.
- Executed every code cell with a lightweight local executor using matplotlib `Agg` backend.
- No heavy training or model reruns were executed.

`jupyter nbconvert` was not available in the current shell PATH, so notebook execution was validated by direct code-cell execution instead.

## Manual Review Needed

Before manuscript rewrite, manually inspect Notebook 15 and decide whether to create a dedicated `results/v5/anchor_error_diagnostics.csv` artifact. The current notebook is scientifically usable because it relies on official multiseed temperature RMSE and anchor-variant comparisons, but a dedicated anchor-error CSV would make the observability section cleaner.

Also review whether to add alias files or symlinks for:

- `phase9_final_v5_report.md` -> `final_v5_research_report.md`
- `phase10_manuscript_readiness_gate.md` -> `manuscript_readiness_gate.md`

## Next Recommended Step

Use Notebook 19 as the manuscript rewrite control panel. Rewrite the paper around:

`anchor_last + calibrated carried inference`

and keep unsupported claims out:

- no full functional-safety compliance claim,
- no hardware readiness claim without WCET/INT8/HIL,
- no universal eta claim beyond the audited single-checkpoint calibration evidence.
