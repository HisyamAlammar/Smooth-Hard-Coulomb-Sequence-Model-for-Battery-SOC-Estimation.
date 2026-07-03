# Final v5 Ablation Study Notebooks

This folder contains the final v5 evidence notebooks for the Hard-Coulomb SOC research campaign.

## Purpose

These notebooks are analysis and manuscript-support notebooks. They load existing v5 result artifacts, summarize evidence, generate reviewer-facing plots/tables, and clarify which claims are supported by the completed campaign.

They do not retrain models and must not be treated as training scripts.

## Source of Truth

The source of truth is the completed v5 artifact ledger:

- `results/v5/`
- `results/v5/multiseed/`
- `results/v5/recursive_inference/`
- `results/v5/delta_calibration/`
- `results/v5/ekf_ecm/`
- `results/v5/final_ablation_matrix.csv`
- `results/v5/final_v5_model_comparison.csv`
- `reports/v5_campaign/claims_register_v2.md`
- `reports/v5_campaign/phase9_final_v5_report.md`
- `reports/v5_campaign/phase10_manuscript_readiness_gate.md`

## Final Selected System

The selected v5 system is:

`anchor_last + calibrated carried inference`

Conservative headline evidence:

- `anchor_last`: Scenario A `9.99 ± 1.09%` RMSE, Scenario B `4.74 ± 0.31%` RMSE.
- Original Hard-Coulomb Scenario-B failure is systematic across `5/5` seeds.
- Load-gated recursive inference: `8.41%` RMSE, about `-2.6 pp` versus windowed inference.
- Eta calibration: delta ratio improves from `0.751` to `1.002`.
- Calibrated recursive Hard-Coulomb: `4.43%` RMSE.
- Calibrated recursive Hard-Coulomb at `-20 C`: `3.59%` RMSE.
- Best EKF: `6.85%` RMSE and nonzero PVR violations.
- Final ablation matrix: `62` rows.
- Claims register v2: `17` claims.
- Manuscript readiness gate: `52/52 PASS`.

## Notebook List

| Notebook | Role |
|---|---|
| `12_Dataset_v5_Label_Decimation_Correction.ipynb` | Explains why v5 corrected labels/decimation were required. |
| `13_v4_vs_v5_Result_Shift_and_Label_Artifact.ipynb` | Shows how v5 changes conclusions versus legacy v4. |
| `14_Multiseed_Stability_and_Model_Ranking.ipynb` | Summarizes multi-seed robustness and model ranking stability. |
| `15_Anchor_Last_vs_Anchor_First_Observability.ipynb` | Explains why window-informed anchor design improves observability. |
| `16_Windowed_vs_Carried_vs_Load_Gated_Recursive_Inference.ipynb` | Shows why recursive inference protocol matters. |
| `17_Eta_Calibration_Delta_Path_Rate_Recovery.ipynb` | Documents eta calibration and delta-path rate recovery. |
| `18_Continuous_EKF_1RC_Correction_vs_Consistency_Frontier.ipynb` | Compares calibrated HC against EKF/ECM recursive baselines. |
| `19_Final_Ablation_Matrix_and_Claims_Register.ipynb` | Consolidates the final ablation matrix, claims register, and readiness gate. |

## Reviewer Use

Use these notebooks to support manuscript rewriting. Keep old `notebooks/ablation_studies/` files as legacy traceability evidence, not as the final source of claims.
