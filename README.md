# Smooth Hard-Coulomb Sequence Model for Battery SOC Estimation

This repository contains the research code for a physics-constrained deep learning pipeline for lithium-ion battery State of Charge (SOC) estimation. The final method is a **Smooth Hard-Coulomb Sequence Model**: an LSTM/TCN sequence estimator whose output layer enforces physically valid SOC motion during discharge by construction.

The repository is currently organized for **project/code review**. Manuscript drafts, raw datasets, trained checkpoints, local virtual environments, and archived experiments are intentionally excluded through `.gitignore` so supervisors can inspect the source code and research logic without receiving a multi-gigabyte workspace.

## Research Summary

SOC estimation is safety-critical in Battery Management Systems (BMS). Standard data-driven models can achieve reasonable average error while still producing physically invalid trajectories, for example predicting SOC increases while the battery is actively discharging.

This project studies that failure mode and proposes a structural fix:

> Instead of asking a loss function to learn physics softly, the final model makes physics-violating SOC increments unreachable in the output space.

The final Smooth Hard-Coulomb layer routes predicted SOC increments according to the current sign and bounds their magnitude by a Coulomb-counting envelope. A sigmoid-scaled magnitude keeps the layer differentiable, avoiding the dead-gradient pathology of earlier clamp-based variants.

## Main Hypothesis

The central hypothesis is:

1. Vanilla sequence models such as LSTM and TCN optimize prediction error but do not guarantee physical safety.
2. Soft physics penalties reduce violations but cannot guarantee zero Physics Violation Rate (PVR).
3. A hard structural Coulomb constraint can guarantee `0.00%` PVR while preserving gradient flow through a smooth sigmoid magnitude parameterization.
4. Remaining large errors at `-20 C` are best explained as an empirical observability bottleneck of the initial SOC anchor, not as a failure of the Coulomb constraint.

## Final Model Idea

For each timestep, the model predicts raw logits rather than raw signed SOC deltas. The Smooth Hard-Coulomb layer converts those logits into physically routed increments:

```text
limit_t = |I_t| * dt / (Q_nominal * 3600) * eta
mag_frac_t = sigmoid(delta_logit_t)

if I_t < -threshold: delta_t = -limit_t * mag_frac_t
if I_t >  threshold: delta_t =  limit_t * mag_frac_t
else:                 delta_t = 0

cumulative_t = cumsum(delta_t)
soc_anchor = lo + width * sigmoid(anchor_logit)
soc_pred_t = soc_anchor + cumulative_t
```

Core implementation files:

- `src/model_v5_coulomb.py` - Smooth Hard-Coulomb LSTM and shared constraint layer.
- `src/model_v5_coulomb_tcn.py` - Smooth Hard-Coulomb TCN backbone.
- `src/model_v6_contextual.py` - Contextual anchor variants.
- `src/preprocessing_v4.py` - zero-leakage preprocessing for final non-contextual models.
- `src/preprocessing_v5_contextual.py` - contextual preprocessing with causal history features.

## Key Results

Final post-refactor evaluation logs are stored as small JSON files under `outputs/`.

| Model | Scenario | RMSE (%) | MaxE (%) | PVR (%) | Parameters |
|---|---|---:|---:|---:|---:|
| Vanilla LSTM | A | 13.3712 | 51.0242 | 49.9694 | 53,569 |
| Hard-Coulomb LSTM | A | 12.7107 | 55.1126 | 0.0000 | 54,626 |
| Hard-Coulomb TCN | A | 11.4587 | 46.7298 | 0.0000 | 208,546 |
| Vanilla LSTM | B | 7.2806 | 48.7994 | 41.0552 | 53,569 |
| Hard-Coulomb LSTM | B | 8.5667 | 34.9985 | 0.0000 | 54,626 |
| Hard-Coulomb TCN | B | 8.5823 | 39.4864 | 0.0000 | 208,546 |

Primary metric files:

- `outputs/v7_final/sprint48_evaluation_results.json`
- `outputs/v7_final/sprint48_safety_ablation_results.json`
- `outputs/v5_contextual/sprint50_contextual/sprint50_contextual_results.json`
- `outputs/v8_tcn_redemption/sprint52/sprint52_tcn_redemption_results.json`

## Repository Layout

```text
src/
  config.py                         Shared configuration values.
  hppc_rint_extractor.py            R_int / V_proxy feature support.
  preprocessing_v4.py               Final non-contextual zero-leakage pipeline.
  preprocessing_v5_contextual.py     Contextual preprocessing pipeline.
  model_v5_coulomb.py               Smooth Hard-Coulomb LSTM.
  model_v5_coulomb_tcn.py           Smooth Hard-Coulomb TCN.
  model_v6_contextual.py            Contextual Hard-Coulomb models.
  sprint48_train_scenario_A.py      Train Vanilla/Hard-Coulomb LSTM Scenario A.
  sprint48_train_scenario_B.py      Train Vanilla/Hard-Coulomb LSTM Scenario B.
  sprint48_evaluate_all.py          Final LSTM evaluation and PVR certification.
  sprint48_safety_ablation.py       Safety-factor ablation.
  sprint50_train_contextual.py      Contextual anchor experiments.
  sprint52_tcn_redemption.py        TCN and contextual TCN evaluation.

notebooks/
  05_q1_eda_money_plots.py          EDA figures for observability collapse.
  ablation_studies/                 Eleven forensic ablation notebooks.

tools/
  generate_data_audit_tables.py     Raw integrity and scenario composition tables.

outputs/
  data_audit_tables.json            Small data audit ledger.
  figures/                          Selected review figures.
  v7_final/                         Final LSTM metrics.
  v5_contextual/                    Contextual metrics.
  v8_tcn_redemption/                Final TCN metrics.
```

Large raw data, processed arrays, trained checkpoints, manuscript drafts, and archive folders are ignored for code review.

## Environment Setup

Recommended Python version: Python 3.10 or newer.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

For Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproducibility Pipeline

The full pipeline expects the LG HG2-style raw battery dataset to be available locally under `data/raw/`. The raw and processed data are intentionally not committed because they are large.

### 1. Build Processed Arrays

```bash
python src/preprocessing_v4.py
python src/preprocessing_v5_contextual.py
```

### 2. Train Final LSTM Models

```bash
python src/sprint48_train_scenario_A.py
python src/sprint48_train_scenario_B.py
```

### 3. Train Contextual Anchor Models

```bash
python src/sprint50_train_contextual.py
```

### 4. Train/Evaluate TCN Models

```bash
python src/sprint52_tcn_redemption.py
```

### 5. Evaluate Final Models

```bash
python src/sprint48_evaluate_all.py
```

### 6. Generate Data Audit Tables and EDA Figures

```bash
python tools/generate_data_audit_tables.py
python notebooks/05_q1_eda_money_plots.py
```

## Ablation Studies

The ablation notebooks document why the final architecture was necessary:

| Notebook | Purpose |
|---|---|
| `01_Seq2Point_Windowing_Artifact_and_Pseudo_PVR.ipynb` | Shows pseudo-trajectory artifacts from pointwise SOC estimation. |
| `02_Vanilla_LSTM_TCN_Physics_Blindness.ipynb` | Shows high PVR from unconstrained sequence models. |
| `03_Soft_PINN_Penalty_Gradient_Collision.ipynb` | Shows that soft penalties reduce but do not eliminate violations. |
| `04_Direction_Only_Hard_Constraint_Cumulative_Drift.ipynb` | Shows monotonic-only constraints can drift without magnitude bounds. |
| `05_Hard_Clamp_vs_Smooth_Coulomb_Gradient_Pathology.ipynb` | Shows why sigmoid-scaled magnitude is preferred over clamp. |
| `06_Vproxy_HPPC_Rint_Feature_Defense.ipynb` | Defends V_proxy and Ohmic-drop compensation. |
| `07_Zero_Leakage_Split_Before_Windowing_Forensics.ipynb` | Verifies split-before-windowing and zero temporal overlap. |
| `08_Anchor_Trap_and_Safety_Factor_NonCause.ipynb` | Shows cold MaxE is not mainly caused by eta restriction. |
| `09_Contextual_Anchor_OCV_Rest_vs_History.ipynb` | Shows rest-based OCV context helps more than history-only context. |
| `10_Gated_Context_Negative_Result_Sparse_Rest_Validity.ipynb` | Explains why gated context failed under sparse cold rest evidence. |
| `11_HardCoulomb_LSTM_vs_TCN_Backbone_Tradeoff.ipynb` | Compares final LSTM and TCN backbones under the same constraint. |

## Figures for Review

Selected figures are kept for quick inspection:

- `outputs/figures/fig_q1_observability_collapse.png`
- `outputs/figures/fig_q1_transient_dynamic_profile.png`
- `outputs/figures/ablation_studies/fig_02_vanilla_physics_blindness_pvr.png`
- `outputs/figures/ablation_studies/fig_05_clamp_vs_smooth_gradient_pathology.png`
- `outputs/figures/ablation_studies/fig_08_anchor_trap_eta_non_cause.png`
- `outputs/figures/ablation_studies/fig_11_hardcoulomb_lstm_vs_tcn_tradeoff.png`

## Known Limitations

This repository is ready for technical review, but the research still has limitations:

- No EKF/UKF/ECM observer baseline is included in the final comparison.
- Final results are not yet reported with multi-seed variance.
- Parameter counts suggest embedded feasibility, but MCU latency, WCET, quantization, and hardware-in-the-loop validation remain future work.
- The PVR guarantee assumes valid current measurements, correct timestep handling, and the defined current threshold.
- High MaxE at `-20 C` remains an observability bottleneck caused by weak cold-temperature voltage/SOC anchoring.

## Notes for Supervisors and Reviewers

For code review, start with:

1. `src/model_v5_coulomb.py` - verify Smooth Hard-Coulomb logic.
2. `src/preprocessing_v4.py` - verify split-before-windowing and no leakage.
3. `src/sprint48_evaluate_all.py` - verify RMSE, MaxE, and PVR computation.
4. `notebooks/ablation_studies/02_Vanilla_LSTM_TCN_Physics_Blindness.ipynb` - inspect baseline PVR failure.
5. `notebooks/ablation_studies/11_HardCoulomb_LSTM_vs_TCN_Backbone_Tradeoff.ipynb` - inspect final backbone comparison.

The manuscript drafts are intentionally ignored at this stage because the current handoff is for project/source-code correction first.
