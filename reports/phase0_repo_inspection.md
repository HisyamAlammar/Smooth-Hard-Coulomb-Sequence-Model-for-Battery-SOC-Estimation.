# Phase 0 — Repo Inspection and Fix Plan

Date: 2026-07-03. Basis: red-team audit of Hard-Coulomb SOC repo (session audit + `scratchpad/audit_pvr_deadband.py` run).

## Relevant files

| File | Role | Audit relevance |
|---|---|---|
| `src/config.py` | Hyperparams, `CURRENT_THRESHOLD=0.05`, `Q_NOMINAL`, `R_INT_PER_TEMP` | Single source of truth target for thresholds |
| `src/model_v5_coulomb.py` | `SmoothHardCoulombConstraint`, `HardCoulombLSTM` (anchor from `h[:,0,:]`) | Findings 1,2,4,7 |
| `src/preprocessing_v4.py` | 1 Hz decimation (first sample/second), per-segment `soc_initial=OCV(V_first)`, `cap_used=abs(cap-cap0)` | Finding 9 |
| `src/sprint48_common.py` | `VanillaLSTM`, loaders, training loop | Finding 11 |
| `src/sprint48_evaluate_all.py` | Unified eval, `compute_pvr` (hardcoded −0.05, discharge-only, eps=0) | Findings 1,2,12 |
| `src/sprint48_safety_ablation.py` | Anchor/eta ablation, duplicated `compute_pvr` | Finding 12 |
| `outputs/v7_final/*.pt` | Final checkpoints (vanilla + HC, scenarios A/B) | Inputs to all new experiments |
| `data/processed/v4_scenario_{A,B}/` | Windowed tensors + `metadata_v4.json` + `temp_labels_test.npy` + `timestamp_key_*.npy` | Shared protocol for all baselines |
| `data/raw/LG Dataset/LG_HG2_Original_Dataset/` | Raw 10 Hz CSVs incl. HPPC (OCV source) | Phase 5 audit, Phase 2/8 OCV lookup |

Existing measured facts driving the plan (from audit session, reproducible):

- HC-LSTM full-seq error ≈ raw anchor-head error (Scen A: MAE 9.87% vs 10.13%); delta-path adjustment ≈ 0.48%.
- Eta ablation at inference (η 1.0→3.0) moves RMSE by 0.017% → delta path near-inert w.r.t. envelope width.
- Vanilla per-window predicted SOC range 28.3% vs true 1.4% (Scen A) → oscillation pathology.
- Vanilla PVR falls 49.97→8.05% (A) and 41.06→2.71% (B) at ε=0.005/step dead-band.
- HC mean |Δpred| during discharge undershoots |Δtrue| (A: 0.00016 vs 0.00022; B: 0.00009 vs 0.00022).
- Training cost is trivial (≤0.5 min/model on RTX 4060) → retraining variants (Phase 6) is cheap.

## Output structure created

`experiments/`, `baselines/`, `analysis/`, `results/{metrics,baselines,diagnostics,figures,robustness,model_variants,edge}/`, `reports/`.

Existing results in `outputs/` are preserved untouched; new code reads them, never overwrites.

## Skills

AI Research Skills installed at `~/.claude/skills/` (86 skills). Read for discipline: `0-autoresearch-skill` (experiment/synthesis loop, machine-readable state), `ml-training-recipes` (training hygiene for Phase 6). `academic-plotting`/`dataviz` to be loaded before figure generation. `ml-paper-writing` deferred until manuscript phase (out of scope here).

## Phase plan (execution order)

1. **Metrics first** — central thresholds in `config.py`; new `analysis/soc_metrics.py` (region-split ε-PVR: discharge/charge/rest/total; delta-magnitude tracking; regression metrics); sanity tests; refactor both sprint48 eval scripts onto it; rerun eval.
2. **Novelty-critical baselines** — `baselines/null_ocv_coulomb.py` (zero-parameter OCV-inverse + Coulomb), `baselines/posthoc_clamp.py` (trained vanilla + inference-time Hard-Coulomb envelope). One comparison table.
3. **Vanilla fairness diagnosis** — range/delta diagnostics, example trajectories, training-code review; fix only if a real bug is found.
4. **Anchor trap** — oracle-anchor experiment (true SOC at t=0, learned delta path kept), anchor-error analysis vs temperature/current/voltage/rest.
5. **Preprocessing/label audit** — intra-second current statistics vs decimated samples; segment-start load bias on `soc_initial`; `abs(cap-cap0)` sign-excursion check. Audit-only; no silent label changes.
6. **Model variants (evidence-gated)** — anchor-last / anchor-pooled variants (retrain, cheap), carried-anchor recursive inference (no retrain).
7. **Sensor-fault stress** — current offset/noise/stuck-at-zero; PVR vs measured and vs true current; forced-drift quantification.
8. **EKF baseline** — OCV-Rint EKF on V_proxy measurement model (V_proxy = OCV(SOC)), vectorized across windows; documented assumptions; no test-set tuning.
9. **Edge/quantization** — analytic MACs/size; PyTorch dynamic quantization PVR check; simulated integer delta-rounding sign-preservation check.
10. **Final report** — aggregated comparison table + claims register mapping evidence → claim status.

## Ground rules honored

- No existing results deleted; legacy JSON field names (`pvr_pct`, etc.) preserved for backward compatibility, new metrics added alongside.
- Every experiment emits JSON (+CSV where tabular) with dataset split, scenario, checkpoint, seed, timestamp, config values.
- Failures/blockers get written into phase reports, not faked.
