# V5 Campaign — Phase 2: Headline Retraining on v5c

Date: 2026-07-03. Seed 42 (comparable to the v4 published run; multi-seed in Phase 3). Artifacts: `experiments/run_multiseed_v5.py`, `baselines/baselines_v5.py`, `analysis/compare_v4_v5.py`, `results/v5/headline_models/{v5_headline_model_comparison.csv,json, v4_vs_v5_comparison.csv, runs_v5c_A-B_42.json}`, `results/v5/baselines/deterministic_baselines_v5c.json`, figures `v4_vs_v5_rmse_by_model.png`, `v5_temperature_breakdown.png`.

## Headline table (v5c, full-sequence RMSE %, seed 42)

| Model | Scen A | A −20 °C | A MaxE | Scen B | B MaxE |
|---|---:|---:|---:|---:|---:|
| null OCV+Coulomb (0 params) | 13.03 | 19.57 | 60.94 | 7.53 | 94.72 |
| Vanilla LSTM | 11.59 | 17.38 | 46.88 | 6.35 | 50.64 |
| HC-LSTM (original anchor) | 11.05 | 16.76 | 46.20 | 9.87 | 35.36 |
| HC-TCN | 10.95 | 16.65 | 43.29 | 7.45 | 41.86 |
| HC anchor_pooled | 10.74 | 16.62 | 42.02 | **4.63** | 33.07 |
| **HC anchor_last** | **10.63** | 16.54 | **37.85** | 4.68 | **31.13** |
| Vanilla + post-hoc clamp | 25.03 | 23.19 | 46.88 | 30.07 | 50.64 |
| null + oracle anchor (ref) | 0.03 | 0.03 | 0.23 | 0.02 | 0.42 |

v4→v5c deltas (same models, same seed protocol): every model improves on Scenario A (vanilla −1.78 pp, HC-LSTM −1.66 pp, TCN −0.51 pp, null −0.21 pp) — consistent with part of the v4 "error" having been label contamination. Scenario B: vanilla −0.93 pp, null −0.03 pp, HC-LSTM **+1.31 pp** (worse; its Scenario-B run early-stopped at epoch 23 with a high val loss — flagged for the multi-seed check before interpretation).

## Answers to the acceptance questions

1. **Does v5 change the conclusions?** Yes, in three ways. (a) The OOD case for Hard-Coulomb *strengthens*: HC-LSTM now beats the null model by 1.98 pp (was 0.53 pp) and every HC variant beats vanilla and null on Scenario A. (b) The **anchor redesign is vindicated**: anchor_last/pooled — inconclusive on v4 labels — are now the best models in *both* scenarios (Scen B 4.6–4.7 % vs null 7.5 %, vanilla 6.3 %), the first configuration that clearly beats the null model in-distribution. (c) Part of v4's catastrophic-MaxE story was label artifact: the null model's Scenario-A MaxE fell 87→61 % just from corrected labels.
2. **Does HC still beat post-hoc clamp?** Yes, unchanged and decisive (25.0/30.1 % clamp collapse).
3. **HC vs null?** Original HC-LSTM: clearly beats null OOD, still loses in-distribution (9.87 vs 7.53). Anchor-last/pooled variants beat null everywhere. The claim must be variant-specific.
4. **Is −20 °C still anchor-dominated?** Yes: oracle-anchor null reaches 0.03 % RMSE at −20 °C while all learned models sit at 16.5–17.4 % — the gap is anchor error by construction of the reference. (Formal per-model oracle re-run rolls into Phase 4/7 evidence.)
5. **Is MaxE a stronger claim than RMSE?** Still yes, but moderated: HC-family MaxE 37.9–46.2 % vs null 60.9 % (A) and 31–42 % vs 94.7 % (B). The v4-sized gap (55 vs 87) was partly label artifact; the corrected gap remains large and favors the learned anchor + bounded windows.

## Provenance

All runs record dataset_version=v5c, label_mode=ohmic_corrected, decimation_mode=mean_per_second, scenario, seed, checkpoint path, git commit (`3c4ba9b` + working tree), timestamps — inside `runs_v5c_A-B_42.json` and the baselines JSON.

Anomaly log: HC-LSTM Scenario-B early stop (epoch 23, val 0.0051 vs anchor variants ~0.0019-0.0020) — possible bad seed/optimization run; Phase 3 multi-seed will decide whether the +1.31 pp regression is real or seed noise.
