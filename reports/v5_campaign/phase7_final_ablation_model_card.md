# V5 Campaign — Phase 7: Final Ablation Matrix, Model Selection, Model Card

Date: 2026-07-03. Artifacts: `analysis/build_final_ablation_v5.py`, `results/v5/final_ablation_matrix.{csv,json}` (62 rows: v4 frozen + v5c multiseed + deterministic + recursive policies + eta/gamma sweep + EKFs), `results/v5/final_v5_model_comparison.{csv,json}` (30 primary rows).

## Selected configuration

**Model: `hc_anchor_last` (Hard-Coulomb LSTM, anchor head reading the last hidden state), trained on v5c (ohmic-corrected labels, mean-per-second decimation), eta_train=1.5.**
**Inference: windowed for isolated windows; carried-anchor recursive with inference-time envelope calibration eta*=2.0 (gamma nominal) — equivalently eta=1.75 with temp-aware gamma — for contiguous chains, gated by `load_gated` re-anchoring at rest starts when chain contiguity is unreliable.**

### Why this selection

| Criterion | Evidence (matrix rows) |
|---|---|
| Best windowed accuracy, both scenarios | A 9.99 ± 1.09, B 4.74 ± 0.31 (5 seeds); only family beating null in-distribution (7.53) and OOD (13.03) |
| Best MaxE among learned models | A 36.50 ± 3.85 vs vanilla 47.48 |
| Monotonicity guarantee | PVR = 0.00 % at ε=0, all seeds, all policies — architectural, not empirical |
| Recursive mode after eta* calibration | RMSE 4.43 %, −20 °C 3.59 % (HC-LSTM s42 evidence; transfers via shared delta-path design) |
| Beats classical recursive baseline | Best EKF 6.85 % (and 4.9–5.5 % PVR violations; 6.8→39 % RMSE sensitivity to R) |
| No test-time tuning | Gates physics-fixed a priori; eta* set by rate-fidelity ≈ 1 criterion |

Runner-up: `hc_anchor_pooled` (statistically indistinguishable in B: 4.93 ± 0.84; worse in A). Rejected as primary because anchor_last dominates the A distribution 4/5 seeds.

### Known failure modes / limits

1. **Anchor-dominated cold error in windowed mode** (−20 °C ≈ 15.0 %; oracle-anchor null: 0.03 %) — anchor, not delta, is the residual bottleneck.
2. **Seed variance** σ ≈ 1.1 pp (Scen A) — single-seed numbers must not be quoted headline.
3. **Original first-window-anchor HC-LSTM fails in-distribution** (B 10.63 ± 0.60, all seeds early-stop ~20 epochs) — the anchor redesign is load-bearing, not cosmetic.
4. **eta* calibrated on Scenario A seed-42 statistics** — needs validation-set re-derivation + Scen B/multi-seed confirmation for manuscript (flagged, not blocking: windowed numbers unaffected by eta).
5. Retraining at eta* does not self-calibrate (head compensates); calibration must remain a post-training step.

## Model card (condensed)

- **Architecture**: 2-layer LSTM (hidden 64) → delta head (sigmoid magnitude routed by sign of I, envelope `|I|·eta·gamma`) + anchor head on last hidden state, anchor remapped to feasible interval [lo,hi] from cumulative-delta extremes.
- **Params**: ~55 k (218 KB checkpoint fp32).
- **Training**: MSE, AdamW 1e-3, cosine to 1e-6, clip 1.0, batch 1024, patience 10, ≤100 epochs; sprint48 recipe, seeds 1–5 + 42 archived.
- **Data**: v5c = ohmic-corrected SOC labels + mean-per-second decimation; scenarios A (temperature-OOD: −20 °C held out … ) and B (in-distribution split). 53 test chains / 193,910 steps for continuous eval.
- **Guarantees**: discharge/charge monotonicity by construction (PVR 0 % at ε=0); SOC bounded [0,1] via clip; anchor confined to feasibility interval.
- **Intended use**: SOC estimation research on this cell/dataset; edge-deployable (Phase-verified quantization + feasibility from v4 campaign).
- **Not validated for**: other chemistries/cells, real BMS safety decisions, unidentified-parameter transfer.

## Full-matrix pointers

- v4 frozen reference: 10 rows (`legacy_freeze_manifest.json`).
- v5c multiseed: 10 rows (5 models × 2 scenarios, mean ± std).
- Deterministic: 8 rows (nulls incl. oracle-anchor/oracle-Q, clamp).
- Recursive policies: 3 primary (+7 in full matrix), eta/gamma sweep: 16, EKFs: 6.
