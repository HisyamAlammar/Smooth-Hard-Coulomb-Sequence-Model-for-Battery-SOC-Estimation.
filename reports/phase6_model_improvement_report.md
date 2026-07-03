# Phase 6 — Model Improvement Experiments (Anchor Variants)

Date: 2026-07-03. Artifacts: `experiments/compare_anchor_variants.py`, `results/model_variants/anchor_variant_comparison.csv`, `results/model_variants/anchor_variant_results.json`, checkpoints under `results/model_variants/checkpoints/` (production `outputs/v7_final` checkpoints untouched). Scenario A (cold-OOD), training recipe identical to sprint48; retrained variants use seeds {42, 123, 2026}.

## Results (Scenario A test; full-sequence)

| Variant | Seed | RMSE % | MaxE % | −20 °C RMSE % | −20 °C MaxE % | PVR disch ε=0 |
|---|---|---:|---:|---:|---:|---:|
| anchor_first (production) | prod | 12.71 | 55.11 | 17.86 | 55.11 | 0.00 |
| anchor_last | 42 | 13.37 | 50.29 | 19.45 | 50.29 | 0.00 |
| anchor_last | 123 | 11.47 | 38.50 | 15.57 | 38.50 | 0.00 |
| anchor_last | 2026 | 11.09 | 37.01 | 14.25 | 37.01 | 0.00 |
| anchor_pooled | 42 | 12.18 | 47.51 | 16.43 | 47.51 | 0.00 |
| anchor_pooled | 123 | 20.00 | 66.60 | 31.47 | 66.60 | 0.00 |
| anchor_pooled | 2026 | 12.39 | 48.23 | 17.02 | 48.23 | 0.00 |
| **recursive_infer (production ckpt)** | prod | **9.23** | **31.19** | **6.68** | **17.25** | 0.00 |

Recursive inference: carried-anchor stitching across overlapping windows (stride 10); only 53 chain starts (0.28 % of windows) use the learned anchor; [0,1] clip applied (monotone, sign-safe). Deterministic — no retraining, no new parameters.

## Findings

1. **Recursive inference is the largest improvement in the entire project, and it requires zero training.** −20 °C RMSE 17.86 → 6.68 %, MaxE 55.1 → 17.3 %; overall RMSE −27 %. This confirms the audit's protocol critique quantitatively: the published evaluation *manufactures* ~19k independent cold-start problems, and most of the reported −20 °C failure is an artifact of per-window re-anchoring rather than an observability wall.
2. **Recursion is not a free lunch — it exposes the delta-path rate undershoot.** At 40 °C recursive inference is *worse* than windowed (13.06 % vs 6.90 %): where per-window anchors were already good, carrying state accumulates the learned deltas' rate underestimation (ratio 0.72, Phase 1) as monotone drift over long chains. A rate-accurate delta path (the null model's is, ratio ≈ 1.0) plus periodic voltage feedback is exactly a recursive observer — which is the Phase 8 EKF baseline's territory. The honest conclusion: **carry state at cold, trust voltage re-anchoring when polarization is low** — a gating question, not solved here.
3. **anchor_last is promising but seed-sensitive** (2 of 3 seeds beat production on every metric; seed 42 is worse on RMSE). Mean ± std across seeds: RMSE 11.97 ± 1.00 %, −20 °C MaxE 41.9 ± 5.9 % vs 55.1 %. Under the project's own multi-seed rule this is "consistent MaxE improvement, inconsistent RMSE improvement" — usable with error bars, not as a headline. Deployment note: a window-end anchor implies 100 s anchor latency at stream start.
4. **anchor_pooled is unstable** (seed 123 diverges to 20 % RMSE) — rejected.
5. All variants keep PVR ≡ 0 (structural property is independent of anchor source, as designed).

## Interpretation for original contribution

The Hard-Coulomb layer's value survives (sign consistency + trainability, Phase 2), but the paper's accuracy story changes: the right deployment of the *existing* trained model is recursive stitched inference, which dissolves most of the "−20 °C anchor trap" the draft attributes to thermodynamic observability. The remaining open problem shifts to delta-path rate fidelity (envelope + sigmoid undershoot; see Phase 5's envelope-exceedance finding).
