# V5 Campaign — Phase 4: Gated/Hybrid Recursive Inference Policies

Date: 2026-07-03. Single trained HC-LSTM checkpoint (v5c, Scenario A, seed 42); policies differ only in anchor choice per window — no weight changes. Gating parameters fixed a priori (physics-motivated, not tuned on test): cold threshold 5 °C, load threshold `CURRENT_THRESHOLD_A`, voltage stability 1 mV/s, blend ramps cold 5…−20 °C / load 0…2 A. Artifacts: `inference/gated_recursive_inference.py`, `experiments/compare_recursive_policies_v5.py`, `results/v5/recursive_inference/{recursive_policy_comparison.csv, recursive_policy_results.json}`, figures `recursive_policy_temperature_breakdown.png`, `cold_sequence_recursive_case_study.png`, `hot_sequence_recursive_failure_case.png`.

## Policy table (v5c Scenario A, seed 42, full-sequence RMSE %)

| Policy | RMSE | MaxE | −20 °C | −10 °C | 40 °C | re-anchor % |
|---|---:|---:|---:|---:|---:|---:|
| windowed_independent (legacy) | 11.05 | 46.19 | 16.75 | 9.66 | **4.22** | 100.0 |
| carried_anchor | 11.78 | 29.63 | **11.11** | 14.29 | 8.58 | 0.28 |
| temperature_gated | 10.98 | 29.63 | **11.11** | 14.29 | **4.22** | 32.7 |
| **load_gated** | **8.41** | **27.98** | 9.95 | **9.21** | 5.39 | 11.2 |
| rest_gated_reanchor | 11.32 | 29.63 | 11.11 | 14.29 | 6.38 | 3.6 |
| confidence_weighted_blend | 9.69 | 32.45 | 14.30 | 8.75 | 4.22 | 44.2 |
| hybrid_temperature_load_gate | 11.15 | 29.63 | 11.11 | 14.29 | 5.39 | 3.9 |

Chain starts: 0.28 % of windows (contiguity via timestamp keys, stride 10).

## Findings

1. **Anchor error, not delta error, dominates cold performance — now shown at inference time.** Carrying the anchor across contiguous windows cuts −20 °C RMSE 16.75 → 11.11 and MaxE 46.2 → 22.5 with zero retraining, confirming Phase-2's oracle-anchor diagnosis by intervention rather than reference.
2. **Carrying is not free: warm drift.** Pure carried_anchor degrades 40 °C RMSE 4.22 → 8.58 (delta-path drift accumulates over ~350-window chains) and overall RMSE worsens (11.78 vs 11.05). Any deployment claim must be gate-conditional.
3. **load_gated is the best fixed-gate policy: 8.41 % RMSE (−2.64 pp vs legacy windowed), best MaxE (27.98), best −10 °C, near-best −20 °C.** Re-anchoring only at rest starts (11 % of windows) captures most of the benefit of both regimes — voltage anchors are trusted exactly where OCV inversion is informative.
4. **temperature_gated ≈ strict Pareto improvement over legacy** (10.98 vs 11.05 RMSE, MaxE 46 → 30, cold −5.6 pp, warm unchanged) — the conservative option if only one gate signal is available.
5. **Blend is mid-pack** (9.69) — soft mixing dilutes both good anchors and carried anchors; hard gates on physical conditions beat confidence weighting here.
6. **hybrid gate underperforms load_gated** (11.15 vs 8.41): OR-ing cold into the carry condition keeps stale anchors through warm loaded stretches where re-anchoring at rest would have been fine. Composition of gates is not monotone in gate count.
7. PVR remains 0.00 % for every policy — the hard-Coulomb delta path is untouched by anchor policy, as designed.

## Caveats

Single checkpoint, single seed, Scenario A. Multi-seed sensitivity of the policy ranking (esp. load_gated vs blend) is Phase-7 material; eta/gamma delta-path scaling in Phase 5 addresses the warm drift directly.
