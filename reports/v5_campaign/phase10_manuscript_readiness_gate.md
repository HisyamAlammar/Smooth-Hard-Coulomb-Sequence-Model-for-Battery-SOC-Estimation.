# V5 Campaign — Phase 10: Manuscript Readiness Gate

Date: 2026-07-03. Automated gate: 52 checks, **52/52 PASS**. Machine record: `phase10_readiness_gate.json`.

## Gate groups

| Group | Checks | Result |
|---|---:|---|
| G1 Artifact existence (14 result files + 12 figures + 11 reports) | 37 | PASS |
| G2 Number traceability (9 headline numbers re-read from source files) | 9 | PASS |
| G3 Provenance completeness (git commit, dataset_version, seed, checkpoint/assumptions in every P4–P6 JSON) | 3 | PASS |
| G4 PVR structural invariant (0.00 % at ε=0 across all HC rows, both scenarios, seeds 1–5) | 1 | PASS |
| G5 Claims register integrity (17 claims, ≥6 language constraints, valid JSON) | 2 | PASS |

## Verified headline numbers (source ↔ report)

- anchor_last windowed: A 9.993 ± 1.090, B 4.743 ± 0.314 (`multiseed_summary.csv`)
- original HC Scen-B failure: 10.629 ± 0.598 (`multiseed_summary.csv`)
- eta*=2.0 calibrated recursive: RMSE 4.432, −20 °C 3.594, delta ratio 1.002 (`eta_gamma_sweep.json`)
- best EKF: 6.849 % RMSE, 5.48 % PVR (`continuous_ekf_results.json`)
- load_gated policy: 8.409 % (`recursive_policy_results.json`)

## Open items (non-blocking, tracked in claims register constraints)

1. eta* multi-seed + Scenario-B confirmation (Claim 15 quoted single-checkpoint until done).
2. eta* re-derivation on validation chains for manuscript hygiene.
3. Optional: identified-parameter ECM for the EKF fairness footnote.
4. Git tag `v5-campaign-complete` after commit.

## Verdict

**READY for manuscript rewrite** per Phase 9 brief. Chain closed: dataset (P1) → models (P2–P3) → inference (P4–P5) → baselines (P6) → matrix/selection (P7) → claims (P8) → report (P9) → gate (P10).
