# Phase 3 — Vanilla Baseline Fairness Diagnosis

Date: 2026-07-03. Artifacts: `analysis/diagnose_vanilla_baseline.py`, `results/diagnostics/vanilla_range_diagnostics.csv`, `results/diagnostics/vanilla_mechanism_probe.json`, `results/figures/vanilla_delta_histogram.png`, `results/figures/vanilla_prediction_examples.png`.

## 1. Is the vanilla implementation correct?

**Yes — no bug found.** Checklist audited in `src/sprint48_common.py` / training logs:

- Targets: SOC already in [0,1]; sigmoid output head matches; no target scaling to invert. ✓
- Loss: MSE on the full sequence (same supervision as Hard-Coulomb — fair). ✓
- Split/leakage: same v4 leakage-checked tensors as HC (timestamp-intersection asserts pass at preprocessing time). ✓
- Shapes: `assert_seq2seq_contract` enforces (N,100,5)/(N,100)/(N,100) alignment. ✓
- Training: converged (Scen A best val loss 0.0044 → 0.0038 by epoch 41; cosine LR decayed as configured; early stop functioned). ✓
- Inference: deterministic, eval mode, no dropout at test. ✓

## 2. Is the vanilla baseline unfairly weak?

**Its weakness is real but partly protocol-induced, and its "≈50 % PVR" headline needs reframing.**

Measured behavior (mechanism probe):
- Predicted per-step deltas have **lag-1 autocorrelation +0.84 / +0.80** — the trajectory is a *smooth wander*, not high-frequency noise. The dead-band analysis (Phase 1) already showed violations are not float dust (median 0.11 %SOC/step).
- Predicted deltas are essentially **uncorrelated with true deltas** (corr −0.04 / −0.00) and only weakly correlated with any input feature (max |corr| 0.20, V_proxy). The within-window movement is belief revision, not tracking.
- Per-window predicted range: mean 28.3 % (A) / 24.9 % (B) vs true 1.4 % / 0.9 % — ratio 20–28×.
- The example plots show the signature: a sharp transient in the first ~5 steps (the causal LSTM has almost no context at early window positions — exactly the observability problem the HC anchor has at t=0), then smooth drift toward a settled estimate.

Fairness assessment:
- The **causal seq2seq protocol** scores every window position, including near-zero-context early positions. Vanilla's last-step metrics are meaningfully better than full-sequence ones (Scen B: 6.22 % vs 7.28 % RMSE) because late positions have full context. Any per-step-supervised causal model will "wander" as evidence accumulates; a metric with zero dead-band then reports ≈50 % of discharge steps as violations.
- However, the wander is *also genuinely unphysical* as a BMS output signal, and the magnitude (20× true movement) is a real pathology worth reporting. It is what a plausibility monitor would flag, and what the Hard-Coulomb layer structurally removes.

## 3. Does Hard-Coulomb improve a fair baseline or suppress a defective one?

**Neither cleanly — it replaces vanilla's failure mode with its own.** Both models face the same fundamental limit (initial-state observability at window start). Vanilla expresses it as a t=0 transient plus wander; HC expresses it as a frozen anchor error carried through the window. HC's structural fix removes the wander (real improvement in trajectory consistency and MaxE vs the null anchor) but does **not** improve average accuracy against the null model in-distribution (Phase 2), and its delta path undershoots true rate (ratio 0.38–0.72). The honest comparison set is therefore: vanilla (wander pathology) vs HC (anchor pathology) vs null model (raw-anchor pathology) — no bug fixes required, but the manuscript must not present vanilla's ε=0 PVR as the sole safety comparison.

**No code changes made** (per protocol: fix only on clear bug; none found). Recommended reporting changes for later manuscript phase: report vanilla with last-step and ε-dead-band metrics alongside; add the wander mechanism explanation.
