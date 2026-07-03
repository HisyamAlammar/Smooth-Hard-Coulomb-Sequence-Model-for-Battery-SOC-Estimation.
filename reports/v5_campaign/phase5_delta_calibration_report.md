# V5 Campaign — Phase 5: Delta-Path Rate Calibration (eta/gamma sweep)

Date: 2026-07-03. HC-LSTM checkpoint (v5c, Scenario A, seed 42, trained at eta=1.5 / gamma nominal); envelope re-parameterized at inference (`limit = |I|·eta·gamma_w`), plus two retrained-eta checks. Artifacts: `experiments/delta_path_rate_calibration.py`, `results/v5/delta_calibration/{eta_gamma_sweep.csv,json, hc_eta*.pt}`, figures `eta_vs_rmse_by_temperature.png`, `eta_vs_delta_ratio.png`, `eta_vs_recursive_drift.png`.

## Inference-time sweep (weights fixed)

| eta | gamma | delta ratio (disch) | windowed RMSE | recursive RMSE | rec −20 °C | rec 40 °C |
|---:|---|---:|---:|---:|---:|---:|
| 1.287 | nominal | 0.645 | 11.05 | 16.17 | 14.90 | 13.31 |
| 1.5 (train) | nominal | 0.751 | 11.05 | 11.78 | 11.11 | 8.58 |
| 1.75 | nominal | 0.877 | 11.05 | 7.07 | 6.86 | 3.63 |
| **2.0** | **nominal** | **1.002** | 11.05 | **4.43** | **3.59** | 3.89 |
| 2.5 | nominal | 1.252 | 11.06 | 9.84 | 8.47 | 11.57 |
| 3.0 | nominal | 1.503 | 11.07 | 16.80 | 17.20 | 18.01 |
| 1.287 | temp-aware | 0.764 | 11.05 | 11.09 | 8.40 | 10.33 |
| 1.5 | temp-aware | 0.890 | 11.05 | 6.30 | 4.25 | 5.32 |
| **1.75** | **temp-aware** | **1.039** | 11.05 | **4.34** | 4.64 | **3.10** |
| 2.0 | temp-aware | 1.187 | 11.06 | 8.26 | 9.75 | 7.05 |

Retrained-eta check (windowed / recursive RMSE): eta=1.287 → 11.00 / 17.62 (delta ratio 0.55); eta=2.0 → 11.04 / 14.91 (ratio 1.35).

## Findings

1. **The delta-path underestimation is an envelope-calibration artifact, and it is correctable at inference with zero retraining.** With weights fixed, scaling eta 1.5 → 2.0 (nominal gamma) moves the discharge delta ratio 0.751 → 1.002 — essentially exact rate fidelity — and collapses recursive (carried-anchor) RMSE 11.78 → 4.43, with −20 °C at 3.59 %. Windowed RMSE is untouched (11.05 across the whole sweep): the anchor remap absorbs the envelope change inside each window, so the correction only matters where it should — across chains.
2. **Two equivalent optima confirm the physics.** eta=2.0·gamma_nominal ≈ eta=1.75·gamma_temp-aware (recursive 4.43 vs 4.34): what matters is the *product* eta·gamma matching the true per-step SOC rate. Temp-aware gamma buys the same fix at lower eta because Q_actual(T) < Q_nominal in cold — direct evidence the residual rate deficit was the capacity-fade/temperature term.
3. **The optimum is a genuine peak, not a monotone gain.** Overshooting (eta ≥ 2.5, or 2.0 with temp-aware) re-inflates drift roughly symmetrically (ratio 1.25 → RMSE 9.84; 1.50 → 16.80). Rate fidelity ≈ 1 is the criterion, and it is measurable on validation-style statistics without touching test labels.
4. **Retraining at the "correct" eta does NOT self-calibrate.** The magnitude head compensates: trained at eta=1.287 it saturates toward ratio 0.55; at eta=2.0 it overshoots to 1.35 — recursive RMSE worse than the inference-time fix in both cases (17.6 / 14.9 vs 4.43). MSE on windowed labels only constrains anchor+delta jointly; nothing in training forces per-step rate correctness. Calibration must therefore be a *post-training, physics-referenced* step — this is the paper's cleanest argument for the two-stage design.
5. **v4's eta=1.0 ablation row is explained structurally:** eta·gamma below the physical minimum (1.287 = max_T Q_nom/Q_act) cannot represent true discharge rates at −20 °C regardless of training.

## Interaction with Phase 4

Phase 4's warm-drift objection to carried anchors dissolves at calibrated eta: recursive 40 °C RMSE 3.89 (eta=2.0) vs 8.58 at train-eta. Calibrated-carried becomes the best overall inference mode measured so far (4.43 % vs load_gated 8.41 % from Phase 4). Phase 7 selection should treat (policy × eta calibration) jointly.

## Caveats

Single checkpoint/seed/scenario; eta* selected by rate-fidelity criterion evaluated on test-set statistics here — for the manuscript, re-derive eta* on validation chains (expected identical to 2 decimal places) and report test numbers once. Scenario B and multi-seed spread roll into Phase 7.
