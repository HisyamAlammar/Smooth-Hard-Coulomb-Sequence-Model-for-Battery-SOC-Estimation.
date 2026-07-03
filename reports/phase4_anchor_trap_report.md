# Phase 4 — Anchor Trap Diagnostics Report

Date: 2026-07-03. Artifacts: `experiments/oracle_anchor.py`, `analysis/anchor_error_analysis.py`, `results/diagnostics/oracle_anchor_results.json`, `results/diagnostics/anchor_error_by_temperature.csv`, `results/diagnostics/anchor_error_analysis.json`, `results/figures/anchor_error_by_temperature.png`, `results/figures/anchor_error_vs_current_start.png`.

## Q1. How much error remains when the anchor is perfect?

Almost none. Oracle anchor (true SOC at t=0) with the **unchanged learned delta path**:

| Scenario | Condition | Normal RMSE % | Oracle RMSE % | Normal MaxE % | Oracle MaxE % |
|---|---|---:|---:|---:|---:|
| A | ALL | 12.71 | **0.54** | 55.11 | **5.05** |
| A | −20 °C | 17.86 | **0.50** | 55.11 | **4.58** |
| A | −10 °C | 11.91 | 0.55 | 52.05 | 5.05 |
| A | 40 °C | 6.90 | 0.56 | 24.43 | 4.24 |
| B | ALL | 8.57 | **0.54** | 35.00 | 4.90 |
| B | −20 °C | 8.67 | 0.29 | 35.00 | 3.01 |

Anchor error accounts for **~96 % of RMSE** in both scenarios (`anchor_share_of_rmse_pct` ≈ 95.8/93.7). Context: the *true-Coulomb* oracle null (Phase 2) reaches 0.03 % RMSE, so of the remaining 0.54 %, most is the learned delta path's rate undershoot — real but second-order.

## Q2. Is −20 °C failure anchor error or delta-path error?

**Anchor error, conclusively.** With a perfect anchor the −20 °C windows are the *easiest* in Scenario B (0.29 % RMSE) and indistinguishable from warm temps in Scenario A (0.50 %). The delta path does not degrade at cold temperatures; the anchor does: raw anchor-head MAE rises monotonically 5.0 % (40 °C) → 10.3 % (−10 °C) → 15.8 % (−20 °C) in Scenario A. The manuscript's "thermodynamic observability limit" framing must be narrowed to "**anchor** observability limit under single-sample cold-start"; nothing in the delta mechanism is limited.

## Q3. Does the current anchor design create a cold-start bottleneck?

Yes, and it is a design choice, not physics:
- Mechanism confirmed at `src/model_v5_coulomb.py:127`: `anchor_logit = anchor_head(h[:, 0, :])` — the LSTM hidden state after **one timestep**. The anchor sees a single (V_proxy, I, T, dVp/dt, dI/dt) sample; it cannot exploit the other 99 causal steps of the same window that the delta head already consumes.
- Scenario A: rest-start windows have lower anchor error than loaded starts (MAE 6.23 % vs 10.64 %) — consistent with the polarization narrative.
- **Complication (reported honestly):** Scenario B shows the *reverse* (rest-start 9.39 % vs load-start 5.09 %). Rest-start windows in B cluster at profile boundaries/pauses (often SOC extremes and flat-OCV regions), so "rest ⇒ good anchor" is not unconditional. The Phase 6 gating idea must not assume rest-start windows are always easy.
- Anchor error vs |I(t=0)| (figure): error is substantial in every current bin; load magnitude alone does not explain it.

## Model change decision

Evidence supports Phase 6 variants (anchor from later/pooled hidden states — the information is causally available at window end; deployment implication is a 100 s anchor latency, to be stated). Main model left untouched, per protocol.
