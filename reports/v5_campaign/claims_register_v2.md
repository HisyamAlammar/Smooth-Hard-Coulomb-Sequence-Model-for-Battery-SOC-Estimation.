# Claims Register v2 — post-v5 campaign

Date: 2026-07-03. Supersedes `reports/claims_register.md` (v1, post-audit). Every manuscript claim re-adjudicated on v5c evidence (corrected labels, multi-seed, recursive policies, eta calibration, continuous EKFs). Statuses: **SUPPORTED** / **PARTIALLY SUPPORTED** / **UNSUPPORTED**. Δv1 column = change from v1.

Machine-readable twin: `claims_register_v2.json`.

| # | Claim | v5 evidence | Status | Δv1 | Source |
|---|---|---|---|---|---|
| 1 | HC enforces sign-consistency w.r.t. measured current (structural) | PVR ≡ 0 across all v5c seeds, policies, eta values, scenarios | **SUPPORTED** | unchanged | P2–P6 |
| 2 | PVR 0.00 % must be stated "by construction", not as result | unchanged; every v5 table footnotes it | **SUPPORTED** | unchanged | P2 |
| 3 | HC improves RMSE over baselines (general) | Now variant-specific: anchor_last/pooled beat vanilla AND null in both scenarios, all 5 seeds (A 9.99±1.09 vs 11.35/13.03; B 4.74±0.31 vs 6.20/7.53). Original HC-LSTM still loses in-distribution (10.63±0.60 vs 7.53). | **PARTIALLY SUPPORTED → SUPPORTED for anchor-redesigned variants** | upgraded | P2, P3 |
| 4 | HC beats null OCV+Coulomb | anchor_last: yes everywhere, every seed. Original HC: OOD only (1.98 pp, up from 0.53 on v4 labels — label correction strengthened the OOD case). | **SUPPORTED (anchor variants); PARTIALLY (original)** | upgraded | P2, P3 |
| 5 | HC beats post-hoc clamp | Clamp collapse reproduced on v5c: 25.0/30.1 % vs HC ≤10.9/10.6 | **SUPPORTED** | unchanged | P2 |
| 6 | −20 °C error is anchor-dominated | Oracle-anchor null: 0.03 % at −20 °C vs learned 15–17 %. NEW interventional proof: carrying anchor cuts −20 °C 16.75→11.11 with zero retraining; calibrated recursive reaches 3.59 %. | **SUPPORTED** (stronger: intervention, not just reference) | strengthened | P2, P4, P5 |
| 7 | Windowed evaluation reflects deployment | Still protocol-dependent, but now *characterized*: policy family + eta calibration map the frontier; calibrated-carried (4.43 %) ≠ windowed (11.05 %). Both protocols must be reported. | **UNSUPPORTED** (as stated); frontier now quantified | refined | P4, P5 |
| 8 | Labels are trustworthy | v4 labels carried ohmic-anchor bias + decimation conflicts; v5c corrects both. v4→v5c deltas: all models improve Scen A (−0.2…−1.8 pp); null MaxE 87→61. Residual: ohmic-only correction is a lower bound; polarization un-modeled. | **PARTIALLY SUPPORTED → largely resolved by v5c; residual limitation stated** | upgraded | P1, P2 |
| 9 | Edge deployment feasible (parameter-level) | Unchanged from v4 (54.6 k params / 218 KB / MCU-class MMAC); v5 adds nothing hardware-side | **PARTIALLY SUPPORTED** | unchanged | v4 P9 |
| 10 | PVR survives quantization (path-dependent) | Unchanged; single-scale-accumulator requirement stands | **PARTIALLY SUPPORTED** | unchanged | v4 P9 |
| 11 | Functional safety supported | Unchanged; only "safety-motivated architectural property" allowed | **UNSUPPORTED** | unchanged | v4 P7 |
| 12 | Sensor-fault robustness | Unchanged; characterized failure envelope | **UNSUPPORTED as robustness / SUPPORTED as characterization** | unchanged | v4 P7 |
| 13 | Vanilla baseline fair | Unchanged (implementation verified); v5c multi-seed adds σ=0.18–0.27 tightness → fairness reinforced | **SUPPORTED** | unchanged | P3 |
| **14** | **NEW: Anchor redesign (last/pooled) is necessary and sufficient to beat the null in-distribution** | Original HC fails Scen B systematically (5/5 seeds early-stop ~20 ep, 10.63±0.60); anchor_last/pooled 4.74/4.93, beating null 7.53 every seed | **SUPPORTED** | new | P3 |
| **15** | **NEW: Delta-path rate deficit is envelope mis-calibration, correctable at inference (eta*: rate ratio→1.00), not model capacity** | eta 1.5→2.0: ratio 0.751→1.002, recursive RMSE 11.78→4.43, −20 °C→3.59; equivalent optimum at eta=1.75·temp-aware gamma; retraining at eta* does NOT self-calibrate (head compensates: 17.6/14.9 %) | **SUPPORTED** (single seed/scenario; multi-seed confirmation pending → quote as such) | new | P5 |
| **16** | **NEW: Calibrated recursive HC beats continuous EKF baselines without their tuning sensitivity** | Best EKF 6.85 % (R=1e-2) vs calibrated HC 4.43 %; EKF spans 6.8–39.1 % over R decades; EKF PVR 4.9–39.6 % vs HC 0 % | **SUPPORTED** (with stated EKF caveat: literature-like params, not identified) | new | P6 |
| **17** | **NEW: Part of v4's catastrophic-MaxE narrative was label artifact** | Null Scen-A MaxE 87→61 % from label correction alone; corrected HC-vs-null MaxE gap remains large (36.5–46.5 vs 60.9) | **SUPPORTED** (must be disclosed in manuscript) | new | P1, P2 |

## Reviewer questions, v5 answers

1. **Better than OCV+Coulomb?** anchor_last: yes, everywhere, every seed (A −3.0 pp, B −2.8 pp vs null). Original HC: OOD only. Claim must name the variant.
2. **Better than post-hoc clamp?** Yes, decisive, unchanged.
3. **−20 °C anchor-driven?** Yes — now proven by intervention (carry/calibrate) not just oracle reference; calibrated recursive reaches 3.59 %.
4. **Vanilla fair?** Yes; multi-seed tightness (σ≤0.27) removes the last doubt.
5. **NEW — Why not "just use an EKF"?** Best-case EKF is 1.5× worse, requires a hidden trust knob spanning 6× RMSE, and violates monotonicity 5–40 % of steps.

## Manuscript language constraints (carried from v1, still binding)

- "By construction" for PVR; never in results tables as an achievement.
- No "suitable for edge MCUs" without hardware numbers.
- No "functional safety" beyond "safety-motivated".
- Sensor faults = characterized envelope.
- Single-seed numbers always labeled; means quoted ± std.
- eta* evidence quoted as single-checkpoint until multi-seed re-run.
