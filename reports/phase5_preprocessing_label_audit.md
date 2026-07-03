# Phase 5 — Preprocessing and Label Audit Report

Date: 2026-07-03. Artifacts: `analysis/preprocessing_audit.py`, `results/diagnostics/preprocessing_audit.json`, `results/diagnostics/soc_initial_stats_by_temperature.csv`, `results/diagnostics/segment_start_condition_report.csv`. Read-only audit over the raw 10 Hz drive-cycle CSVs (103 segments, 6 temperatures); labels **not** changed.

## Q1. Does 1 Hz decimation hide important current dynamics? — YES, materially.

`preprocessing_v4.to_strict_1hz_segments` keeps the **first sample** of each second; ground-truth SOC integrates the cycler's full-rate Capacity. Comparing kept sample vs intra-second mean current:

| Temp | mean \|I_first−I_mean\| (A) | p95 (A) | max (A) | sign conflict % | envelope exceeded % |
|---|---:|---:|---:|---:|---:|
| 0 °C | 0.31 | 1.22 | 12.7 | 2.35 | 9.42 |
| 10 °C | 0.54 | 2.09 | 18.0 | 4.13 | 13.20 |
| 25 °C | 0.37 | 1.42 | 13.4 | 2.69 | 10.18 |
| 40 °C | 0.23 | 0.99 | 16.7 | 1.82 | 7.36 |
| −10 °C | 0.28 | 1.14 | 11.9 | 2.11 | 8.36 |
| −20 °C | 0.20 | 0.83 | 5.2 | 1.78 | 6.97 |

Consequences:
1. **Routing-sign conflicts on ~2–4 % of seconds**: the kept sample and the true net intra-second current lie on *opposite* sides of the ±0.05 A dead-band. On those steps the Hard-Coulomb layer is forced to move SOC in the physically wrong direction (or freeze) *by construction*, while PVR (computed against the same kept sample) still reads 0.00 %. This is direct, quantified evidence for audit finding 3 at the data level — no sensor fault needed.
2. **Envelope violations on ~7–13 % of seconds**: true per-second |ΔSOC| (mean current) exceeds η·γ·|I_kept| with η=1.5, i.e. the label can legally move faster than the model's hard bound. This contributes to the delta-rate undershoot measured in Phase 1 (ratio 0.38–0.72). Caveat: the percentage counts all multi-sample seconds including low-current ones; magnitude-weighted impact is smaller but nonzero (p95 deviation ≈1–2 A ≙ ~1.4–2.8e-4 SOC/s against the envelope).
3. ~1 % of seconds are rest-labelled (|I_kept| ≤ 0.05 A) while current actually flowed.

**Recommendation:** decimate by *intra-second mean* (anti-aliased) instead of first-sample, or store the per-second integrated charge as the constraint input. Not changed now — it invalidates every published tensor; must be a versioned `v5` pipeline.

## Q2. Are SOC initial labels biased by loaded voltage? — YES, worst-case badly.

- **48 of 103 segments (33–56 % per temp) start under load** (|I| > 0.05 A; mean loaded start −0.65 A, extremes −9.5/+2.3 A). `soc_initial = OCV_lookup(V_first)` on a loaded start inherits the polarization sag.
- Ohmic-only correction (V0 − I0·R_int) shifts the initial label by **mean 2.44 %SOC, p95 10.9 %, max 33.5 %** on loaded starts (worst at 10 °C: mean 5.5 %). This is a *lower bound* — diffusion overpotential is uncorrected.
- Impact: the entire segment's SOC labels are offset by this amount (constant bias, since SOC_cc integrates deltas from soc_initial). Some of the reported "model error" is therefore label error, and it is temperature-correlated — it inflates the cold-temperature error narrative by an unknown but bounded amount.

## Q3. Is `cap_used = abs(cap − cap0)` safe? — Effectively yes.

Positive net-capacity excursions above segment start exist in 12/103 segments but are tiny: worst label flip **0.185 %SOC** (25 °C), −20 °C exactly 0. Document as a theoretical defect with negligible measured impact; no change needed.

## Q4. Change labels or document?

Implemented the *obvious, cheap* fix **behind a config flag, default off**: `config.LABEL_MODE = "legacy" | "ohmic_corrected"`, honored by `preprocessing_v4.engineer_features_v4` (segment-start OCV lookup uses V0 − I0·R_int(T) in corrected mode). Smoke-tested: corrected mode raises soc_initial for discharge-loaded starts; legacy default is byte-identical behavior. **All published tensors/models remain legacy**; switching requires regenerating `data/processed` and retraining, which is left as a deliberate, versioned decision (regenerate → retrain → compare, one experiment). The decimation fix (mean-per-second) is recommended for the same future `v5` pipeline but NOT implemented, as it touches the windowing contract.

## New facts feeding the claims register

- "PVR guarantee is relative to measured current" now has an in-dataset demonstration: 2–4 % of audited steps have decimation-corrupted routing signs while PVR reads 0.00 %.
- "Delta-rate undershoot" has a structural contributor: the envelope is unsatisfiable against the label on 7–13 % of steps.
- Cold-temperature MaxE numbers include a label-bias component (p95 ≈ 11 %SOC on loaded-start segments, ohmic-only lower bound).
