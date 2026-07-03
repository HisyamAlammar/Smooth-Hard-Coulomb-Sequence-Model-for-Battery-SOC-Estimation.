# Phase 2 — Novelty-Critical Baselines Report

Date: 2026-07-03. New code: `baselines/null_ocv_coulomb.py`, `baselines/posthoc_clamp.py`, `baselines/build_comparison.py`. Same test splits, same windows, same metrics module as the learned models.

## Headline table (full-sequence, ALL temps; complete table in `results/baselines/baseline_comparison_table.csv`)

| Model | Params | Scen A RMSE % | Scen A MaxE % | Scen B RMSE % | Scen B MaxE % | PVR disch ε=0 | Δ-ratio disch (A/B) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla LSTM | 53,569 | 13.37 | 51.02 | 7.28 | 48.80 | 49.97 / 41.06 % | 20.09 / 16.30 |
| Vanilla + post-hoc clamp | 53,569 | **25.56** | 48.16 | **24.67** | 48.80 | 0.00 % | 0.57 / 0.60 |
| **Null OCV+Coulomb** (ocv25_qnom) | **0** | 13.24 | 86.97 | 7.56 | 94.73 | 0.00 % | 0.85 / 0.90 |
| Hard-Coulomb LSTM | 54,626 | 12.71 | 55.11 | 8.57 | 35.00 | 0.00 % | 0.72 / 0.38 |
| Hard-Coulomb TCN (sprint52) | 208,546 | 11.46 | 46.73 | 8.58 | 39.49 | 0.00 % | 0.85 / 0.44 |
| Null + **oracle anchor** (reference) | 0 | 0.03 | 0.28 | 0.02 | 0.36 | 0.00 % | 1.02 |

Null-model calibration variants: per-temperature OCV curves (`ocvT_*`) are *worse* than the 25 °C curve (A: 19.2 % vs 13.2 %) — cold-temperature HPPC rest segments give distorted OCV–SOC lookups (hysteresis / fewer valid rests). The reported primary null uses only train-temperature knowledge (Scenario-A-consistent).

## Answers to the two acceptance questions

**Q1. Does Hard-Coulomb LSTM beat the zero-parameter null model?**
*Mixed, and unfavorable in-distribution.*
- Scenario A (OOD): HC-LSTM RMSE beats the null by only **0.53 pp** (12.71 vs 13.24 %) and *loses on MAE* (9.87 vs 9.72 %). HC-TCN does better (11.46 %, −1.78 pp vs null). Where the learned models clearly win is **MaxE** (55.1 / 46.7 % vs 87.0 %): the learned anchor never emits the catastrophic anchors the raw OCV lookup emits under cold polarization, and the windowed [0,1] construction bounds excursions.
- Scenario B (in-distribution): the null model **beats both** Hard-Coulomb networks on RMSE (7.56 vs 8.57 / 8.58 %). Vanilla also beats them (7.28 %).
- The oracle-anchor null (RMSE 0.03 %, MaxE 0.28 %) shows pure Coulomb counting over a 100 s window is essentially exact once the anchor is known: **the entire estimation problem in this protocol is the anchor**, and the learned delta path competes against a solved problem.
- Rate fidelity: the null model tracks the true discharge rate at ratio 0.85–1.03; HC-LSTM at 0.72 (A) and 0.38 (B). The learned "Coulomb" path moves SOC *slower than physics* — plain Coulomb counting is more physically faithful than the trained Hard-Coulomb magnitude head.

**Q2. Does training-through-constraint beat post-hoc clamping?**
*Yes, decisively.* Clamping the trained vanilla's deltas with the identical envelope collapses accuracy (RMSE 25.6 / 24.7 %): vanilla's oscillating deltas mean-revert, and one-sided clamping converts the oscillation into monotone drift. Training through the constraint produces deltas compatible with the envelope. This is the strongest evidence *for* the contribution in this phase — the Hard-Coulomb layer is not reducible to a three-line output filter *given this vanilla backbone*. Caveat (Phase 3): the clamp collapse is partly a symptom of the vanilla baseline's oscillation pathology; a well-behaved backbone might clamp more gracefully.

## Honest implications for the manuscript (not applied yet)

1. "Hard-Coulomb improves RMSE" is **not supported** against the null model in-distribution, and only weakly (LSTM) supported OOD. The defensible claims are: (a) MaxE reduction vs the null anchor, (b) structural PVR, (c) training-through >> post-hoc.
2. Any claims table must include the null model; reviewers will run it otherwise.
3. The delta-ratio column should appear in every results table; PVR alone rewards frozen outputs.

## Deliverables

- `results/baselines/null_ocv_coulomb_results.json` (+ per-scenario prediction `.npz`)
- `results/baselines/posthoc_clamp_results.json` (+ `.npz`)
- `results/baselines/baseline_comparison_table.csv` (per-temperature rows included)
- `results/baselines/baseline_comparison_results.json`

Blockers: none. HC-TCN sprint52 checkpoints loaded cleanly and reproduce the draft's 11.46 % / 46.73 % Scenario-A numbers.
