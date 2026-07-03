# V5 Campaign — Phase 3: Multi-Seed Robustness (v5c, seeds 1–5)

Date: 2026-07-03. 5 models × 2 scenarios × 5 seeds (1–5), identical sprint48 recipe; seed-42 headline runs from Phase 2 kept separate. Artifacts: `experiments/run_multiseed_v5.py`, `analysis/multiseed_summary.py`, `results/v5/multiseed/{runs_A.json, runs_B.json, seed_level_results.csv, multiseed_summary.csv, ranking_stability.json}`, figures `multiseed_rmse_boxplot.png`, `multiseed_maxe_boxplot.png`.

## Aggregate table (full-sequence RMSE %, mean ± std over seeds 1–5)

| Model | Scen A RMSE | A MaxE | Scen B RMSE | B MaxE |
|---|---:|---:|---:|---:|
| Vanilla LSTM | 11.35 ± 0.18 | 47.48 ± 1.54 | 6.20 ± 0.27 | 50.84 ± 1.24 |
| HC-LSTM (original anchor) | 10.87 ± 0.19 | 46.47 ± 0.76 | 10.63 ± 0.60 | 35.55 ± 0.62 |
| HC-TCN | 11.12 ± 1.71 | 45.44 ± 3.97 | 7.91 ± 1.16 | 40.35 ± 2.36 |
| HC anchor_pooled | 10.75 ± 0.98 | 40.49 ± 1.42 | 4.93 ± 0.84 | 33.98 ± 3.20 |
| **HC anchor_last** | **9.99 ± 1.09** | **36.50 ± 3.85** | **4.74 ± 0.31** | 34.49 ± 1.31 |

Null OCV+Coulomb reference (deterministic, seed-free): A 13.03, B 7.53.

## Findings

1. **Phase-2 anomaly resolved: the HC-LSTM Scenario-B regression is systematic, not seed noise.** All 5 seeds early-stop at 20–21 epochs with val loss 0.0051–0.0052 (anchor variants: 25–58 epochs, val 0.0014–0.0021), RMSE 10.63 ± 0.60 pp — seed 42's 9.87 was at the *good* end of the distribution. The original first-window anchor design genuinely fails in-distribution on corrected v5c labels. The v4→v5c "+1.31 pp regression" is real and must be reported as a property of that anchor design, not an optimization accident.
2. **Anchor redesign conclusion survives multi-seed.** anchor_last is best-mean in both scenarios (A 9.99, B 4.74); anchor_pooled second in B (4.93). Both beat vanilla (6.20) and null (7.53) in-distribution across every seed — Phase 2's headline claim holds with seed uncertainty attached.
3. **Ranking stability (winner by lowest RMSE, per seed):** Scenario A: anchor_last wins 4/5 (seed 5 → HC-LSTM). Scenario B: anchor family wins 5/5 (pooled seeds 1–2, last seeds 3–5). `winner_stable=false` at strict single-model level, but the *family-level* ranking (anchored HC > {vanilla, original HC, TCN} > null) is stable in B and directionally stable in A.
4. **Seed variance is model-dependent.** Vanilla and original HC-LSTM are tight (σ ≈ 0.2 pp, A); anchor variants and TCN spread more (σ 0.8–1.7 pp), TCN having one bad-seed outlier (A seed 4: 13.99). Claims about anchor variants need mean ± std, not single-seed points.
5. **MaxE story unchanged:** HC-family MaxE (A 36.5–46.5, B 34.0–40.4) stays well below vanilla (A 47.5, B 50.8) and null (A 60.9, B 94.7); anchor_last again lowest in A (36.50 ± 3.85).

## Consistency with seed-42 headline

anchor_last A 10.63 (s42) vs 9.99 ± 1.09; B 4.68 vs 4.74 ± 0.31; pooled B 4.63 vs 4.93 ± 0.84 — all inside 1σ. No Phase-2 conclusion is overturned; the HC-LSTM Scen-B caveat is upgraded from "flagged" to "confirmed systematic."

## Provenance

Every run records dataset_version=v5c, label_mode=ohmic_corrected, decimation_mode=mean_per_second, scenario, seed, checkpoint path, git commit, timestamps — in `runs_A.json` / `runs_B.json` and per-checkpoint payloads under `results/v5/headline_models/checkpoints/`.
