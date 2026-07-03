# V5 Campaign — Phase 1: Dataset v5 Report

Date: 2026-07-03. Artifacts: `datasets/build_v5.py`, extended `src/preprocessing_v4.py` (+`src/config.py` flags), `results/v5/dataset_variant_comparison.{csv,json}`, `results/v5/figures/soc_initial_bias_by_temperature.png`, `results/v5/figures/routing_conflict_by_decimation_mode.png`.

## What changed in code

- `config.py`: `DATASET_VERSION`, `DECIMATION_MODE`, `DATASET_VARIANTS` map (v4_legacy / v5a / v5b / v5c). Defaults preserve legacy behavior everywhere.
- `preprocessing_v4.py`:
  - `to_strict_1hz_segments(..., decimation_mode=)` — `first_sample` (legacy), `mean_per_second` (V/I/T intra-second mean, Capacity last-sample), `integrated_current_per_second` (only current averaged). Smoke-tested: identical segmentation, only signal values change.
  - `run_pipeline_v4(..., label_mode=, decimation_mode=, variant_name=)` — routes output to `data/processed/<variant>_scenario_<X>`; legacy call sites unchanged.
- Split logic, window=100, stride=10, features, physics scaling: **unchanged** (verified: identical segment count 103 and identical window counts per split in every variant; timestamp-leakage checks pass with 0 overlaps in all 8 builds).

## Variant comparison (both scenarios; full table in CSV)

| Variant | Labels | Decimation | Windows (A: tr/va/te) | Label shift vs v4, Scen A (mean / max %SOC) | Scen B (mean / max) |
|---|---|---|---|---|---|
| v4_legacy | legacy | first_sample | 15,641/9,084/18,914 | 0 / 0 | 0 / 0 |
| v5a | ohmic | first_sample | identical | 3.53 / **93.58** | 1.09 / 38.05 |
| v5b | legacy | mean_per_second | identical | 1.09 / 12.55 | 0.81 / 12.48 |
| **v5c (final)** | ohmic | mean_per_second | identical | 1.35 / 16.24 | 0.68 / 16.24 |

Notes:
- The extreme v5a max shift (93.6 %SOC) is a real correction, not a bug: a −20 °C segment starts at ≈−9.5 A where R_int = 0.11 Ω ⇒ ≈1.04 V ohmic sag; the legacy OCV lookup read a near-empty cell for a near-full one. Under mean decimation (v5c) the start-second current spike is averaged, so the same correction is smaller (16.2 %) — the two fixes interact.
- Decimation defect rates: `first_sample` measured at 2.4 % routing-sign conflicts and 9.2 % envelope-unsatisfiable seconds (mean over temperatures, v4 audit); `mean_per_second` eliminates both **by construction** (the kept current *is* the intra-second mean). Figure: `routing_conflict_by_decimation_mode.png`.
- Ohmic label correction removes the quantified ohmic share of loaded-start `soc_initial` bias (mean 2.4 %SOC, p95 10.9 %, worst 33.5 % on 48/103 segments); the diffusion-overpotential share remains uncorrected and unquantified — stated as residual label uncertainty. Figure: `soc_initial_bias_by_temperature.png`.

## Decisions

- **`v5c` is the final v5 dataset** (both corrections). `v5a`/`v5b` isolate each correction for the Phase 2 attribution if needed.
- `rest_validated` label mode NOT implemented: only 55/103 segments start at rest; a rest-only anchor policy would need cross-segment SOC carry-over — a pipeline redesign, out of scope. Documented as future work.
- v4 tensors untouched and still regenerable (`run_pipeline_v4("A")` default path unchanged).

## Acceptance criteria

- [x] v4 loadable/regenerable; [x] v5 variants separated by path + metadata (`dataset_version`, `label_mode`, `decimation_mode` recorded in each `metadata_v4.json`); [x] v5c defined; [x] changes auditable (this report + manifest); [x] contamination/conflict reduction quantified above.
