# V5 Campaign — Phase 0: Legacy Freeze Manifest

Date: 2026-07-03. Machine-readable twin: `results/v5/legacy_freeze_manifest.json`.

## Freeze status

- Git tree was clean; the entire v4 audit campaign is committed at **`3c4ba9b`**, now tagged **`audit-campaign-v4-complete`**.
- All required legacy deliverables verified present: `reports/claims_register.md`, `reports/final_code_audit_fix_report.md`, `results/final_model_comparison.{csv,json}`, 10 phase reports (`reports/phase0–9_*.md`), production checkpoints (`outputs/v7_final/*.pt`, sprint52 TCN, variant checkpoints), v4 tensors + `metadata_v4.json`.
- Reproduction check: metric sanity suite 7/7 PASS today; bit-identical legacy metric reproduction was verified during the v4 campaign (its Phase 1).

## Versioning decisions for v5

1. **Legacy results stay in place** (`results/`, `outputs/`); they are NOT moved into `results/v4_legacy/` — moving would break every path recorded in the v4 reports and provenance blocks. The manifest JSON is the authoritative v4 index instead. All new work writes exclusively under `results/v5/`, `reports/v5_campaign/`, `data/processed/v5*`, and new checkpoints under `results/v5/**/checkpoints/`.
2. Dataset provenance keys every v5 result must carry: `dataset_version`, `label_mode`, `decimation_mode`, `scenario`, `checkpoint`, `seed`, `timestamp`, `git_commit`, config values (threshold, epsilons, eta/gamma).
3. Comparison anchor: the `key_v4_metrics` block in the JSON manifest is the frozen v4 reference for every v4-vs-v5 delta reported later in this campaign.

## Acceptance criteria

- [x] Legacy v4 evidence preserved and indexed
- [x] v4/v5 separation defined (paths + provenance schema)
- [x] Nothing overwritten
- [x] Git tag created: `audit-campaign-v4-complete` → `3c4ba9b`
