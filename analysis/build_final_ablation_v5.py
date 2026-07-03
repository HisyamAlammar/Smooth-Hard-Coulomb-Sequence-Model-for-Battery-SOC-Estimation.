"""
build_final_ablation_v5.py -- Phase 7: assemble the full ablation matrix from
every campaign result file (v4 frozen manifest + v5 phases 2-6).

Axes recorded per row: dataset_version, label_mode, decimation_mode, model,
anchor_strategy, inference_policy, eta/gamma, seed(s), baseline_type.

Outputs:
  results/v5/final_ablation_matrix.{csv,json}
  results/v5/final_v5_model_comparison.{csv,json}   (v5c primary table,
      multiseed mean±std where available, single-seed rows labeled)
"""

from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
V5 = BASE_DIR / "results" / "v5"


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def add(rows, **kw):
    base = {"dataset_version": "", "label_mode": "", "decimation_mode": "", "scenario": "",
            "model": "", "anchor_strategy": "", "inference_policy": "windowed", "eta": 1.5,
            "gamma_mode": "nominal", "seeds": "", "baseline_type": "", "rmse_pct": "",
            "rmse_std": "", "maxe_pct": "", "rmse_n20": "", "rmse_40": "",
            "pvr_disch_eps0": "", "source": ""}
    base.update(kw)
    rows.append(base)


def main() -> None:
    rows = []

    # ---- v4 frozen reference rows
    v4 = load(V5 / "legacy_freeze_manifest.json")["key_v4_metrics"]
    for scen, models in v4.items():
        for name, m in models.items():
            add(rows, dataset_version="v4_legacy", label_mode="legacy", decimation_mode="first_sample",
                scenario=scen, model=name, seeds="42(v4)", baseline_type="frozen_v4",
                rmse_pct=m.get("rmse_pct"), maxe_pct=m.get("maxe_pct", ""),
                rmse_n20=m.get("n20_rmse_pct", ""), pvr_disch_eps0=m.get("pvr_disch_eps0", ""),
                source="legacy_freeze_manifest.json")

    # ---- v5 multiseed aggregates
    ms_rows = []
    p = V5 / "multiseed" / "seed_level_results.csv"
    if p.exists():
        with p.open(encoding="utf-8") as fh:
            ms_rows = list(csv.DictReader(fh))
        groups = {}
        for r in ms_rows:
            groups.setdefault((r["scenario"], r["model"]), []).append(r)
        for (scen, model), rs in sorted(groups.items()):
            rmse = np.array([float(r["rmse_pct"]) for r in rs])
            maxe = np.array([float(r["maxe_pct"]) for r in rs])
            n20 = [float(r["rmse_n20degC"]) for r in rs if r.get("rmse_n20degC")]
            f40 = [float(r["rmse_40degC"]) for r in rs if r.get("rmse_40degC")]
            anchor = {"hc_anchor_last": "last", "hc_anchor_pooled": "pooled"}.get(model,
                     "first" if "coulomb" in model else "n/a")
            add(rows, dataset_version="v5c", label_mode="ohmic_corrected",
                decimation_mode="mean_per_second", scenario=scen, model=model,
                anchor_strategy=anchor, seeds="1-5",
                rmse_pct=round(float(rmse.mean()), 4), rmse_std=round(float(rmse.std(ddof=1)), 4),
                maxe_pct=round(float(maxe.mean()), 4),
                rmse_n20=round(float(np.mean(n20)), 4) if n20 else "",
                rmse_40=round(float(np.mean(f40)), 4) if f40 else "",
                pvr_disch_eps0=round(float(np.mean([float(r["pvr_disch_eps0"]) for r in rs])), 4),
                source="multiseed/seed_level_results.csv")

    # ---- v5 deterministic baselines
    det = load(V5 / "baselines" / "deterministic_baselines_v5c.json")
    if det:
        for scen, models in det.items():
            for name, entry in models.items():
                if "metrics" not in entry:
                    continue
                m = entry["metrics"]
                reg = m["regression"]
                pt = m.get("per_temperature", {})
                add(rows, dataset_version="v5c", label_mode="ohmic_corrected",
                    decimation_mode="mean_per_second", scenario=scen, model=name,
                    baseline_type="deterministic", seeds="n/a" if name.startswith("null") else "42",
                    rmse_pct=round(reg["rmse_full_pct"], 4), maxe_pct=round(reg["maxe_full_pct"], 4),
                    rmse_n20=round(pt.get("n20degC", {}).get("regression", {}).get("rmse_full_pct", np.nan), 4)
                    if "n20degC" in pt else "",
                    rmse_40=round(pt.get("40degC", {}).get("regression", {}).get("rmse_full_pct", np.nan), 4)
                    if "40degC" in pt else "",
                    pvr_disch_eps0=round(m["pvr"]["discharge"]["by_epsilon"]["0"]["rate_pct"], 4),
                    source="baselines/deterministic_baselines_v5c.json")

    # ---- recursive policies (phase 4)
    rec = load(V5 / "recursive_inference" / "recursive_policy_results.json")
    if rec:
        prov = rec["provenance"]
        for policy, m in rec["results"].items():
            reg = m["regression"]
            pt = m.get("per_temperature", {})
            add(rows, dataset_version=prov["dataset_version"], label_mode=prov.get("label_mode", ""),
                decimation_mode=prov.get("decimation_mode", ""), scenario=prov["scenario"],
                model="hard_coulomb_lstm", anchor_strategy="first", inference_policy=policy,
                seeds=str(prov.get("seed")),
                rmse_pct=round(reg["rmse_full_pct"], 4), maxe_pct=round(reg["maxe_full_pct"], 4),
                rmse_n20=round(pt.get("n20degC", {}).get("regression", {}).get("rmse_full_pct", np.nan), 4)
                if "n20degC" in pt else "",
                rmse_40=round(pt.get("40degC", {}).get("regression", {}).get("rmse_full_pct", np.nan), 4)
                if "40degC" in pt else "",
                pvr_disch_eps0=round(m["pvr"]["discharge"]["by_epsilon"]["0"]["rate_pct"], 4),
                source="recursive_inference/recursive_policy_results.json")

    # ---- eta/gamma calibration (phase 5)
    cal = load(V5 / "delta_calibration" / "eta_gamma_sweep.json")
    if cal:
        for r in cal["rows"]:
            add(rows, dataset_version="v5c", label_mode="ohmic_corrected",
                decimation_mode="mean_per_second", scenario=cal["provenance"]["scenario"],
                model="hard_coulomb_lstm", anchor_strategy="first",
                inference_policy=f"{r['mode']}", eta=r["eta"], gamma_mode=r["gamma_mode"],
                seeds=str(cal["provenance"]["seed"]),
                rmse_pct=r.get("win_rmse_pct", ""), maxe_pct=r.get("win_maxe_pct", ""),
                rmse_n20=r.get("rec_rmse_n20", ""), rmse_40=r.get("rec_rmse_40", ""),
                source="delta_calibration/eta_gamma_sweep.json")

    # ---- continuous EKFs (phase 6)
    ekf = load(V5 / "ekf_ecm" / "continuous_ekf_results.json")
    if ekf:
        for name, m in ekf["results"].items():
            reg = m["regression"]
            pt = m.get("per_temperature", {})
            add(rows, dataset_version="v5c", label_mode="ohmic_corrected",
                decimation_mode="mean_per_second", scenario=ekf["provenance"]["scenario"],
                model=name, baseline_type="classical_recursive",
                inference_policy="continuous" if "cont" in name else "windowed",
                seeds="n/a",
                rmse_pct=round(reg["rmse_full_pct"], 4), maxe_pct=round(reg["maxe_full_pct"], 4),
                rmse_n20=round(pt.get("n20degC", {}).get("regression", {}).get("rmse_full_pct", np.nan), 4)
                if "n20degC" in pt else "",
                rmse_40=round(pt.get("40degC", {}).get("regression", {}).get("rmse_full_pct", np.nan), 4)
                if "40degC" in pt else "",
                pvr_disch_eps0=round(m["pvr"]["discharge"]["by_epsilon"]["0"]["rate_pct"], 4),
                source="ekf_ecm/continuous_ekf_results.json")

    stamp = {"generated": datetime.datetime.now().isoformat(timespec="seconds"), "rows": rows}
    (V5 / "final_ablation_matrix.json").write_text(json.dumps(stamp, indent=2, default=float))
    with (V5 / "final_ablation_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # primary v5 comparison = v5c rows only
    v5_rows = [r for r in rows if r["dataset_version"] == "v5c" and r["inference_policy"]
               in ("windowed", "windowed_independent", "carried_anchor",
                   "hybrid_temperature_load_gate", "continuous")]
    (V5 / "final_v5_model_comparison.json").write_text(json.dumps(
        {"generated": stamp["generated"], "rows": v5_rows}, indent=2, default=float))
    with (V5 / "final_v5_model_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(v5_rows)
    print(f"ablation rows: {len(rows)}; v5 primary rows: {len(v5_rows)}")


if __name__ == "__main__":
    main()
