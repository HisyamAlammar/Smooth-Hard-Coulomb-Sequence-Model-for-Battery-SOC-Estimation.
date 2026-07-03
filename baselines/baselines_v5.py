"""
baselines_v5.py -- Phase 2: deterministic baselines on a v5 dataset variant.

  * null OCV+Coulomb (0 params): 25 degC OCV curve + Q(T) variants, plus
    oracle-anchor reference (same definitions as the v4 campaign)
  * vanilla + post-hoc Hard-Coulomb clamp (requires the v5 vanilla checkpoint)

Usage: python baselines/baselines_v5.py --variant v5c --scenarios A B --seed 42
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "src"), str(BASE_DIR / "analysis"),
          str(BASE_DIR / "baselines"), str(BASE_DIR / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import PHYS_MAX_V3, PHYS_MIN_V3, Q_NOMINAL  # noqa: E402
from null_ocv_coulomb import coulomb_trajectory  # noqa: E402
from posthoc_clamp import apply_hard_coulomb_clamp  # noqa: E402
from preprocessing_v4 import Q_ACTUAL_PER_TEMP, build_ocv_soc_lookup  # noqa: E402
from run_multiseed_v5 import build_model, forward, load_splits  # noqa: E402
from soc_metrics import evaluate_soc_predictions  # noqa: E402

OUT_DIR = BASE_DIR / "results" / "v5" / "baselines"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE_DIR).decode().strip()
    except Exception:
        return "unknown"


def prov(variant, scenario, meta, extra):
    return {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(), "dataset_version": variant,
            "label_mode": meta.get("label_mode"), "decimation_mode": meta.get("decimation_mode"),
            "scenario": f"scenario_{scenario}", "split": "test", **extra}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v5c")
    ap.add_argument("--scenarios", nargs="+", default=["A", "B"])
    ap.add_argument("--seed", type=int, default=42, help="which vanilla checkpoint to clamp")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ocv25, _ = build_ocv_soc_lookup("25degC")
    results = {}
    for scen in args.scenarios:
        data = load_splits(args.variant, scen)
        X, y_true, I, temps = data["X_test"], data["y_test"], data["I_test"], data["temp_labels"]
        v_proxy_t0 = (X[:, 0, 0] * (PHYS_MAX_V3[0] - PHYS_MIN_V3[0]) + PHYS_MIN_V3[0]).astype(np.float64)
        q_nom = np.full(len(temps), Q_NOMINAL)
        q_temp = np.array([Q_ACTUAL_PER_TEMP.get(str(t), Q_NOMINAL) for t in temps])

        scen_res = {}
        anchors = {
            "ocv25_qnom": (np.clip(ocv25(v_proxy_t0), 0, 1), q_nom),
            "ocv25_qtemp": (np.clip(ocv25(v_proxy_t0), 0, 1), q_temp),
            "oracle_qtemp": (y_true[:, 0].astype(np.float64), q_temp),
        }
        for name, (soc0, q) in anchors.items():
            y_pred = coulomb_trajectory(soc0, I.astype(np.float64), q).astype(np.float32)
            m = evaluate_soc_predictions(y_true, y_pred, I, temps)
            scen_res[f"null[{name}]"] = {"provenance": prov(args.variant, scen, data["meta"],
                                                            {"model": f"null[{name}]", "params": 0}),
                                         "metrics": m}
            print(f"  {args.variant} S{scen} null[{name:12s}] RMSE {m['regression']['rmse_full_pct']:8.4f}% "
                  f"| MaxE {m['regression']['maxe_full_pct']:8.4f}%")

        ckpt = BASE_DIR / "results" / "v5" / "headline_models" / "checkpoints" / \
            f"vanilla_lstm_{args.variant}_scenario_{scen}_seed{args.seed}.pt"
        if ckpt.exists():
            payload = torch.load(ckpt, map_location=device, weights_only=False)
            model = build_model("vanilla_lstm").to(device)
            model.load_state_dict(payload["model_state_dict"])
            model.eval()
            preds = []
            with torch.no_grad():
                for i in range(0, len(X), 1024):
                    xb = torch.from_numpy(X[i:i+1024]).to(device)
                    ib = torch.from_numpy(I[i:i+1024]).to(device)
                    preds.append(forward(model, "vanilla_lstm", xb, ib).squeeze(-1).cpu().numpy())
            y_van = np.concatenate(preds, 0)
            y_clamp = apply_hard_coulomb_clamp(y_van, I)
            m = evaluate_soc_predictions(y_true, y_clamp, I, temps)
            scen_res["vanilla+posthoc_clamp"] = {
                "provenance": prov(args.variant, scen, data["meta"],
                                   {"model": "vanilla+posthoc_clamp", "seed": args.seed,
                                    "checkpoint": str(ckpt.relative_to(BASE_DIR))}),
                "metrics": m}
            print(f"  {args.variant} S{scen} vanilla+clamp      RMSE {m['regression']['rmse_full_pct']:8.4f}% "
                  f"| MaxE {m['regression']['maxe_full_pct']:8.4f}%")
        else:
            scen_res["vanilla+posthoc_clamp"] = {"blocker": f"missing vanilla checkpoint {ckpt.name}"}
        results[f"scenario_{scen}"] = scen_res

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"deterministic_baselines_{args.variant}.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
