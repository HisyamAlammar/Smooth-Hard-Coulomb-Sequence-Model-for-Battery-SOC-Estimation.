"""
oracle_anchor.py -- Phase 4: how much error remains when the anchor is perfect?

Keeps the trained Hard-Coulomb delta path EXACTLY as learned (same LSTM, same
delta head, same routing/envelope) but replaces the anchor with the true SOC
at t=0. The gap between normal and oracle inference isolates anchor error from
delta-path error, per scenario and per temperature.

Reference points from Phase 2:
  * null model with oracle anchor (true Coulomb deltas): RMSE ~0.03 %
  * normal HC-LSTM: Scen A RMSE 12.71 %, -20 degC last-step RMSE 17.63 %
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "src"), str(BASE_DIR / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

from predict_utils import load_test_bundle, provenance  # noqa: E402
from soc_metrics import evaluate_soc_predictions  # noqa: E402
from sprint48_common import checkpoint_path, load_checkpoint, resolve_device  # noqa: E402

OUT_DIR = BASE_DIR / "results" / "diagnostics"


def hc_cumulative_delta(model: torch.nn.Module, X: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
    """Reproduce the constraint's routed/bounded deltas; return cumsum (B, T)."""
    hc = model.hard_constraint
    h, _ = model.lstm(X)
    delta_logits = model.delta_head(h)
    I3 = I.unsqueeze(-1)
    limit = I3.abs() * hc.gamma_factor
    mag = torch.sigmoid(delta_logits)
    zero = torch.zeros_like(delta_logits)
    delta = torch.where(I3 < -hc.threshold, -limit * mag, zero)
    delta = torch.where(I3 > hc.threshold, limit * mag, delta)
    return torch.cumsum(delta, dim=1).squeeze(-1)


def run_scenario(scenario: str, batch_size: int, device: torch.device) -> Dict[str, object]:
    bundle = load_test_bundle(scenario)
    ckpt = checkpoint_path("hard_coulomb_lstm", scenario, latest=False)
    model, _ = load_checkpoint(ckpt, device)
    model.eval()

    oracle_preds, normal_preds = [], []
    with torch.no_grad():
        for i in range(0, len(bundle.X), batch_size):
            xb = torch.from_numpy(bundle.X[i : i + batch_size]).to(device)
            ib = torch.from_numpy(bundle.I[i : i + batch_size]).to(device)
            cumulative = hc_cumulative_delta(model, xb, ib)
            soc0 = torch.from_numpy(bundle.y_true[i : i + batch_size, 0]).to(device)
            oracle_preds.append((soc0[:, None] + cumulative).cpu().numpy())
            normal_preds.append(model(xb, ib).squeeze(-1).cpu().numpy())

    y_oracle = np.concatenate(oracle_preds, axis=0)
    y_normal = np.concatenate(normal_preds, axis=0)

    m_oracle = evaluate_soc_predictions(bundle.y_true, y_oracle, bundle.I, bundle.temp_labels)
    m_normal = evaluate_soc_predictions(bundle.y_true, y_normal, bundle.I, bundle.temp_labels)

    ro, rn = m_oracle["regression"], m_normal["regression"]
    print(f"\n  {scenario}  normal HC : RMSE {rn['rmse_full_pct']:7.4f}% | MAE {rn['mae_full_pct']:7.4f}% | MaxE {rn['maxe_full_pct']:7.4f}%")
    print(f"  {scenario}  oracle HC : RMSE {ro['rmse_full_pct']:7.4f}% | MAE {ro['mae_full_pct']:7.4f}% | MaxE {ro['maxe_full_pct']:7.4f}%")
    for temp, m in m_oracle.get("per_temperature", {}).items():
        mn = m_normal["per_temperature"][temp]["regression"]
        print(f"      {temp:>8s}: oracle RMSE {m['regression']['rmse_full_pct']:7.4f}% "
              f"(normal {mn['rmse_full_pct']:7.4f}%) | oracle MaxE {m['regression']['maxe_full_pct']:7.4f}% "
              f"(normal {mn['maxe_full_pct']:7.4f}%)")

    return {
        "provenance": provenance(scenario, str(ckpt.relative_to(BASE_DIR)), {
            "experiment": "oracle_anchor",
            "definition": "anchor := y_true[:,0]; learned delta path unchanged",
        }),
        "oracle": m_oracle,
        "normal": m_normal,
        "anchor_share_of_rmse_pct": round(100.0 * (1.0 - ro["rmse_full_pct"] / rn["rmse_full_pct"]), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle-anchor isolation experiment.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", default=str(OUT_DIR / "oracle_anchor_results.json"))
    args = parser.parse_args()

    device = resolve_device(args.device)
    results = {s: run_scenario(s, args.batch_size, device) for s in ("scenario_A", "scenario_B")}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
