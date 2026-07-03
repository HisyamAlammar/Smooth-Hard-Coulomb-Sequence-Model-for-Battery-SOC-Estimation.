"""
compare_anchor_variants.py -- Phase 6: anchor redesign experiments.

Evidence gate (Phase 4): anchor error is ~96 % of RMSE; the production anchor
reads h[:, 0, :] (one timestep of context). Variants tested here, original
model untouched:

  HC_LSTM_anchor_first    : production model (existing checkpoint, no retrain)
  HC_LSTM_anchor_last     : anchor from h[:, -1, :]  (full causal window;
                            deployment implication: anchor latency = window)
  HC_LSTM_anchor_pooled   : anchor from mean-pooled hidden states
  HC_LSTM_recursive_infer : production checkpoint, carried-anchor stitched
                            inference across overlapping windows (stride 10);
                            only the FIRST window of each contiguous chain uses
                            the learned anchor -- deployment-realistic recursion

Retrained variants use 3 seeds; improvements are only claimed if consistent.
Checkpoints go to results/model_variants/checkpoints/ (never outputs/v7_final).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "src"), str(BASE_DIR / "analysis"), str(BASE_DIR / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import BATCH_SIZE, EPOCHS, LEARNING_RATE  # noqa: E402
from model_v5_coulomb import HardCoulombLSTM  # noqa: E402
from oracle_anchor import hc_cumulative_delta  # noqa: E402
from predict_utils import load_test_bundle, provenance  # noqa: E402
from preprocessing_v4 import PROFILE_KEY_SCALE  # noqa: E402
from soc_metrics import evaluate_soc_predictions  # noqa: E402
from sprint48_common import (  # noqa: E402
    PATIENCE,
    checkpoint_path,
    load_checkpoint,
    load_training_splits,
    make_loader,
    resolve_device,
    set_reproducibility,
)

DATA_DIRS = {
    "scenario_A": BASE_DIR / "data" / "processed" / "v4_scenario_A",
    "scenario_B": BASE_DIR / "data" / "processed" / "v4_scenario_B",
}
OUT_DIR = BASE_DIR / "results" / "model_variants"
CKPT_DIR = OUT_DIR / "checkpoints"
SEEDS = (42, 123, 2026)
STRIDE = 10


class HCAnchorVariant(HardCoulombLSTM):
    """HardCoulombLSTM with a configurable anchor context source."""

    def __init__(self, anchor_source: str = "first", **kwargs) -> None:
        super().__init__(**kwargs)
        if anchor_source not in ("first", "last", "pooled"):
            raise ValueError(anchor_source)
        self.anchor_source = anchor_source

    def forward(self, x, current_seq, return_delta: bool = False):
        h, _ = self.lstm(x)
        delta_logits = self.delta_head(h)
        if self.anchor_source == "first":
            ctx = h[:, 0, :]
        elif self.anchor_source == "last":
            ctx = h[:, -1, :]
        else:
            ctx = h.mean(dim=1)
        anchor_logit = self.anchor_head(ctx)
        soc_pred, delta = self.hard_constraint(delta_logits, current_seq, anchor_logit)
        return (soc_pred, delta) if return_delta else soc_pred


def train_variant(anchor_source: str, scenario: str, seed: int, device: torch.device) -> Path:
    """Same recipe as sprint48 (MSE/AdamW/cosine/clip/patience), isolated output."""
    ckpt = CKPT_DIR / f"hc_anchor_{anchor_source}_{scenario}_seed{seed}.pt"
    if ckpt.exists():
        return ckpt
    set_reproducibility(seed)
    splits = load_training_splits(DATA_DIRS[scenario], scenario)
    train_loader = make_loader(splits.X_train, splits.y_train, splits.I_train, BATCH_SIZE, True, seed, device)
    val_loader = make_loader(splits.X_val, splits.y_val, splits.I_val, BATCH_SIZE, False, seed, device)

    model = HCAnchorVariant(anchor_source=anchor_source).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val, best_state, patience = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb, ib in train_loader:
            xb, ib = xb.to(device), ib.to(device)
            yt = yb.unsqueeze(-1).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb, ib), yt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        model.eval()
        vl, nb = 0.0, 0
        with torch.no_grad():
            for xb, yb, ib in val_loader:
                xb, ib = xb.to(device), ib.to(device)
                yt = yb.unsqueeze(-1).to(device)
                vl += float(criterion(model(xb, ib), yt).item())
                nb += 1
        vl /= max(nb, 1)
        if vl < best_val:
            best_val, best_state, patience = vl, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
        if patience >= PATIENCE:
            break
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "anchor_source": anchor_source,
        "scenario": scenario,
        "seed": seed,
        "best_val_loss": best_val,
        "epochs_ran": epoch,
        "minutes": round((time.time() - t0) / 60.0, 2),
        "recipe": "sprint48-equivalent (MSE/AdamW/cosine/clip1.0/patience10)",
    }, ckpt)
    print(f"    trained {ckpt.name}: best val {best_val:.6f}, {epoch} epochs, {(time.time()-t0)/60:.1f} min")
    return ckpt


def predict_variant(ckpt: Path, bundle, device, batch_size=1024) -> np.ndarray:
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model = HCAnchorVariant(anchor_source=payload["anchor_source"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(bundle.X), batch_size):
            xb = torch.from_numpy(bundle.X[i : i + batch_size]).to(device)
            ib = torch.from_numpy(bundle.I[i : i + batch_size]).to(device)
            preds.append(model(xb, ib).squeeze(-1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def recursive_inference(scenario: str, device, batch_size=1024) -> Dict[str, object]:
    """Carried-anchor stitched inference on the PRODUCTION checkpoint."""
    bundle = load_test_bundle(scenario)
    if bundle.timestamp_keys is None:
        return {"blocker": "timestamp_key_test.npy missing; cannot stitch windows"}
    model, _ = load_checkpoint(checkpoint_path("hard_coulomb_lstm", scenario, latest=False), device)
    model.eval()

    n = len(bundle.X)
    cumulative = np.empty_like(bundle.y_true)
    anchor_model = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(bundle.X[i : i + batch_size]).to(device)
            ib = torch.from_numpy(bundle.I[i : i + batch_size]).to(device)
            cum = hc_cumulative_delta(model, xb, ib).cpu().numpy()
            y = model(xb, ib).squeeze(-1).cpu().numpy()
            cumulative[i : i + batch_size] = cum
            anchor_model[i : i + batch_size] = y[:, 0] - cum[:, 0]

    start_key = bundle.timestamp_keys[:, 0]
    order = np.argsort(start_key, kind="stable")
    y_rec = np.empty_like(bundle.y_true)
    chain_starts = 0
    prev_idx = -1
    for j in order:
        if prev_idx >= 0 and start_key[j] == start_key[prev_idx] + STRIDE and (
            start_key[j] // PROFILE_KEY_SCALE == start_key[prev_idx] // PROFILE_KEY_SCALE
        ):
            # carry the previous window's estimate at the overlapping timestep
            anchor = y_rec[prev_idx, STRIDE] - cumulative[j, 0]
        else:
            anchor = anchor_model[j]
            chain_starts += 1
        y_rec[j] = np.clip(anchor + cumulative[j], 0.0, 1.0)
        prev_idx = j

    metrics = evaluate_soc_predictions(bundle.y_true, y_rec, bundle.I, bundle.temp_labels)
    metrics["chain_starts"] = int(chain_starts)
    metrics["chain_start_pct"] = round(100.0 * chain_starts / n, 2)
    return metrics


def summarize_variant(name: str, scenario: str, metrics: Dict) -> Dict:
    reg = metrics["regression"]
    row = {
        "variant": name, "scenario": scenario,
        "rmse_pct": round(reg["rmse_full_pct"], 4),
        "mae_pct": round(reg["mae_full_pct"], 4),
        "maxe_pct": round(reg["maxe_full_pct"], 4),
        "pvr_discharge_eps0": round(metrics["pvr"]["discharge"]["by_epsilon"]["0"]["rate_pct"], 4),
    }
    for temp in ("n20degC", "n10degC"):
        pt = metrics.get("per_temperature", {}).get(temp)
        if pt:
            row[f"rmse_{temp}"] = round(pt["regression"]["rmse_full_pct"], 4)
            row[f"maxe_{temp}"] = round(pt["regression"]["maxe_full_pct"], 4)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Anchor variant comparison (Phase 6).")
    parser.add_argument("--scenario", default="scenario_A")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    scenario = args.scenario
    bundle = load_test_bundle(scenario)
    rows: List[Dict] = []
    detail: Dict[str, object] = {"provenance": provenance(scenario, None, {
        "experiment": "anchor_variants", "seeds": list(args.seeds),
    })}

    # production model (anchor_first), existing checkpoint
    from predict_utils import predict_checkpoint
    base = predict_checkpoint(scenario, "hard_coulomb_lstm", device)
    m = evaluate_soc_predictions(bundle.y_true, base["y_pred"], bundle.I, bundle.temp_labels)
    rows.append({**summarize_variant("HC_LSTM_anchor_first(prod)", scenario, m), "seed": "prod"})
    detail["HC_LSTM_anchor_first(prod)"] = m

    for source in ("last", "pooled"):
        for seed in args.seeds:
            ckpt = train_variant(source, scenario, seed, device)
            y = predict_variant(ckpt, bundle, device)
            m = evaluate_soc_predictions(bundle.y_true, y, bundle.I, bundle.temp_labels)
            rows.append({**summarize_variant(f"HC_LSTM_anchor_{source}", scenario, m), "seed": seed})
            detail[f"HC_LSTM_anchor_{source}_seed{seed}"] = m

    m_rec = recursive_inference(scenario, device)
    if "blocker" not in m_rec:
        rows.append({**summarize_variant("HC_LSTM_recursive_infer(prod)", scenario, m_rec), "seed": "prod"})
    detail["HC_LSTM_recursive_infer(prod)"] = m_rec

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "anchor_variant_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT_DIR / "anchor_variant_results.json").write_text(json.dumps(detail, indent=2, default=float))

    for r in rows:
        print(f"  {r['variant']:32s} seed={str(r['seed']):>5s} RMSE {r['rmse_pct']:8.4f}% | "
              f"MaxE {r['maxe_pct']:8.4f}% | n20 RMSE {r.get('rmse_n20degC','-')} | "
              f"n20 MaxE {r.get('maxe_n20degC','-')} | PVRd0 {r['pvr_discharge_eps0']}")
    print(f"Saved to {OUT_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
