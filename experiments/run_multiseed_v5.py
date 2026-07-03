"""
run_multiseed_v5.py -- Phases 2-3: train/evaluate headline models on any
dataset variant with full seed control and provenance.

Models: vanilla_lstm | hard_coulomb_lstm | hc_anchor_last | hc_anchor_pooled |
        hard_coulomb_tcn
Training recipe identical to sprint48 (MSE, AdamW 1e-3, cosine, clip 1.0,
patience 10, batch 1024). Checkpoints/results are versioned per
(model, variant, scenario, seed) under results/v5/headline_models/.

Usage examples:
  python experiments/run_multiseed_v5.py --variant v5c --scenarios A B --models vanilla_lstm hard_coulomb_lstm --seeds 42
  python experiments/run_multiseed_v5.py --variant v5c --scenarios A --models all --seeds 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "src"), str(BASE_DIR / "analysis"), str(BASE_DIR / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

from compare_anchor_variants import HCAnchorVariant  # noqa: E402
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_INPUTS, PVR_EPSILONS, CURRENT_THRESHOLD_A  # noqa: E402
from model_v5_coulomb import HardCoulombLSTM  # noqa: E402
from model_v5_coulomb_tcn import HardCoulombTCN  # noqa: E402
from soc_metrics import evaluate_soc_predictions  # noqa: E402
from sprint48_common import VanillaLSTM, make_loader, set_reproducibility  # noqa: E402

OUT_DIR = BASE_DIR / "results" / "v5" / "headline_models"
CKPT_DIR = OUT_DIR / "checkpoints"
PATIENCE = 10
ALL_MODELS = ("vanilla_lstm", "hard_coulomb_lstm", "hc_anchor_last", "hc_anchor_pooled", "hard_coulomb_tcn")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE_DIR).decode().strip()
    except Exception:
        return "unknown"


def load_splits(variant: str, scenario: str) -> Dict[str, np.ndarray]:
    d = BASE_DIR / "data" / "processed" / f"{variant}_scenario_{scenario}"
    if not d.exists():
        raise FileNotFoundError(f"Dataset variant not built: {d}")
    meta = json.loads((d / "metadata_v4.json").read_text())
    arrays = {}
    for split in ("train", "val", "test"):
        arrays[f"X_{split}"] = np.load(d / f"X_{split}.npy").astype(np.float32)
        arrays[f"y_{split}"] = np.load(d / f"y_{split}.npy").astype(np.float32)
        arrays[f"I_{split}"] = np.load(d / f"I_unscaled_{split}.npy").astype(np.float32)
    tl = d / "temp_labels_test.npy"
    arrays["temp_labels"] = np.load(tl, allow_pickle=True) if tl.exists() else None
    arrays["meta"] = meta
    return arrays


def build_model(kind: str) -> nn.Module:
    if kind == "vanilla_lstm":
        return VanillaLSTM()
    if kind == "hard_coulomb_lstm":
        return HardCoulombLSTM(num_inputs=NUM_INPUTS)
    if kind == "hc_anchor_last":
        return HCAnchorVariant(anchor_source="last")
    if kind == "hc_anchor_pooled":
        return HCAnchorVariant(anchor_source="pooled")
    if kind == "hard_coulomb_tcn":
        return HardCoulombTCN(num_inputs=NUM_INPUTS, num_filters=64, kernel_size=7,
                              dropout=0.2, dilation_rates=[1, 2, 4, 8], safety_factor=1.5)
    raise ValueError(kind)


def forward(model: nn.Module, kind: str, xb: torch.Tensor, ib: torch.Tensor) -> torch.Tensor:
    out = model(xb) if kind == "vanilla_lstm" else model(xb, ib)
    if out.ndim == 2:
        out = out.unsqueeze(-1)
    return out


def train_one(kind: str, variant: str, scenario: str, seed: int, device: torch.device,
              force: bool = False) -> Path:
    ckpt = CKPT_DIR / f"{kind}_{variant}_scenario_{scenario}_seed{seed}.pt"
    if ckpt.exists() and not force:
        return ckpt
    data = load_splits(variant, scenario)
    set_reproducibility(seed)
    train_loader = make_loader(data["X_train"], data["y_train"], data["I_train"], BATCH_SIZE, True, seed, device)
    val_loader = make_loader(data["X_val"], data["y_val"], data["I_val"], BATCH_SIZE, False, seed, device)
    model = build_model(kind).to(device)
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
            loss = criterion(forward(model, kind, xb, ib), yt)
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
                vl += float(criterion(forward(model, kind, xb, ib), yt).item())
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
        "model_kind": kind, "dataset_variant": variant, "scenario": scenario, "seed": seed,
        "label_mode": data["meta"].get("label_mode", "legacy"),
        "decimation_mode": data["meta"].get("decimation_mode", "first_sample"),
        "model_state_dict": best_state, "best_val_loss": best_val, "epochs_ran": epoch,
        "minutes": round((time.time() - t0) / 60, 2),
        "recipe": "sprint48-equivalent", "git_commit": git_commit(),
    }, ckpt)
    print(f"    trained {ckpt.name}: val {best_val:.6f}, {epoch} ep, {(time.time()-t0)/60:.1f} min")
    return ckpt


def evaluate_ckpt(ckpt: Path, device: torch.device, batch: int = 1024) -> Dict:
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    kind, variant, scenario = payload["model_kind"], payload["dataset_variant"], payload["scenario"]
    data = load_splits(variant, scenario)
    model = build_model(kind).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(data["X_test"]), batch):
            xb = torch.from_numpy(data["X_test"][i:i+batch]).to(device)
            ib = torch.from_numpy(data["I_test"][i:i+batch]).to(device)
            preds.append(forward(model, kind, xb, ib).squeeze(-1).cpu().numpy())
    y_pred = np.concatenate(preds, 0)
    m = evaluate_soc_predictions(data["y_test"], y_pred, data["I_test"], data["temp_labels"])
    return {
        "provenance": {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "git_commit": payload.get("git_commit", git_commit()),
            "dataset_version": variant, "label_mode": payload.get("label_mode"),
            "decimation_mode": payload.get("decimation_mode"),
            "scenario": f"scenario_{scenario}", "split": "test",
            "checkpoint": str(ckpt.relative_to(BASE_DIR)), "seed": payload["seed"],
            "config": {"threshold_A": CURRENT_THRESHOLD_A, "pvr_epsilons": list(PVR_EPSILONS)},
        },
        "model": kind, "metrics": m,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="v5c")
    parser.add_argument("--scenarios", nargs="+", default=["A"])
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    models = ALL_MODELS if args.models == ["all"] else tuple(args.models)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    results = []
    for scenario in args.scenarios:
        for kind in models:
            for seed in args.seeds:
                ckpt = train_one(kind, args.variant, scenario, seed, device)
                res = evaluate_ckpt(ckpt, device)
                reg = res["metrics"]["regression"]
                pt = res["metrics"].get("per_temperature", {})
                n20 = pt.get("n20degC", {}).get("regression", {})
                print(f"  {args.variant} S{scenario} {kind:20s} seed{seed:>4d} "
                      f"RMSE {reg['rmse_full_pct']:8.4f}% | MaxE {reg['maxe_full_pct']:8.4f}% | "
                      f"n20 RMSE {n20.get('rmse_full_pct', float('nan')):.4f}%")
                results.append(res)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out_json) if args.out_json else OUT_DIR / (
        f"runs_{args.variant}_{'-'.join(args.scenarios)}_{'-'.join(map(str, args.seeds))}.json")
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
