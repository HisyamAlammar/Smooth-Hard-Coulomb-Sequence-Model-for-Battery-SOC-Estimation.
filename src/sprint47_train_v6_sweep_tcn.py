"""
sprint47_train_v6_sweep_tcn.py -- Sprint 47: V6 Robust Anchor TCN Sweep
=======================================================================

Purpose
-------
Run the same weighted-anchor-loss hypothesis test as the V6 LSTM sweep, but on
the V5 Hard-Coulomb TCN backbone. This tests whether explicit anchor supervision
mitigates the -20C Anchor Collapse on a convolutional temporal backbone.

Safety
------
This script trains Scenario A only and writes only TCN-specific V6 artifacts:
  - outputs/models/best_model_v6_tcn_alpha_{alpha}.pt
  - logs/training_log_v6_tcn_alpha_{alpha}.csv
  - logs/sprint47_v6_sweep_tcn_results.json

It does not overwrite LSTM sweep artifacts or V5 TCN artifacts.

Usage
-----
python src/sprint47_train_v6_sweep_tcn.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (  # noqa: E402
    BATCH_SIZE,
    CURRENT_THRESHOLD,
    DATA_PROC,
    DILATION_RATES,
    DROPOUT,
    EPOCHS,
    KERNEL_SIZE,
    LEARNING_RATE,
    LOG_DIR,
    NUM_FILTERS,
    NUM_INPUTS,
    OUTPUT_MOD,
    RANDOM_SEED,
)
from model_v5_coulomb_tcn import HardCoulombTCN, count_parameters  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA_VALUES = [1.0, 5.0, 10.0, 20.0]
SAFETY_FACTOR = 1.5
GAMMA = 1.0 / (3.0 * 3600.0)
PATIENCE = 10
SCENARIO = "scenario_A"
ARTIFACT_PREFIX = "v6_tcn_alpha"
SUMMARY_PATH = os.path.join(LOG_DIR, "sprint47_v6_sweep_tcn_results.json")


class ExposedHardCoulombTCN(HardCoulombTCN):
    """V5 HardCoulombTCN with exposed anchor and constrained delta outputs."""

    def _apply_hard_coulomb(self, delta_soc_raw: torch.Tensor,
                            current_seq: torch.Tensor,
                            soc_anchor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        I_t = current_seq.unsqueeze(-1)
        hard = self.hard_constraint

        if hasattr(hard, "gamma_factor"):
            coulomb_limit = torch.abs(I_t) * hard.gamma_factor
        else:
            coulomb_limit = torch.abs(I_t) * hard.gamma * hard.safety_factor

        discharge_mask = I_t < -hard.threshold
        charge_mask = I_t > hard.threshold

        delta_constrained = torch.zeros_like(delta_soc_raw)
        delta_constrained[discharge_mask] = torch.clamp(
            delta_soc_raw[discharge_mask],
            min=-coulomb_limit[discharge_mask],
            max=torch.zeros_like(coulomb_limit[discharge_mask]),
        )
        delta_constrained[charge_mask] = torch.clamp(
            delta_soc_raw[charge_mask],
            min=torch.zeros_like(coulomb_limit[charge_mask]),
            max=coulomb_limit[charge_mask],
        )

        cumulative = torch.cumsum(delta_constrained, dim=1)
        y_pred = (soc_anchor.unsqueeze(1) + cumulative).clamp(0.0, 1.0)
        return y_pred, delta_constrained

    def forward(self, x: torch.Tensor, current_seq: torch.Tensor):
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        delta_raw = self.delta_head(h)
        anchor_pred = self.anchor_head(h[:, 0, :])
        y_pred, delta_out = self._apply_hard_coulomb(delta_raw, current_seq, anchor_pred)
        return y_pred, anchor_pred, delta_out


def log_print(msg: str = "") -> None:
    print(msg, flush=True)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def alpha_tag(alpha: float) -> str:
    return f"{alpha:.1f}"


def load_scenario_a() -> dict:
    data_dir = os.path.join(DATA_PROC, f"v3_{SCENARIO}")
    log_print(f"\n  Loading Scenario A from {data_dir} ...")
    data = {}
    for split in ["train", "val", "test"]:
        data[f"X_{split}"] = np.load(os.path.join(data_dir, f"X_{split}.npy")).astype(np.float32)
        data[f"y_{split}"] = np.load(os.path.join(data_dir, f"y_{split}.npy")).astype(np.float32)
        data[f"I_{split}"] = np.load(os.path.join(data_dir, f"I_unscaled_{split}.npy")).astype(np.float32)
        log_print(
            f"    {split}: X={data[f'X_{split}'].shape}, "
            f"y={data[f'y_{split}'].shape}, I={data[f'I_{split}'].shape}"
        )
    return data


def make_dataloaders(data: dict, batch_size: int):
    def make_loader(split: str, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(data[f"X_{split}"]),
            torch.from_numpy(data[f"y_{split}"]),
            torch.from_numpy(data[f"I_{split}"]),
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(DEVICE.type == "cuda"),
            drop_last=False,
        )

    return make_loader("train", True), make_loader("val", False), make_loader("test", False)


def build_model(seed_offset: int = 0) -> ExposedHardCoulombTCN:
    set_seed(RANDOM_SEED + seed_offset)
    return ExposedHardCoulombTCN(
        num_inputs=NUM_INPUTS,
        num_filters=NUM_FILTERS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
        dilation_rates=DILATION_RATES,
        safety_factor=SAFETY_FACTOR,
    ).to(DEVICE)


def artifact_paths(alpha: float) -> Dict[str, str]:
    tag = alpha_tag(alpha)
    return {
        "best_model": os.path.join(OUTPUT_MOD, f"best_model_v6_tcn_alpha_{tag}.pt"),
        "train_log": os.path.join(LOG_DIR, f"training_log_v6_tcn_alpha_{tag}.csv"),
    }


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_flat = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred_flat = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - y_true_flat.mean()) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_pvr(y_pred: np.ndarray, current_values: np.ndarray):
    delta_soc = y_pred[:, 1:] - y_pred[:, :-1]
    discharge_mask = current_values[:, 1:] < -CURRENT_THRESHOLD
    n_discharge = int(discharge_mask.sum())
    if n_discharge == 0:
        return 0.0, 0, 0
    violations = (delta_soc > 0.0) & discharge_mask
    n_violations = int(violations.sum())
    return (n_violations / n_discharge) * 100.0, n_violations, n_discharge


def compute_loss(model, X_b, y_b, I_b, criterion, alpha: float):
    y_true = y_b.unsqueeze(-1)
    y_pred, anchor_pred, delta_out = model(X_b, I_b)
    seq_loss = criterion(y_pred, y_true)
    anchor_true = y_b[:, 0:1]
    anchor_loss = criterion(anchor_pred, anchor_true)
    loss = seq_loss + (alpha * anchor_loss)
    return loss, seq_loss, anchor_loss, y_pred, anchor_pred, delta_out


def train_one_epoch(model, loader, criterion, optimizer, alpha: float, epoch: int, total_epochs: int):
    model.train()
    total_loss = 0.0
    total_seq_loss = 0.0
    total_anchor_loss = 0.0
    n_batches = 0

    pbar = tqdm(
        loader,
        desc=f"TCN Alpha {alpha_tag(alpha)} | Epoch {epoch}/{total_epochs} Train",
        leave=False,
        dynamic_ncols=True,
    )
    for X_b, y_b, I_b in pbar:
        X_b = X_b.to(DEVICE)
        y_b = y_b.to(DEVICE)
        I_b = I_b.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        loss, seq_loss, anchor_loss, _, _, _ = compute_loss(model, X_b, y_b, I_b, criterion, alpha)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_seq_loss += seq_loss.item()
        total_anchor_loss += anchor_loss.item()
        n_batches += 1
        pbar.set_postfix(
            loss=f"{loss.item():.6f}",
            seq=f"{seq_loss.item():.6f}",
            anchor=f"{anchor_loss.item():.6f}",
        )

    return {
        "loss": total_loss / max(n_batches, 1),
        "seq_loss": total_seq_loss / max(n_batches, 1),
        "anchor_loss": total_anchor_loss / max(n_batches, 1),
    }


def validate(model, loader, criterion, alpha: float, epoch: int, total_epochs: int):
    model.eval()
    total_loss = 0.0
    total_seq_loss = 0.0
    total_anchor_loss = 0.0
    n_batches = 0

    pbar = tqdm(
        loader,
        desc=f"TCN Alpha {alpha_tag(alpha)} | Epoch {epoch}/{total_epochs} Val",
        leave=False,
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for X_b, y_b, I_b in pbar:
            X_b = X_b.to(DEVICE)
            y_b = y_b.to(DEVICE)
            I_b = I_b.to(DEVICE)
            loss, seq_loss, anchor_loss, _, _, _ = compute_loss(model, X_b, y_b, I_b, criterion, alpha)

            total_loss += loss.item()
            total_seq_loss += seq_loss.item()
            total_anchor_loss += anchor_loss.item()
            n_batches += 1
            pbar.set_postfix(
                val=f"{loss.item():.6f}",
                seq=f"{seq_loss.item():.6f}",
                anchor=f"{anchor_loss.item():.6f}",
            )

    return {
        "loss": total_loss / max(n_batches, 1),
        "seq_loss": total_seq_loss / max(n_batches, 1),
        "anchor_loss": total_anchor_loss / max(n_batches, 1),
    }


def evaluate_test(model, test_loader, alpha: float) -> Dict:
    model.eval()
    all_preds, all_trues, all_currents, all_anchors = [], [], [], []
    with torch.no_grad():
        for X_b, y_b, I_b in tqdm(
            test_loader,
            desc=f"TCN Alpha {alpha_tag(alpha)} | Test Eval",
            leave=False,
            dynamic_ncols=True,
        ):
            X_b = X_b.to(DEVICE)
            I_b = I_b.to(DEVICE)
            y_pred, anchor_pred, _ = model(X_b, I_b)
            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(y_b.numpy())
            all_currents.append(I_b.cpu().numpy())
            all_anchors.append(anchor_pred.cpu().numpy())

    yp = np.concatenate(all_preds, axis=0).squeeze(-1)
    yt = np.concatenate(all_trues, axis=0)
    currents = np.concatenate(all_currents, axis=0)
    anchors = np.concatenate(all_anchors, axis=0).squeeze(-1)

    err = yp - yt
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    maxe = float(np.max(np.abs(err)))
    r2_full = r2_score_np(yt, yp)
    pvr, n_viol, n_dis = compute_pvr(yp, currents)
    anchor_err = np.abs(anchors - yt[:, 0])

    worst_flat = int(np.argmax(np.abs(err)))
    worst_seq, worst_step = np.unravel_index(worst_flat, err.shape)

    return {
        "alpha": alpha,
        "backbone": "HardCoulombTCN",
        "rmse_pct": rmse * 100.0,
        "mae_pct": mae * 100.0,
        "maxe_pct": maxe * 100.0,
        "r2_full": r2_full,
        "pvr_pct": pvr,
        "pvr_violations": n_viol,
        "pvr_discharge_steps": n_dis,
        "anchor_mae_pct": float(anchor_err.mean() * 100.0),
        "anchor_maxe_pct": float(anchor_err.max() * 100.0),
        "worst_sequence_index": int(worst_seq),
        "worst_timestep": int(worst_step),
        "worst_true_soc_pct": float(yt[worst_seq, worst_step] * 100.0),
        "worst_pred_soc_pct": float(yp[worst_seq, worst_step] * 100.0),
        "n_test_sequences": int(len(yt)),
    }


def save_training_log(history: List[Dict], path: str) -> None:
    if not history:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def load_existing_summary(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
            return payload["runs"]
    except json.JSONDecodeError:
        pass
    return []


def append_summary(result: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    runs = load_existing_summary(path)
    runs.append(result)
    payload = {
        "sprint": 47,
        "experiment": "V6 Robust Anchor TCN alpha sweep",
        "scenario": SCENARIO,
        "alpha_values": ALPHA_VALUES,
        "safety_factor": SAFETY_FACTOR,
        "gamma": GAMMA,
        "artifact_namespace": "_v6_tcn_alpha_{val}_",
        "runs": runs,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def train_alpha(alpha: float, train_loader, val_loader, test_loader, epochs: int,
                seed_offset: int) -> Dict:
    paths = artifact_paths(alpha)
    model = build_model(seed_offset=seed_offset)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []
    start_time = time.time()

    log_print(f"\n{'=' * 82}")
    log_print(f"  V6 TCN Alpha Sweep | alpha={alpha_tag(alpha)} | Scenario A")
    log_print(f"{'=' * 82}")
    log_print(f"  Fresh HardCoulombTCN initialized | params={count_parameters(model):,}")
    log_print(f"  Receptive field={model.receptive_field} steps")
    log_print(f"  Gamma={model.hard_constraint.gamma:.6e} | safety_factor={model.hard_constraint.safety_factor}")
    log_print(f"  Best model path: {paths['best_model']}")

    os.makedirs(OUTPUT_MOD, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, alpha, epoch, epochs)
        val_stats = validate(model, val_loader, criterion, alpha, epoch, epochs)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        record = {
            "alpha": alpha,
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_seq_loss": train_stats["seq_loss"],
            "train_anchor_loss": train_stats["anchor_loss"],
            "val_loss": val_stats["loss"],
            "val_seq_loss": val_stats["seq_loss"],
            "val_anchor_loss": val_stats["anchor_loss"],
            "lr": lr,
            "time_sec": round(elapsed, 1),
        }
        history.append(record)

        improved = ""
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "alpha": alpha,
                "model_state_dict": model.state_dict(),
                "val_loss": val_stats["loss"],
                "val_seq_loss": val_stats["seq_loss"],
                "val_anchor_loss": val_stats["anchor_loss"],
                "train_loss": train_stats["loss"],
                "config": {
                    "version": f"{ARTIFACT_PREFIX}_{alpha_tag(alpha)}",
                    "base_model": "HardCoulombTCN",
                    "num_inputs": NUM_INPUTS,
                    "num_filters": NUM_FILTERS,
                    "kernel_size": KERNEL_SIZE,
                    "dropout": DROPOUT,
                    "dilation_rates": DILATION_RATES,
                    "safety_factor": SAFETY_FACTOR,
                    "gamma": GAMMA,
                    "anchor_loss_alpha": alpha,
                },
            }, paths["best_model"])
            improved = " ** BEST **"
        else:
            patience_counter += 1

        log_print(
            f"  TCN Alpha {alpha_tag(alpha)} | Epoch {epoch:3d}/{epochs} | "
            f"Train={train_stats['loss']:.6f} "
            f"(seq={train_stats['seq_loss']:.6f}, anchor={train_stats['anchor_loss']:.6f}) | "
            f"Val={val_stats['loss']:.6f} "
            f"(seq={val_stats['seq_loss']:.6f}, anchor={val_stats['anchor_loss']:.6f}) | "
            f"LR={lr:.2e} | {elapsed:.1f}s{improved}"
        )

        save_training_log(history, paths["train_log"])

        if patience_counter >= PATIENCE:
            log_print(f"\n  Early stopping TCN alpha={alpha_tag(alpha)} at epoch {epoch}")
            break

    if os.path.exists(paths["best_model"]):
        checkpoint = torch.load(paths["best_model"], map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_epoch = checkpoint.get("epoch", "?")
    else:
        best_epoch = None

    eval_metrics = evaluate_test(model, test_loader, alpha)
    total_minutes = (time.time() - start_time) / 60.0
    result = {
        **eval_metrics,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "train_minutes": round(total_minutes, 2),
        "model_path": paths["best_model"],
        "train_log_path": paths["train_log"],
    }

    append_summary(result, SUMMARY_PATH)
    log_print(f"\n  TCN Alpha {alpha_tag(alpha)} test metrics:")
    log_print(f"    RMSE={result['rmse_pct']:.4f}% | MAE={result['mae_pct']:.4f}% | "
              f"MaxE={result['maxe_pct']:.4f}% | PVR={result['pvr_pct']:.2f}%")
    log_print(f"    Anchor MAE={result['anchor_mae_pct']:.4f}% | "
              f"Anchor MaxE={result['anchor_maxe_pct']:.4f}%")
    log_print(f"    Summary updated: {SUMMARY_PATH}")

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Sprint 47 V6 TCN alpha sweep for anchor loss.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Epochs per alpha. Default: {EPOCHS}")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size. Default: {BATCH_SIZE}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_SEED)

    log_print("=" * 82)
    log_print("  Sprint 47: V6 Robust Anchor TCN Alpha Sweep")
    log_print("  V5 HardCoulombTCN architecture + weighted anchor objective")
    log_print("=" * 82)
    log_print(f"  Device: {DEVICE} | Batch: {args.batch_size} | Epochs/alpha: {args.epochs}")
    log_print(f"  Alpha values: {ALPHA_VALUES}")
    log_print("  Artifact namespace: _v6_tcn_alpha_{val}_")
    log_print("  LSTM sweep and V5 TCN artifacts are not overwritten.")

    data = load_scenario_a()
    train_loader, val_loader, test_loader = make_dataloaders(data, args.batch_size)
    del data
    log_print(f"  Batches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    all_results = []
    for run_idx, alpha in enumerate(ALPHA_VALUES):
        try:
            result = train_alpha(alpha, train_loader, val_loader, test_loader, args.epochs, seed_offset=run_idx)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and DEVICE.type == "cuda":
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA OOM during TCN alpha sweep. Re-run with a smaller --batch-size."
                ) from exc
            raise
        all_results.append(result)

    log_print(f"\n{'=' * 82}")
    log_print("  Sprint 47 V6 TCN Alpha Sweep Complete")
    log_print(f"{'=' * 82}")
    log_print(f"  {'Alpha':<8} {'RMSE%':<10} {'MAE%':<10} {'MaxE%':<10} {'PVR%':<8} {'AnchorMAE%':<12}")
    log_print(f"  {'-' * 68}")
    for result in all_results:
        log_print(
            f"  {alpha_tag(result['alpha']):<8} {result['rmse_pct']:<10.4f} "
            f"{result['mae_pct']:<10.4f} {result['maxe_pct']:<10.4f} "
            f"{result['pvr_pct']:<8.2f} {result['anchor_mae_pct']:<12.4f}"
        )
    log_print(f"\n  Summary JSON: {SUMMARY_PATH}")
    log_print(f"  Models: {OUTPUT_MOD}")


if __name__ == "__main__":
    main()
