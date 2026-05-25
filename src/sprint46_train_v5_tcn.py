"""
sprint46_train_v5_tcn.py -- Sprint 46: V5 Hard-Coulomb TCN Training
====================================================================

Backbone-agnostic validation of the V5 Hard-Coulomb constraint. This script
trains the V3 TCN backbone with the V5 Coulomb-bounded delta layer.

CRITICAL NAMESPACE SAFETY:
  All artifacts written by this script include '_v5_coulomb_tcn_' and therefore
  do not overwrite V3, V4, or V5 LSTM models/logs/figures.

Usage:
    python src/sprint46_train_v5_tcn.py
    python src/sprint46_train_v5_tcn.py --resume
    python src/sprint46_train_v5_tcn.py --eval-only
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (  # noqa: E402
    BATCH_SIZE,
    DATA_PROC,
    DILATION_RATES,
    EPOCHS,
    KERNEL_SIZE,
    LEARNING_RATE,
    LOG_DIR,
    NUM_FILTERS,
    NUM_INPUTS,
    OUTPUT_FIG,
    OUTPUT_MOD,
    RANDOM_SEED,
)
from model_v5_coulomb_tcn import HardCoulombTCN, count_parameters  # noqa: E402


torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10
SAFETY_FACTOR = 1.5
GAMMA = 1.0 / (3.0 * 3600.0)
DROPOUT = 0.2
ARTIFACT_TAG = "v5_coulomb_tcn"


def log_print(msg: str) -> None:
    print(msg, flush=True)


def safe_torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_flat = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred_flat = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - y_true_flat.mean()) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_pvr_local(y_pred: np.ndarray, current_values: np.ndarray):
    delta_soc = y_pred[:, 1:] - y_pred[:, :-1]
    discharge_mask = current_values[:, 1:] < -0.05
    n_discharge = int(discharge_mask.sum())
    if n_discharge == 0:
        return 0.0, 0, 0
    violations = (delta_soc > 0) & discharge_mask
    n_violations = int(violations.sum())
    return (n_violations / n_discharge) * 100.0, n_violations, n_discharge


def compute_per_temp_rmse_local(y_pred: np.ndarray, y_true: np.ndarray, temp_labels: np.ndarray):
    results = {}
    for temp in sorted(np.unique(temp_labels)):
        mask = temp_labels == temp
        if mask.sum() == 0:
            continue
        errors = y_pred[mask] - y_true[mask]
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))
        results[temp] = {
            "rmse": rmse,
            "rmse_pct": rmse * 100,
            "mae": mae,
            "mae_pct": mae * 100,
            "n_samples": int(mask.sum()),
        }
    return results


def load_scenario(scenario_name: str) -> dict:
    data_dir = os.path.join(DATA_PROC, f"v3_{scenario_name}")
    log_print(f"\n  Loading v3 {scenario_name} from {data_dir} ...")

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
    def make_loader(split: str, shuffle: bool):
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


def build_model() -> HardCoulombTCN:
    return HardCoulombTCN(
        num_inputs=NUM_INPUTS,
        num_filters=NUM_FILTERS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
        dilation_rates=DILATION_RATES,
        safety_factor=SAFETY_FACTOR,
    ).to(DEVICE)


def checkpoint_config() -> dict:
    return {
        "version": ARTIFACT_TAG,
        "num_inputs": NUM_INPUTS,
        "num_filters": NUM_FILTERS,
        "kernel_size": KERNEL_SIZE,
        "dropout": DROPOUT,
        "dilation_rates": DILATION_RATES,
        "safety_factor": SAFETY_FACTOR,
        "gamma": GAMMA,
    }


def artifact_paths(scenario_name: str) -> dict:
    return {
        "log": os.path.join(LOG_DIR, f"training_log_{ARTIFACT_TAG}_{scenario_name}.csv"),
        "best": os.path.join(OUTPUT_MOD, f"best_model_{ARTIFACT_TAG}_{scenario_name}.pt"),
        "latest": os.path.join(OUTPUT_MOD, f"latest_model_{ARTIFACT_TAG}_{scenario_name}.pt"),
        "figure": os.path.join(OUTPUT_FIG, f"soc_plot_{ARTIFACT_TAG}_{scenario_name}.png"),
    }


def train_one_epoch(model, loader, criterion, optimizer, device, tag, epoch, total_epochs):
    model.train()
    total_loss, n_batches = 0.0, 0
    pbar = tqdm(
        loader,
        desc=f"  [{tag}] Ep {epoch:3d}/{total_epochs} Train",
        leave=False,
        dynamic_ncols=True,
    )

    for X_b, y_b, I_b in pbar:
        X_b = X_b.to(device)
        y_t = y_b.unsqueeze(-1).to(device)
        I_b = I_b.to(device)

        optimizer.zero_grad(set_to_none=True)
        y_p = model(X_b, I_b)
        loss = criterion(y_p, y_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / max(n_batches, 1)


def validate(model, loader, criterion, device, tag, epoch, total_epochs):
    model.eval()
    total_loss, n_batches = 0.0, 0
    pbar = tqdm(
        loader,
        desc=f"  [{tag}] Ep {epoch:3d}/{total_epochs} Val  ",
        leave=False,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for X_b, y_b, I_b in pbar:
            X_b = X_b.to(device)
            y_t = y_b.unsqueeze(-1).to(device)
            I_b = I_b.to(device)
            y_p = model(X_b, I_b)
            loss = criterion(y_p, y_t)
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(val_loss=f"{loss.item():.6f}")

    return total_loss / max(n_batches, 1)


def train_scenario(scenario_name: str, batch_size: int = BATCH_SIZE, resume: bool = False):
    log_print(f"\n{'=' * 70}")
    log_print(f"  TRAINING V5-COULOMB-TCN: {scenario_name}")
    log_print("  V3 TCN Backbone | Hard-Coulomb Envelope | Pure MSE")
    log_print(f"{'=' * 70}")

    data = load_scenario(scenario_name)
    train_loader, val_loader, test_loader = make_dataloaders(data, batch_size)
    del data

    model = build_model()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)

    paths = artifact_paths(scenario_name)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MOD, exist_ok=True)

    log_print(f"  Batches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    log_print(f"  Device: {DEVICE}")
    log_print(f"  Parameters: {count_parameters(model):,}")
    log_print(f"  Receptive field: {model.receptive_field} steps")
    log_print(f"  Gamma (SOC/A/s): {model.hard_constraint.gamma:.6e}")
    log_print(f"  Safety factor: {model.hard_constraint.safety_factor}")
    log_print(f"  Gamma factor: {model.hard_constraint.gamma_factor:.6e}")
    log_print(f"  Loss: Pure MSELoss")
    log_print(f"  Scheduler: CosineAnnealingLR (T_max={EPOCHS})")
    log_print(f"  Early Stopping: patience={PATIENCE}")
    log_print("  Artifact namespace: _v5_coulomb_tcn_")

    history = []
    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 1

    if resume and os.path.exists(paths["latest"]):
        log_print(f"  [RESUME] Loading {paths['latest']} ...")
        ckpt = safe_torch_load(paths["latest"], DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        history = ckpt.get("history", [])
        start_epoch = ckpt["epoch"] + 1
        log_print(f"  [RESUME] Continuing from epoch {start_epoch}")

    log_print(f"\n  Starting training for {EPOCHS} epochs...\n")
    t_start = time.time()

    for epoch in range(start_epoch, EPOCHS + 1):
        t_epoch = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            scenario_name, epoch, EPOCHS)
        val_loss = validate(
            model, val_loader, criterion, DEVICE,
            scenario_name, epoch, EPOCHS)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t_epoch
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "time_sec": round(elapsed, 1),
        }
        history.append(record)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
                "config": checkpoint_config(),
            }, paths["best"])
            improved = " ** BEST **"
        else:
            patience_counter += 1

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
            "config": checkpoint_config(),
        }, paths["latest"])

        log_print(
            f"  Epoch {epoch:3d}/{EPOCHS} | Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | LR: {lr:.2e} | {elapsed:.1f}s{improved}"
        )

        if patience_counter >= PATIENCE:
            log_print(f"\n  Early stopping at epoch {epoch}")
            break

    total_time = time.time() - t_start
    log_print(f"\n  Training complete in {total_time / 60:.1f} min")
    log_print(f"  Best val loss: {best_val_loss:.6f}")

    if history:
        with open(paths["log"], "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        log_print(f"  Log saved: {paths['log']}")

    if os.path.exists(paths["best"]):
        ckpt = safe_torch_load(paths["best"], DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        log_print(f"  Loaded best weights from epoch {ckpt.get('epoch', '?')}")

    return model, history, test_loader


def evaluate_model(model, test_loader, device, scenario_name: str):
    model.eval()
    all_preds, all_trues, all_currents = [], [], []

    with torch.no_grad():
        for X_b, y_b, I_b in tqdm(test_loader, desc=f"  Eval {scenario_name}", leave=False):
            X_b = X_b.to(device)
            I_b = I_b.to(device)
            y_p = model(X_b, I_b)
            all_preds.append(y_p.cpu().numpy())
            all_trues.append(y_b.numpy())
            all_currents.append(I_b.cpu().numpy())

    yp = np.concatenate(all_preds, axis=0).squeeze(-1)
    yt = np.concatenate(all_trues, axis=0)
    Ia = np.concatenate(all_currents, axis=0)

    err = yp - yt
    rmse_full = np.sqrt(np.mean(err ** 2))
    mae_full = np.mean(np.abs(err))
    maxe_full = np.max(np.abs(err))
    r2_full = r2_score_np(yt, yp)
    rmse_last = np.sqrt(np.mean((yp[:, -1] - yt[:, -1]) ** 2))
    mae_last = np.mean(np.abs(yp[:, -1] - yt[:, -1]))
    maxe_last = np.max(np.abs(yp[:, -1] - yt[:, -1]))
    r2_last = r2_score_np(yt[:, -1], yp[:, -1])
    pvr, n_viol, n_dis = compute_pvr_local(yp, Ia)

    metrics = {
        "scenario": scenario_name,
        "model": ARTIFACT_TAG,
        "rmse_full_pct": rmse_full * 100,
        "mae_full_pct": mae_full * 100,
        "max_error_full_pct": maxe_full * 100,
        "r2_full": r2_full,
        "rmse_last_pct": rmse_last * 100,
        "mae_last_pct": mae_last * 100,
        "max_error_last_pct": maxe_last * 100,
        "r2_last": r2_last,
        "pvr": pvr,
        "pvr_violations": n_viol,
        "pvr_discharge_steps": n_dis,
        "n_samples": len(yt),
    }

    log_print(f"\n  Results ({scenario_name}):")
    log_print(f"    [Full-Seq]  RMSE: {rmse_full * 100:.4f}%  R2={r2_full:.6f}")
    log_print(f"    [Full-Seq]  MAE : {mae_full * 100:.4f}%")
    log_print(f"    [Full-Seq]  MaxE: {maxe_full * 100:.4f}%")
    log_print(f"    [Last-Step] RMSE: {rmse_last * 100:.4f}%  R2={r2_last:.6f}")
    log_print(f"    PVR: {pvr:.2f}% ({n_viol:,} / {n_dis:,})")

    return metrics, yp, yt, Ia


def plot_soc(y_true, y_pred, scenario_name, save_path, n_points=5000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    y_t = y_true.flatten()[:n_points]
    y_p = y_pred.flatten()[:n_points]
    idx = np.arange(len(y_t))
    errors = np.abs(y_t - y_p)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    fig.suptitle(f"Hard-Coulomb TCN (v5) -- {scenario_name}",
                 fontweight="bold", fontsize=13)

    axes[0].plot(idx, y_t * 100, color="#1565C0", lw=0.8, alpha=0.9, label="True SOC")
    axes[0].plot(idx, y_p * 100, color="#E65100", lw=0.8, alpha=0.75,
                 label="Predicted SOC", linestyle="--")
    axes[0].set_ylabel("SOC (%)")
    axes[0].set_xlabel("Time Step Index")
    axes[0].legend(loc="upper right", framealpha=0.9)

    rmse_val = np.sqrt(np.mean((y_t - y_p) ** 2)) * 100
    axes[1].fill_between(idx, errors * 100, color="#FF9800", alpha=0.45)
    axes[1].axhline(y=rmse_val, color="#C62828", ls=":", lw=1.0,
                    label=f"RMSE = {rmse_val:.2f}%")
    axes[1].set_ylabel("Abs. Error (%)")
    axes[1].set_xlabel("Time Step Index")
    axes[1].legend(loc="upper right", framealpha=0.9)

    sc = axes[2].scatter(y_t * 100, y_p * 100, c=errors * 100, cmap="YlOrRd",
                         vmin=0, vmax=60, s=1, alpha=0.5, edgecolors="none",
                         rasterized=True)
    axes[2].plot([0, 100], [0, 100], "k--", lw=0.8, alpha=0.5, label="Ideal (y=x)")
    axes[2].set_xlabel("True SOC (%)")
    axes[2].set_ylabel("Predicted SOC (%)")
    axes[2].set_aspect("equal")
    axes[2].set_xlim(0, 100)
    axes[2].set_ylim(0, 100)
    axes[2].legend(loc="lower right", framealpha=0.9)
    fig.colorbar(sc, ax=axes[2], shrink=0.6, pad=0.02).set_label("Abs. Error (%)", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    fig.savefig(save_path.replace(".png", ".pdf"))
    plt.close(fig)
    log_print(f"  Figure saved: {save_path}")


def run_eval_only(scenario_name: str, device):
    log_print(f"\n{'=' * 70}")
    log_print(f"  EVAL-ONLY V5-COULOMB-TCN: {scenario_name}")
    log_print(f"{'=' * 70}")

    paths = artifact_paths(scenario_name)
    if not os.path.exists(paths["best"]):
        raise FileNotFoundError(f"No checkpoint: {paths['best']}")

    ckpt = safe_torch_load(paths["best"], device)
    cfg = ckpt.get("config", {})
    model = HardCoulombTCN(
        num_inputs=cfg.get("num_inputs", NUM_INPUTS),
        num_filters=cfg.get("num_filters", NUM_FILTERS),
        kernel_size=cfg.get("kernel_size", KERNEL_SIZE),
        dropout=cfg.get("dropout", DROPOUT),
        dilation_rates=cfg.get("dilation_rates", DILATION_RATES),
        safety_factor=cfg.get("safety_factor", SAFETY_FACTOR),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    log_print(f"  Loaded best weights from epoch {ckpt.get('epoch', '?')}")

    data_dir = os.path.join(DATA_PROC, f"v3_{scenario_name}")
    X = torch.from_numpy(np.load(os.path.join(data_dir, "X_test.npy")).astype(np.float32))
    y = torch.from_numpy(np.load(os.path.join(data_dir, "y_test.npy")).astype(np.float32))
    I = torch.from_numpy(np.load(os.path.join(data_dir, "I_unscaled_test.npy")).astype(np.float32))
    test_loader = DataLoader(TensorDataset(X, y, I), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    metrics, yp, yt, Ia = evaluate_model(model, test_loader, device, scenario_name)
    maybe_print_per_temp(scenario_name, yp, yt)
    os.makedirs(OUTPUT_FIG, exist_ok=True)
    plot_soc(yt, yp, scenario_name.replace("_", " ").title(), paths["figure"])
    return metrics, yp, yt


def maybe_print_per_temp(scenario_name: str, yp: np.ndarray, yt: np.ndarray) -> None:
    temp_path = os.path.join(DATA_PROC, f"v3_{scenario_name}", "temp_labels_test.npy")
    if not os.path.exists(temp_path):
        return

    temp_labels = np.load(temp_path, allow_pickle=True)
    yp_last = yp[:, -1]
    yt_last = yt[:, -1]
    if len(temp_labels) != len(yp_last):
        return

    per_temp = compute_per_temp_rmse_local(yp_last, yt_last, temp_labels)
    log_print("    Per-Temperature RMSE (last-step):")
    for temp, metrics in per_temp.items():
        log_print(f"      {temp:>8s}: RMSE={metrics['rmse_pct']:6.2f}%  N={metrics['n_samples']:,}")


def print_final_comparison(all_metrics: dict, title: str) -> None:
    log_print(f"\n{'=' * 70}")
    log_print(f"  {title}")
    log_print(f"{'=' * 70}")
    log_print(f"  {'Metric':<22} {'Scenario A':<18} {'Scenario B':<18}")
    log_print(f"  {'-' * 60}")

    scenario_a = all_metrics.get("scenario_A", {})
    scenario_b = all_metrics.get("scenario_B", {})
    for key in ["rmse_full_pct", "mae_full_pct", "max_error_full_pct", "pvr", "r2_full"]:
        value_a = scenario_a.get(key, 0)
        value_b = scenario_b.get(key, 0)
        fmt = ".4f" if "r2" in key else ".2f"
        log_print(f"  {key:<22} {value_a:<18{fmt}} {value_b:<18{fmt}}")


def parse_args():
    parser = argparse.ArgumentParser(description="Sprint 46: V5 Hard-Coulomb TCN")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    log_print("=" * 70)
    log_print("  Sprint 46: V5 Hard-Coulomb TCN")
    log_print("  V3 TCN backbone + Coulomb-bounded delta constraint")
    if args.eval_only:
        log_print("  [MODE] EVAL-ONLY")
    log_print("=" * 70)
    log_print(f"  Device: {DEVICE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")
    log_print(f"  Safety factor: {SAFETY_FACTOR} | Gamma: {GAMMA:.6e} SOC/A/s")
    log_print("  Artifact namespace: _v5_coulomb_tcn_")

    if args.eval_only:
        metrics_a, _, _ = run_eval_only("scenario_A", DEVICE)
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        metrics_b, _, _ = run_eval_only("scenario_B", DEVICE)
        print_final_comparison(
            {"scenario_A": metrics_a, "scenario_B": metrics_b},
            "V5-COULOMB-TCN FINAL COMPARISON (eval-only)",
        )
        return

    all_metrics = {}
    batch_size = BATCH_SIZE

    for scenario in ["scenario_A", "scenario_B"]:
        try:
            model, _, test_loader = train_scenario(scenario, batch_size, args.resume)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and DEVICE.type == "cuda":
                torch.cuda.empty_cache()
                batch_size //= 2
                log_print(f"  OOM! Reducing batch to {batch_size}")
                model, _, test_loader = train_scenario(scenario, batch_size, args.resume)
            else:
                raise

        metrics, yp, yt, Ia = evaluate_model(model, test_loader, DEVICE, scenario)
        all_metrics[scenario] = metrics
        maybe_print_per_temp(scenario, yp, yt)

        paths = artifact_paths(scenario)
        os.makedirs(OUTPUT_FIG, exist_ok=True)
        plot_soc(yt, yp, scenario.replace("_", " ").title(), paths["figure"])

        del model, test_loader
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    results_path = os.path.join(LOG_DIR, f"sprint46_results_{ARTIFACT_TAG}_all.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    log_print(f"\n  Results JSON: {results_path}")

    print_final_comparison(all_metrics, "V5-COULOMB-TCN FINAL COMPARISON")

    log_print("\n  All V5-Coulomb-TCN artifacts saved with _v5_coulomb_tcn_ namespace.")
    log_print(f"  Models : {OUTPUT_MOD}")
    log_print(f"  Logs   : {LOG_DIR}")
    log_print(f"  Figures: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
