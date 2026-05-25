"""
train.py — Sprint 5: Seq2Seq Training Pipeline
============================================================================
Trains PI-TCN SOC Estimator (Seq2Seq) on both Scenario A (Zero-Shot) and
Scenario B (In-Distribution), with Early Stopping, checkpointing, metric
logging, intra-window PVR computation, per-temperature RMSE, and
publication-quality plots.

Architecture: Input (B,100,5) -> TCN -> Output (B,100,1)  [Seq2Seq]
Physics Loss: Intra-window monotonicity penalty during discharge

Created : 2026-04-08
Updated : 2026-05-10 — Sprint 5 (Seq2Seq pivot)
Framework: PyTorch (FINAL)
Usage   : python src/train.py
"""

import os
import sys
import time
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score

# ── Ensure src/ is importable ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_PROC, OUTPUT_MOD, OUTPUT_FIG, LOG_DIR,
    NUM_INPUTS, NUM_FILTERS, KERNEL_SIZE, DROPOUT, DILATION_RATES,
    BATCH_SIZE, LEARNING_RATE, EPOCHS, LAMBDA_PHYS, RANDOM_SEED,
)
from model import TCN_SOC_Estimator, PhysicsInformedLoss, count_parameters
from evaluate import compute_pvr, compute_per_temp_rmse


# =====================================================================
# 0. Reproducibility & Device
# =====================================================================
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10  # Early stopping patience
MAX_LAMBDA = 5.0  # Peak physics loss weight (curriculum schedule)
LAMBDA_WARMUP_EPOCHS = 30  # Epochs to ramp lambda from 0 → MAX_LAMBDA


def log_print(msg: str):
    """Print with flush for real-time visibility in terminal."""
    print(msg, flush=True)


# =====================================================================
# 1. Data Loading
# =====================================================================
def load_scenario(scenario_name: str):
    """
    Load preprocessed .npy files for a given scenario.

    Returns
    -------
    dict with keys: X_train, y_train, X_val, y_val, X_test, y_test
    """
    d = os.path.join(DATA_PROC, scenario_name)

    log_print(f"\n  Loading {scenario_name} from {d} ...")

    data = {}
    for split in ["train", "val", "test"]:
        x_path = os.path.join(d, f"X_{split}.npy")
        y_path = os.path.join(d, f"y_{split}.npy")

        if not os.path.exists(x_path):
            raise FileNotFoundError(f"Missing: {x_path}")

        # Load as float32 numpy arrays
        data[f"X_{split}"] = np.load(x_path).astype(np.float32)
        data[f"y_{split}"] = np.load(y_path).astype(np.float32)

        log_print(f"    {split}: X={data[f'X_{split}'].shape}, "
                  f"y={data[f'y_{split}'].shape}")

    return data


def make_dataloaders(data: dict, batch_size: int):
    """
    Convert numpy arrays to PyTorch DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    def _make(x_key, y_key, shuffle):
        X = torch.from_numpy(data[x_key])
        y = torch.from_numpy(data[y_key])
        ds = TensorDataset(X, y)
        # num_workers=0 for Windows stability; pin_memory for GPU
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=(DEVICE.type == "cuda"),
                          drop_last=False)

    train_loader = _make("X_train", "y_train", shuffle=True)
    val_loader   = _make("X_val",   "y_val",   shuffle=False)
    test_loader  = _make("X_test",  "y_test",  shuffle=False)

    return train_loader, val_loader, test_loader


# =====================================================================
# 2. Training Loop (Single Epoch) — Seq2Seq
# =====================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, tag, epoch, total_epochs):
    """Train for one epoch (Seq2Seq). Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    # Curriculum lambda: ramp from 0 → MAX_LAMBDA over LAMBDA_WARMUP_EPOCHS
    lambda_curr = min(MAX_LAMBDA, MAX_LAMBDA * (epoch / LAMBDA_WARMUP_EPOCHS))
    if hasattr(criterion, 'lambda_phys'):
        criterion.lambda_phys = lambda_curr

    pbar = tqdm(loader, desc=f"  [{tag}] Ep {epoch}/{total_epochs} Train", leave=False, dynamic_ncols=True)
    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device)         # (B, 100, 5)
        y_batch = y_batch.to(device)         # (B, 100)

        # Reshape y to (B, 100, 1) for Seq2Seq loss
        y_target = y_batch.unsqueeze(-1)     # (B, 100, 1)

        # Full current sequence for intra-window physics loss
        # Unscale: I_real = I_scaled * 40.0 - 20.0
        current_seq = X_batch[:, :, 1] * 40.0 - 20.0  # (B, 100)

        optimizer.zero_grad()
        y_pred = model(X_batch)              # (B, 100, 1)
        loss = criterion(y_pred, y_target, current_seq)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.6f}", λ=f"{lambda_curr:.3f}")

    return total_loss / max(n_batches, 1)


def validate(model, loader, criterion, device, tag, epoch, total_epochs):
    """Validate model (Seq2Seq). Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    # Mirror curriculum lambda from training epoch
    lambda_curr = min(MAX_LAMBDA, MAX_LAMBDA * (epoch / LAMBDA_WARMUP_EPOCHS))
    if hasattr(criterion, 'lambda_phys'):
        criterion.lambda_phys = lambda_curr

    pbar = tqdm(loader, desc=f"  [{tag}] Ep {epoch}/{total_epochs} Val  ", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_target = y_batch.unsqueeze(-1)           # (B, 100, 1)
            current_seq = X_batch[:, :, 1] * 40.0 - 20.0  # (B, 100)

            y_pred = model(X_batch)                    # (B, 100, 1)
            loss = criterion(y_pred, y_target, current_seq)

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}", λ=f"{lambda_curr:.3f}")

    return total_loss / max(n_batches, 1)


def validate_mse_only(model, loader, device):
    """
    Compute PURE MSE validation loss (no physics penalty).

    Used for:
      - best_val_loss comparison & best-model checkpoint saving
      - Early stopping patience counter

    This ensures the curriculum lambda (0→5.0) cannot artificially
    inflate val_loss and trigger incorrect early stopping or LR decay.
    """
    mse_fn = nn.MSELoss()
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch  = X_batch.to(device)
            y_target = y_batch.unsqueeze(-1).to(device)  # (B, 100, 1)
            y_pred   = model(X_batch)                     # (B, 100, 1)
            total   += mse_fn(y_pred, y_target).item()
            n       += 1
    return total / max(n, 1)


# =====================================================================
# 3. Full Training Pipeline
# =====================================================================
def train_scenario(scenario_name: str, batch_size: int = BATCH_SIZE, resume: bool = False):
    """
    Full training pipeline for one scenario.

    Returns
    -------
    model : trained TCN_SOC_Estimator
    history : list of dicts with epoch metrics
    test_loader : DataLoader for testing
    """
    log_print(f"\n{'='*65}")
    log_print(f"  TRAINING: {scenario_name}")
    log_print(f"{'='*65}")

    # ── Load data ────────────────────────────────────────────────────
    data = load_scenario(scenario_name)
    train_loader, val_loader, test_loader = make_dataloaders(data, batch_size)
    del data  # free memory

    log_print(f"  Batches: train={len(train_loader)}, "
              f"val={len(val_loader)}, test={len(test_loader)}")
    log_print(f"  Device: {DEVICE}")
    log_print(f"  Batch size: {batch_size}")

    # ── Model, Loss, Optimizer ───────────────────────────────────────
    model = TCN_SOC_Estimator(
        num_inputs=NUM_INPUTS,
        num_filters=NUM_FILTERS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
        dilation_rates=DILATION_RATES,
    ).to(DEVICE)

    criterion = PhysicsInformedLoss(lambda_phys=LAMBDA_PHYS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Fix R5: CosineAnnealingLR is immune to curriculum-lambda-inflated val_loss.
    # It decays LR on a fixed cosine schedule regardless of loss magnitude.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    log_print(f"  Parameters: {count_parameters(model):,}")
    log_print(f"  Optimizer: AdamW (lr={LEARNING_RATE})")
    log_print(f"  Scheduler: CosineAnnealingLR (T_max={EPOCHS}, eta_min=1e-6)")
    log_print(f"  Loss: PhysicsInformedLoss + Curriculum λ →0→{MAX_LAMBDA}")
    log_print(f"  Early Stopping: patience={PATIENCE} (on pure MSE val loss)")

    # ── Prepare logging ──────────────────────────────────────────────
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MOD, exist_ok=True)

    log_path = os.path.join(LOG_DIR, f"training_log_{scenario_name}.csv")
    best_model_path = os.path.join(OUTPUT_MOD, f"pi_tcn_{scenario_name}.pt")
    latest_model_path = os.path.join(OUTPUT_MOD, f"pi_tcn_{scenario_name}_latest.pt")

    history = []
    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 1

    # ── Resume Logic ─────────────────────────────────────────────────
    if resume and os.path.exists(latest_model_path):
        log_print(f"\n  [INFO] Found checkpoint, loading {latest_model_path}...")
        checkpoint = torch.load(latest_model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        history = checkpoint.get('history', [])
        start_epoch = checkpoint['epoch'] + 1
        log_print(f"  [INFO] Resuming training from epoch {start_epoch}")

    # ── Training Loop ────────────────────────────────────────────────
    log_print(f"\n  Starting training for {EPOCHS} epochs...\n")
    t_start = time.time()

    for epoch in range(start_epoch, EPOCHS + 1):
        t_epoch = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, DEVICE, scenario_name, epoch, EPOCHS)
        # Composite val loss (for logging — influenced by curriculum lambda)
        val_loss = validate(model, val_loader, criterion, DEVICE, scenario_name, epoch, EPOCHS)

        # Pure MSE val loss (Fix F2: curriculum-independent, used for early stopping)
        val_mse = validate_mse_only(model, val_loader, DEVICE)

        # CosineAnnealingLR steps every epoch (no plateau arg needed)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        elapsed = time.time() - t_epoch

        # Log
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss_composite": val_loss,   # MSE + λ×physics (for tracking)
            "val_loss_mse": val_mse,           # pure MSE (for early stopping)
            "lr": current_lr,
            "lambda_phys": min(MAX_LAMBDA, MAX_LAMBDA * (epoch / LAMBDA_WARMUP_EPOCHS)),
            "time_sec": round(elapsed, 1),
        }
        history.append(record)

        # Fix F2: best model selection uses pure MSE (not composite loss)
        improved = ""
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            patience_counter = 0
            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_mse": val_mse,
                "val_loss_composite": val_loss,
                "train_loss": train_loss,
                "config": {
                    "num_inputs": NUM_INPUTS,
                    "num_filters": NUM_FILTERS,
                    "kernel_size": KERNEL_SIZE,
                    "dropout": DROPOUT,
                    "dilation_rates": DILATION_RATES,
                    "lambda_phys": LAMBDA_PHYS,
                }
            }, best_model_path)
            improved = " ** BEST **"
        else:
            patience_counter += 1

        # Save latest model every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'history': history,
            "config": {
                "num_inputs": NUM_INPUTS,
                "num_filters": NUM_FILTERS,
                "kernel_size": KERNEL_SIZE,
                "dropout": DROPOUT,
                "dilation_rates": DILATION_RATES,
                "lambda_phys": LAMBDA_PHYS,
            }
        }, latest_model_path)

        log_print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train: {train_loss:.6f} | "
                  f"Val(MSE): {val_mse:.6f} | "
                  f"Val(+λ): {val_loss:.6f} | "
                  f"LR: {current_lr:.2e} | "
                  f"{elapsed:.1f}s{improved}")

        # Early stopping check
        if patience_counter >= PATIENCE:
            log_print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
            break

    total_time = time.time() - t_start
    log_print(f"\n  Training complete in {total_time:.1f}s "
              f"({total_time/60:.1f} min)")
    log_print(f"  Best val loss: {best_val_loss:.6f}")
    log_print(f"\n  Training completed in {total_time/60:.2f} minutes.")
    log_print(f"  Best model saved: {best_model_path}")

    # ── Save training log CSV ────────────────────────────────────────
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    log_print(f"  Training log: {log_path}")

    # Load best model for evaluation
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        log_print(f"  Loaded best weights from epoch {checkpoint.get('epoch', '?')}")

    return model, history, test_loader


# =====================================================================
# 4. Evaluation Metrics (Seq2Seq: Dual-Metric)
# =====================================================================
def evaluate_model(model, test_loader, device, scenario_name: str):
    """
    Evaluate Seq2Seq model on test set.

    Dual Metrics:
      - Full-sequence RMSE: error across all 100 timesteps
      - Last-step RMSE: error only at t=-1 (backward-compatible)
    """
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc=f"  Eval {scenario_name}",
                                      leave=False):
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)              # (B, 100, 1)

            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(y_batch.numpy())    # (B, 100)

    # y_pred_all: (N, 100, 1) -> (N, 100)
    y_pred_all = np.concatenate(all_preds, axis=0).squeeze(-1)  # (N, 100)
    y_true_all = np.concatenate(all_trues, axis=0)              # (N, 100)

    # --- Full-sequence metrics (across all 100 timesteps) ---
    errors_full = y_pred_all - y_true_all
    rmse_full = np.sqrt(np.mean(errors_full ** 2))
    mae_full = np.mean(np.abs(errors_full))
    max_err_full = np.max(np.abs(errors_full))

    # --- Last-step metrics (t=-1 only, backward-compatible) ---
    y_pred_last = y_pred_all[:, -1]
    y_true_last = y_true_all[:, -1]
    errors_last = y_pred_last - y_true_last
    rmse_last = np.sqrt(np.mean(errors_last ** 2))
    mae_last = np.mean(np.abs(errors_last))
    max_err_last = np.max(np.abs(errors_last))

    # R² scores (Fix R8 — consistent with evaluate.py and sprint44_ablation.py)
    r2_full = float(r2_score(y_true_all.flatten(), y_pred_all.flatten()))
    r2_last = float(r2_score(y_true_last, y_pred_last))

    metrics = {
        "scenario": scenario_name,
        "rmse_full": rmse_full,
        "mae_full": mae_full,
        "max_error_full": max_err_full,
        "r2_full": r2_full,
        "rmse_last": rmse_last,
        "mae_last": mae_last,
        "max_error_last": max_err_last,
        "r2_last": r2_last,
        # backward compat: 'rmse' points to full-sequence
        "rmse": rmse_full,
        "mae": mae_full,
        "max_error": max_err_full,
        "n_samples": len(y_true_all),
    }

    log_print(f"\n  Evaluation Results ({scenario_name}):")
    log_print(f"    [Full-Seq]  RMSE: {rmse_full:.6f} ({rmse_full*100:.4f}%)")
    log_print(f"    [Full-Seq]  MAE : {mae_full:.6f} ({mae_full*100:.4f}%)")
    log_print(f"    [Full-Seq]  MaxE: {max_err_full:.6f} ({max_err_full*100:.4f}%)")
    log_print(f"    [Full-Seq]  R\u00b2  : {r2_full:.6f}")
    log_print(f"    [Last-Step] RMSE: {rmse_last:.6f} ({rmse_last*100:.4f}%)")
    log_print(f"    [Last-Step] MAE : {mae_last:.6f} ({mae_last*100:.4f}%)")
    log_print(f"    [Last-Step] MaxE: {max_err_last:.6f} ({max_err_last*100:.4f}%)")
    log_print(f"    [Last-Step] R\u00b2  : {r2_last:.6f}")
    log_print(f"    Samples   : {len(y_true_all):,}")

    return metrics, y_pred_all, y_true_all


# =====================================================================
# 5. Visualization: True vs Predicted SOC (Publication Quality)
# =====================================================================
def plot_soc_predictions(y_true, y_pred, scenario_name: str,
                         save_path: str, n_points: int = 5000):
    """
    Plot True SOC vs Predicted SOC for publication.

    Parameters
    ----------
    y_true, y_pred : 1D arrays
    scenario_name : str
    save_path : str — output path for PNG
    n_points : int — number of points to plot (subsample for clarity)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Publication rcParams ─────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    # Subsample for visual clarity
    total = len(y_true)
    if total > n_points:
        idx = np.linspace(0, total - 1, n_points, dtype=int)
        y_t = y_true[idx]
        y_p = y_pred[idx]
    else:
        y_t = y_true
        y_p = y_pred
        idx = np.arange(total)

    errors = np.abs(y_t - y_p)

    # ── Create figure with 3 subplots ────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), height_ratios=[3, 1.5, 1.5])
    fig.suptitle(f"PI-TCN SOC Estimation - {scenario_name.replace('_', ' ').title()}",
                 fontweight="bold", fontsize=14)

    # --- Subplot 1: True vs Predicted ---
    ax1 = axes[0]
    ax1.plot(idx, y_t * 100, color="#2196F3", linewidth=0.8,
             alpha=0.9, label="True SOC")
    ax1.plot(idx, y_p * 100, color="#FF5722", linewidth=0.8,
             alpha=0.7, label="Predicted SOC", linestyle="--")
    ax1.set_ylabel("SOC (%)")
    ax1.set_title("True vs Predicted State of Charge")
    ax1.legend(loc="upper right")
    ax1.set_xlim(idx[0], idx[-1])

    # --- Subplot 2: Absolute Error ---
    ax2 = axes[1]
    ax2.fill_between(idx, errors * 100, color="#FF9800", alpha=0.5)
    ax2.plot(idx, errors * 100, color="#E65100", linewidth=0.5, alpha=0.8)
    ax2.set_ylabel("Abs. Error (%)")
    ax2.set_title("Absolute Prediction Error")
    ax2.set_xlim(idx[0], idx[-1])

    # Add horizontal lines for error thresholds
    rmse_val = np.sqrt(np.mean((y_t - y_p) ** 2)) * 100
    ax2.axhline(y=rmse_val, color="#D32F2F", linestyle=":", linewidth=1,
                label=f"RMSE = {rmse_val:.2f}%")
    ax2.legend(loc="upper right")

    # --- Subplot 3: Scatter plot (correlation) ---
    ax3 = axes[2]
    ax3.scatter(y_t * 100, y_p * 100, c=errors * 100, cmap="YlOrRd",
                s=1, alpha=0.5, edgecolors="none")
    ax3.plot([0, 100], [0, 100], "k--", linewidth=1, alpha=0.5,
             label="Perfect prediction")
    ax3.set_xlabel("True SOC (%)")
    ax3.set_ylabel("Predicted SOC (%)")
    ax3.set_title("Prediction Correlation")
    ax3.set_aspect("equal")
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 100)
    ax3.legend(loc="lower right")

    plt.tight_layout()

    # Save PNG and PDF
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    fig.savefig(save_path.replace(".png", ".pdf"))
    plt.close(fig)

    log_print(f"  Figure saved: {save_path}")
    log_print(f"  Figure saved: {save_path.replace('.png', '.pdf')}")


def plot_learning_curves(history_a, history_b, save_path: str):
    """
    Plot learning curves for both scenarios side by side.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PI-TCN Training Curves", fontweight="bold", fontsize=14)

    for ax, history, name in [(ax1, history_a, "Scenario A (Zero-Shot)"),
                               (ax2, history_b, "Scenario B (In-Distribution)")]:
        epochs = [h["epoch"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]

        ax.plot(epochs, train_loss, color="#1976D2", linewidth=1.5,
                label="Train Loss")
        ax.plot(epochs, val_loss, color="#D32F2F", linewidth=1.5,
                label="Val Loss")

        # Mark best epoch
        best_idx = np.argmin(val_loss)
        ax.axvline(x=epochs[best_idx], color="#4CAF50", linestyle=":",
                   alpha=0.7, label=f"Best @ epoch {epochs[best_idx]}")
        ax.scatter([epochs[best_idx]], [val_loss[best_idx]],
                   color="#4CAF50", s=50, zorder=5)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE + Physics)")
        ax.set_title(name)
        ax.legend()
        ax.set_yscale("log")

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    fig.savefig(save_path.replace(".png", ".pdf"))
    plt.close(fig)

    log_print(f"  Learning curves saved: {save_path}")


# =====================================================================
# 6. Main Execution
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="PI-TCN Train Script")
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoints')
    args = parser.parse_args()

    log_print("=" * 65)
    log_print("  Sprint 5: Seq2Seq Training + Intra-Window PVR")
    log_print("  PI-TCN SOC Estimator (Seq2Seq Architecture)")
    if args.resume:
        log_print("  [MODE] RESUMING FROM LATEST CHECKPOINTS")
    log_print("=" * 65)
    log_print(f"  Device       : {DEVICE}")
    log_print(f"  Batch size   : {BATCH_SIZE}")
    log_print(f"  Epochs       : {EPOCHS}")
    log_print(f"  LR           : {LEARNING_RATE}")
    log_print(f"  Lambda_phys  : {LAMBDA_PHYS} → {MAX_LAMBDA} (curriculum over {LAMBDA_WARMUP_EPOCHS} epochs)")
    log_print(f"  Early stop   : patience={PATIENCE}")
    log_print(f"  Random seed  : {RANDOM_SEED}")

    all_metrics = {}
    current_batch_size = BATCH_SIZE

    # ── SCENARIO A ───────────────────────────────────────────────────
    try:
        model_a, history_a, test_loader_a = train_scenario(
            "scenario_A", batch_size=current_batch_size, resume=args.resume)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            log_print(f"\n  OOM detected! Reducing batch size: "
                      f"{current_batch_size} -> {current_batch_size // 2}")
            torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
            current_batch_size //= 2
            model_a, history_a, test_loader_a = train_scenario(
                "scenario_A", batch_size=current_batch_size, resume=args.resume)
        else:
            raise

    metrics_a, y_pred_a, y_true_a = evaluate_model(
        model_a, test_loader_a, DEVICE, "scenario_A")
    all_metrics["scenario_A"] = metrics_a

    # Free memory before scenario B
    del model_a, test_loader_a
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # ── SCENARIO B ───────────────────────────────────────────────────
    try:
        model_b, history_b, test_loader_b = train_scenario(
            "scenario_B", batch_size=current_batch_size, resume=args.resume)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            log_print(f"\n  OOM detected! Reducing batch size: "
                      f"{current_batch_size} -> {current_batch_size // 2}")
            torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
            current_batch_size //= 2
            model_b, history_b, test_loader_b = train_scenario(
                "scenario_B", batch_size=current_batch_size, resume=args.resume)
        else:
            raise

    metrics_b, y_pred_b, y_true_b = evaluate_model(
        model_b, test_loader_b, DEVICE, "scenario_B")
    all_metrics["scenario_B"] = metrics_b

    del model_b, test_loader_b

    # ── VISUALIZATIONS ───────────────────────────────────────────────
    log_print(f"\n{'='*65}")
    log_print(f"  GENERATING VISUALIZATIONS")
    log_print(f"{'='*65}")

    os.makedirs(OUTPUT_FIG, exist_ok=True)

    # SOC prediction plots
    plot_soc_predictions(
        y_true_a, y_pred_a, "Scenario A (Zero-Shot)",
        os.path.join(OUTPUT_FIG, "soc_prediction_scenario_A.png"))

    plot_soc_predictions(
        y_true_b, y_pred_b, "Scenario B (In-Distribution)",
        os.path.join(OUTPUT_FIG, "soc_prediction_scenario_B.png"))

    # Learning curves
    plot_learning_curves(
        history_a, history_b,
        os.path.join(OUTPUT_FIG, "learning_curves.png"))

    # ── PVR + PER-TEMPERATURE ANALYSIS ─────────────────────────────
    log_print(f"\n{'='*65}")
    log_print(f"  PHYSICS VIOLATION RATE (PVR) ANALYSIS")
    log_print(f"{'='*65}")

    for name, y_pred, y_true, scenario_label in [
        ('scenario_A', y_pred_a, y_true_a, 'Scenario A (Zero-Shot)'),
        ('scenario_B', y_pred_b, y_true_b, 'Scenario B (In-Distribution)'),
    ]:
        # Load unscaled current for intra-window PVR
        data_dir = os.path.join(DATA_PROC, name)
        X_test_raw = np.load(os.path.join(data_dir, 'X_test.npy'))
        # Full current sequence per window: (N, 100)
        current_scaled = X_test_raw[:, :, 1]
        current_real = current_scaled * 40.0 - 20.0  # unscale

        # y_pred and y_true are (N, 100) from Seq2Seq evaluate_model
        pvr, n_viol, n_dis = compute_pvr(y_pred, current_real)
        log_print(f"\n  {scenario_label}:")
        log_print(f"    PVR = {pvr:.2f}% ({n_viol:,} violations / "
                  f"{n_dis:,} discharge steps)")

        # Per-temperature RMSE (last-step for backward compat)
        temp_labels_path = os.path.join(data_dir, 'temp_labels_test.npy')
        if os.path.exists(temp_labels_path):
            temp_labels = np.load(temp_labels_path, allow_pickle=True)
            # Use last-step predictions for per-temp breakdown
            y_pred_last = y_pred[:, -1] if y_pred.ndim == 2 else y_pred
            y_true_last = y_true[:, -1] if y_true.ndim == 2 else y_true
            if len(temp_labels) == len(y_pred_last):
                per_temp = compute_per_temp_rmse(
                    y_pred_last, y_true_last, temp_labels)
                log_print(f"    Per-Temperature RMSE (last-step):")
                for temp, m in per_temp.items():
                    log_print(f"      {temp:>8s}: RMSE={m['rmse_pct']:6.2f}%, "
                              f"MAE={m['mae_pct']:6.2f}%, N={m['n_samples']:,}")

    # ── FINAL COMPARISON TABLE ───────────────────────────────────────
    log_print(f"\n{'='*65}")
    log_print(f"  FINAL METRICS COMPARISON")
    log_print(f"{'='*65}")
    log_print(f"  {'Metric':<15} {'Scenario A (ZS)':<20} {'Scenario B (ID)':<20}")
    log_print(f"  {'-'*55}")
    log_print(f"  {'RMSE':<15} "
              f"{metrics_a['rmse']:.6f} ({metrics_a['rmse']*100:.4f}%)   "
              f"{metrics_b['rmse']:.6f} ({metrics_b['rmse']*100:.4f}%)")
    log_print(f"  {'MAE':<15} "
              f"{metrics_a['mae']:.6f} ({metrics_a['mae']*100:.4f}%)   "
              f"{metrics_b['mae']:.6f} ({metrics_b['mae']*100:.4f}%)")
    log_print(f"  {'Max Error':<15} "
              f"{metrics_a['max_error']:.6f} ({metrics_a['max_error']*100:.4f}%)   "
              f"{metrics_b['max_error']:.6f} ({metrics_b['max_error']*100:.4f}%)")
    log_print(f"  {'N samples':<15} "
              f"{metrics_a['n_samples']:<20,} {metrics_b['n_samples']:<20,}")

    log_print(f"\n  Training complete! All artifacts saved.")
    log_print(f"  Models : {OUTPUT_MOD}")
    log_print(f"  Logs   : {LOG_DIR}")
    log_print(f"  Figures: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
