"""
train_v3.py — Hybrid Physics-ML Training Pipeline
====================================================================
Pure MSE training with Hard-Constraint model (no physics loss, no
curriculum lambda). The Hard Constraint layer structurally guarantees
PVR = 0%, so no penalty term is needed.

Key differences from train.py (v2):
  1. Loads I_unscaled arrays for Hard Constraint forward pass
  2. Uses TCN_SOC_V3 model with dual-head (delta + anchor)
  3. Loss = MSE only (single-objective, no gradient collision)
  4. No lambda scheduling, no curriculum
  5. Loads from v3_scenario_{A,B} directories

Created : 2026-05-16
Updated : 2026-05-16 — fix plot 1D squeeze, add --eval-only mode
Usage   : python src/train_v3.py [--resume] [--eval-only]
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_PROC, OUTPUT_MOD, OUTPUT_FIG, LOG_DIR,
    NUM_INPUTS, NUM_FILTERS, KERNEL_SIZE, DROPOUT, DILATION_RATES,
    BATCH_SIZE, LEARNING_RATE, EPOCHS, RANDOM_SEED,
)
from model_v3 import TCN_SOC_V3, count_parameters
from evaluate import compute_pvr, compute_per_temp_rmse

# =====================================================================
# 0. Reproducibility & Device
# =====================================================================
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10


def log_print(msg: str):
    print(msg, flush=True)


# =====================================================================
# 1. Data Loading (v3: includes I_unscaled)
# =====================================================================
def load_scenario_v3(scenario_name: str):
    """Load v3 preprocessed data with I_unscaled arrays."""
    d = os.path.join(DATA_PROC, f"v3_{scenario_name}")
    log_print(f"\n  Loading v3 {scenario_name} from {d} ...")

    data = {}
    for split in ["train", "val", "test"]:
        data[f"X_{split}"] = np.load(os.path.join(d, f"X_{split}.npy")).astype(np.float32)
        data[f"y_{split}"] = np.load(os.path.join(d, f"y_{split}.npy")).astype(np.float32)
        data[f"I_{split}"] = np.load(os.path.join(d, f"I_unscaled_{split}.npy")).astype(np.float32)

        log_print(f"    {split}: X={data[f'X_{split}'].shape}, "
                  f"y={data[f'y_{split}'].shape}, "
                  f"I={data[f'I_{split}'].shape}")
    return data


def make_dataloaders_v3(data: dict, batch_size: int):
    """Create DataLoaders with (X, y, I_unscaled) triplets."""
    def _make(split, shuffle):
        X = torch.from_numpy(data[f"X_{split}"])
        y = torch.from_numpy(data[f"y_{split}"])
        I = torch.from_numpy(data[f"I_{split}"])
        ds = TensorDataset(X, y, I)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=(DEVICE.type == "cuda"),
                          drop_last=False)

    return _make("train", True), _make("val", False), _make("test", False)


# =====================================================================
# 2. Training Loop — Pure MSE (no physics penalty)
# =====================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, tag, epoch, total_epochs):
    """Train for one epoch. Returns average MSE loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"  [{tag}] Ep {epoch}/{total_epochs} Train",
                leave=False, dynamic_ncols=True)
    for X_batch, y_batch, I_batch in pbar:
        X_batch = X_batch.to(device)
        y_target = y_batch.unsqueeze(-1).to(device)  # (B, 100, 1)
        I_batch = I_batch.to(device)                 # (B, 100)

        optimizer.zero_grad()
        y_pred = model(X_batch, I_batch)             # (B, 100, 1)
        loss = criterion(y_pred, y_target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / max(n_batches, 1)


def validate_v3(model, loader, criterion, device):
    """Validate model. Returns average MSE loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for X_batch, y_batch, I_batch in loader:
            X_batch = X_batch.to(device)
            y_target = y_batch.unsqueeze(-1).to(device)
            I_batch = I_batch.to(device)

            y_pred = model(X_batch, I_batch)
            loss = criterion(y_pred, y_target)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# =====================================================================
# 3. Full Training Pipeline
# =====================================================================
def train_scenario_v3(scenario_name: str, batch_size: int = BATCH_SIZE,
                      resume: bool = False):
    log_print(f"\n{'='*65}")
    log_print(f"  TRAINING v3: {scenario_name}")
    log_print(f"  Hybrid Physics-ML | Hard Constraint | Pure MSE")
    log_print(f"{'='*65}")

    data = load_scenario_v3(scenario_name)
    train_loader, val_loader, test_loader = make_dataloaders_v3(data, batch_size)
    del data

    log_print(f"  Batches: train={len(train_loader)}, "
              f"val={len(val_loader)}, test={len(test_loader)}")
    log_print(f"  Device: {DEVICE}")

    # Model
    model = TCN_SOC_V3(
        num_inputs=NUM_INPUTS, num_filters=NUM_FILTERS,
        kernel_size=KERNEL_SIZE, dropout=DROPOUT,
        dilation_rates=DILATION_RATES,
    ).to(DEVICE)

    # Pure MSE — no physics penalty needed
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)

    log_print(f"  Parameters: {count_parameters(model):,}")
    log_print(f"  Loss: Pure MSELoss (Hard Constraint handles physics)")
    log_print(f"  Scheduler: CosineAnnealingLR (T_max={EPOCHS})")
    log_print(f"  Early Stopping: patience={PATIENCE}")

    # Logging
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MOD, exist_ok=True)

    log_path = os.path.join(LOG_DIR, f"training_log_v3_{scenario_name}.csv")
    best_path = os.path.join(OUTPUT_MOD, f"hybrid_tcn_v3_{scenario_name}.pt")
    latest_path = os.path.join(OUTPUT_MOD, f"hybrid_tcn_v3_{scenario_name}_latest.pt")

    history = []
    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 1

    # Resume
    if resume and os.path.exists(latest_path):
        log_print(f"  [RESUME] Loading {latest_path}...")
        ckpt = torch.load(latest_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        history = ckpt.get('history', [])
        start_epoch = ckpt['epoch'] + 1
        log_print(f"  [RESUME] Continuing from epoch {start_epoch}")

    # Training loop
    log_print(f"\n  Starting training for {EPOCHS} epochs...\n")
    t_start = time.time()

    for epoch in range(start_epoch, EPOCHS + 1):
        t_ep = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            scenario_name, epoch, EPOCHS)
        val_loss = validate_v3(model, val_loader, criterion, DEVICE)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t_ep

        record = {"epoch": epoch, "train_loss": train_loss,
                  "val_loss": val_loss, "lr": lr, "time_sec": round(elapsed, 1)}
        history.append(record)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss, "train_loss": train_loss,
                "config": {
                    "version": "v3",
                    "num_inputs": NUM_INPUTS, "num_filters": NUM_FILTERS,
                    "kernel_size": KERNEL_SIZE, "dropout": DROPOUT,
                    "dilation_rates": DILATION_RATES,
                }
            }, best_path)
            improved = " ** BEST **"
        else:
            patience_counter += 1

        # Save latest
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss, 'history': history,
        }, latest_path)

        log_print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {lr:.2e} | {elapsed:.1f}s{improved}")

        if patience_counter >= PATIENCE:
            log_print(f"\n  Early stopping at epoch {epoch}")
            break

    total_time = time.time() - t_start
    log_print(f"\n  Training complete in {total_time/60:.1f} min")
    log_print(f"  Best val loss: {best_val_loss:.6f}")

    # Save log CSV
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    # Load best model
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        log_print(f"  Loaded best weights from epoch {ckpt.get('epoch', '?')}")

    return model, history, test_loader


# =====================================================================
# 4. Evaluation
# =====================================================================
def evaluate_model_v3(model, test_loader, device, scenario_name: str):
    """Evaluate v3 model. Returns metrics + predictions."""
    model.eval()
    all_preds, all_trues, all_currents = [], [], []

    with torch.no_grad():
        for X_batch, y_batch, I_batch in tqdm(
                test_loader, desc=f"  Eval {scenario_name}", leave=False):
            X_batch = X_batch.to(device)
            I_batch = I_batch.to(device)
            y_pred = model(X_batch, I_batch)

            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(y_batch.numpy())
            all_currents.append(I_batch.cpu().numpy())

    y_pred_all = np.concatenate(all_preds, axis=0).squeeze(-1)  # (N, 100)
    y_true_all = np.concatenate(all_trues, axis=0)              # (N, 100)
    I_all = np.concatenate(all_currents, axis=0)                # (N, 100)

    # Full-sequence metrics
    err = y_pred_all - y_true_all
    rmse_full = np.sqrt(np.mean(err ** 2))
    mae_full = np.mean(np.abs(err))
    max_err_full = np.max(np.abs(err))
    r2_full = float(r2_score(y_true_all.flatten(), y_pred_all.flatten()))

    # Last-step metrics
    rmse_last = np.sqrt(np.mean((y_pred_all[:, -1] - y_true_all[:, -1]) ** 2))
    r2_last = float(r2_score(y_true_all[:, -1], y_pred_all[:, -1]))

    # PVR (should be 0.00%)
    pvr, n_viol, n_dis = compute_pvr(y_pred_all, I_all)

    metrics = {
        "scenario": scenario_name,
        "rmse_full_pct": rmse_full * 100,
        "mae_full_pct": mae_full * 100,
        "max_error_full_pct": max_err_full * 100,
        "r2_full": r2_full,
        "rmse_last_pct": rmse_last * 100,
        "r2_last": r2_last,
        "pvr": pvr,
        "pvr_violations": n_viol,
        "pvr_discharge_steps": n_dis,
        "n_samples": len(y_true_all),
    }

    log_print(f"\n  Results ({scenario_name}):")
    log_print(f"    [Full-Seq]  RMSE: {rmse_full*100:.4f}%  R2={r2_full:.6f}")
    log_print(f"    [Full-Seq]  MAE : {mae_full*100:.4f}%")
    log_print(f"    [Full-Seq]  MaxE: {max_err_full*100:.4f}%")
    log_print(f"    [Last-Step] RMSE: {rmse_last*100:.4f}%  R2={r2_last:.6f}")
    log_print(f"    PVR: {pvr:.2f}% ({n_viol:,} / {n_dis:,})")

    return metrics, y_pred_all, y_true_all, I_all


# =====================================================================
# 5. Visualization — IEEE Publication-Ready
# =====================================================================
def plot_soc_v3(y_true, y_pred, scenario_name, save_path, n_points=5000,
                temp_labels=None, scenario_tag=None):
    """IEEE publication-quality SOC prediction plot with zoomed inset.

    Handles both 1D (N,) and 2D (N, T) Seq2Seq arrays by flattening
    before passing to matplotlib.

    Parameters
    ----------
    y_true, y_pred : ndarray (N, T) or (N,)
    scenario_name  : str — title label
    save_path      : str — output PNG path (PDF auto-generated)
    n_points       : int — downsampling target for main plot
    temp_labels    : ndarray (N,) or None — per-sample temperature labels
    scenario_tag   : str or None — 'A' for OOD inset (-20C), 'B' for
                     in-distribution inset (25C). Controls inset selection.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch

    # ── IEEE-compliant rcParams ──────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 0.9,
    })

    # ── Flatten to 1D ────────────────────────────────────────────────
    is_2d = y_true.ndim == 2
    seq_len = y_true.shape[1] if is_2d else 1
    y_true_1d = np.ravel(y_true)
    y_pred_1d = np.ravel(y_pred)

    total = len(y_true_1d)
    if total > n_points:
        idx = np.linspace(0, total - 1, n_points, dtype=int)
        y_t = y_true_1d[idx]
        y_p = y_pred_1d[idx]
    else:
        idx = np.arange(total)
        y_t = y_true_1d
        y_p = y_pred_1d

    errors = np.abs(y_t - y_p)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), height_ratios=[3, 1.5, 1.5])
    fig.suptitle(f"Hybrid Physics-ML TCN v3 — {scenario_name}",
                 fontweight="bold", fontsize=13)

    # ── Subplot 0: True vs Predicted SOC ─────────────────────────────
    axes[0].plot(idx, y_t * 100, color="#1565C0", lw=0.8, alpha=0.9,
                 label="True SOC")
    axes[0].plot(idx, y_p * 100, color="#E65100", lw=0.8, alpha=0.75,
                 label="Predicted SOC", linestyle="--")
    axes[0].set_ylabel("SOC (%)")
    axes[0].set_xlabel("Time Step Index (1s sampling interval)")
    axes[0].legend(loc="upper right", framealpha=0.9, edgecolor="#CCCCCC")

    # ── Zoomed Inset: scenario-dependent ──────────────────────────────
    if temp_labels is not None and is_2d:
        if scenario_tag == 'B' and np.isin('25degC', temp_labels).any():
            _add_indist_inset(axes[0], y_true, y_pred, temp_labels, seq_len)
        elif np.isin('n20degC', temp_labels).any():
            _add_n20_inset(axes[0], y_true, y_pred, temp_labels, seq_len)

    # ── Subplot 1: Absolute Error ────────────────────────────────────
    rmse_val = np.sqrt(np.mean((y_t - y_p) ** 2)) * 100
    axes[1].fill_between(idx, errors * 100, color="#FF9800", alpha=0.45)
    axes[1].axhline(y=rmse_val, color="#C62828", ls=":", lw=1.0,
                    label=f"RMSE = {rmse_val:.2f}%")
    axes[1].set_ylabel("Abs. Error (%)")
    axes[1].set_xlabel("Time Step Index (1s sampling interval)")
    axes[1].legend(loc="upper right", framealpha=0.9, edgecolor="#CCCCCC")

    # ── Subplot 2: Scatter (True vs Predicted) ───────────────────────
    scatter_vmin, scatter_vmax = 0, 60
    actual_max_err = float(errors.max() * 100)
    log_print(f"  Scatter colorbar: vmin={scatter_vmin}, vmax={scatter_vmax}, "
              f"actual_max_error={actual_max_err:.2f}%")
    sc = axes[2].scatter(y_t * 100, y_p * 100, c=errors * 100, cmap="YlOrRd",
                         vmin=scatter_vmin, vmax=scatter_vmax,
                         s=1, alpha=0.5, edgecolors="none", rasterized=True)
    axes[2].plot([0, 100], [0, 100], "k--", lw=0.8, alpha=0.5,
                 label="Ideal (y = x)")
    axes[2].set_xlabel("True SOC (%)")
    axes[2].set_ylabel("Predicted SOC (%)")
    axes[2].set_aspect("equal")
    axes[2].set_xlim(0, 100)
    axes[2].set_ylim(0, 100)
    axes[2].legend(loc="lower right", framealpha=0.9, edgecolor="#CCCCCC")
    cbar = fig.colorbar(sc, ax=axes[2], shrink=0.6, pad=0.02)
    cbar.set_label("Abs. Error (%)", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    fig.savefig(save_path.replace(".png", ".pdf"))
    plt.close(fig)
    log_print(f"  Figure saved: {save_path}")


def _add_n20_inset(ax_main, y_true_2d, y_pred_2d, temp_labels, seq_len):
    """Add a zoomed inset showing one full −20°C discharge cycle.

    Selects the longest contiguous block of n20degC windows and renders
    the true/predicted SOC trajectory inside an inset axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    mask_n20 = (temp_labels == 'n20degC')
    n20_indices = np.where(mask_n20)[0]
    if len(n20_indices) == 0:
        return

    # Find contiguous blocks in n20_indices; pick the largest one
    diffs = np.diff(n20_indices)
    breaks = np.where(diffs > 1)[0]
    if len(breaks) == 0:
        # All contiguous
        block = n20_indices
    else:
        # Split into segments, pick longest
        segments = np.split(n20_indices, breaks + 1)
        block = max(segments, key=len)

    # Pick one full discharge cycle from the block:
    # Find where true SOC at last timestep drops to its minimum, then take
    # from the start of that descent. Use at most 600 windows to keep it
    # visually clean.
    soc_last = y_true_2d[block, -1]
    cycle_len = min(len(block), 600)
    # Find the sub-range with maximum SOC drop (most complete discharge)
    best_start = 0
    best_drop = 0
    for s in range(0, len(block) - cycle_len + 1, max(1, cycle_len // 4)):
        drop = soc_last[s] - soc_last[s + cycle_len - 1]
        if drop > best_drop:
            best_drop = drop
            best_start = s

    sel = block[best_start:best_start + cycle_len]

    # Flatten to 1D time series for plotting
    yt_inset = y_true_2d[sel].ravel() * 100
    yp_inset = y_pred_2d[sel].ravel() * 100
    t_inset = np.arange(len(yt_inset))

    # Create inset axes (bottom-left quadrant of main plot)
    ax_ins = ax_main.inset_axes([0.08, 0.08, 0.42, 0.42])  # [x0, y0, w, h]

    ax_ins.plot(t_inset, yt_inset, color="#1565C0", lw=0.7, alpha=0.9)
    ax_ins.plot(t_inset, yp_inset, color="#E65100", lw=0.7, alpha=0.8,
                linestyle="--")
    ax_ins.set_title("Zoom: -20 degC discharge cycle", fontsize=8, pad=3)
    ax_ins.set_xlabel("Step", fontsize=7)
    ax_ins.set_ylabel("SOC (%)", fontsize=7)
    ax_ins.tick_params(labelsize=6)
    ax_ins.grid(True, alpha=0.2, linewidth=0.4)

    # Annotation: temperature & OOD status
    ax_ins.text(0.97, 0.95, "T = $-$20\u00b0C (OOD)",
               transform=ax_ins.transAxes, fontsize=7, fontweight="bold",
               ha="right", va="top", color="#B71C1C",
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#B71C1C",
                         alpha=0.85, lw=0.6))

    # Style the inset border
    for spine in ax_ins.spines.values():
        spine.set_edgecolor('#666666')
        spine.set_linewidth(0.8)
    ax_ins.patch.set_alpha(0.92)


def _add_indist_inset(ax_main, y_true_2d, y_pred_2d, temp_labels, seq_len):
    """Add a zoomed inset showing one full 25 degC in-distribution discharge cycle."""
    import matplotlib.pyplot as plt

    target_temp = '25degC'
    mask = (temp_labels == target_temp)
    t_indices = np.where(mask)[0]
    if len(t_indices) == 0:
        return

    # Find longest contiguous block
    diffs = np.diff(t_indices)
    breaks = np.where(diffs > 1)[0]
    if len(breaks) == 0:
        block = t_indices
    else:
        segments = np.split(t_indices, breaks + 1)
        block = max(segments, key=len)

    # Pick sub-range with maximum SOC drop (most complete discharge)
    soc_last = y_true_2d[block, -1]
    cycle_len = min(len(block), 600)
    best_start, best_drop = 0, 0
    for s in range(0, len(block) - cycle_len + 1, max(1, cycle_len // 4)):
        drop = soc_last[s] - soc_last[s + cycle_len - 1]
        if drop > best_drop:
            best_drop = drop
            best_start = s

    sel = block[best_start:best_start + cycle_len]

    yt_inset = y_true_2d[sel].ravel() * 100
    yp_inset = y_pred_2d[sel].ravel() * 100
    t_inset = np.arange(len(yt_inset))

    ax_ins = ax_main.inset_axes([0.08, 0.08, 0.42, 0.42])

    ax_ins.plot(t_inset, yt_inset, color="#1565C0", lw=0.7, alpha=0.9)
    ax_ins.plot(t_inset, yp_inset, color="#E65100", lw=0.7, alpha=0.8,
                linestyle="--")
    ax_ins.set_title("Zoom: 25 degC discharge cycle", fontsize=8, pad=3)
    ax_ins.set_xlabel("Step", fontsize=7)
    ax_ins.set_ylabel("SOC (%)", fontsize=7)
    ax_ins.tick_params(labelsize=6)
    ax_ins.grid(True, alpha=0.2, linewidth=0.4)

    # Annotation: temperature & in-distribution status
    ax_ins.text(0.97, 0.95, "T = 25\u00b0C (In-Dist.)",
               transform=ax_ins.transAxes, fontsize=7, fontweight="bold",
               ha="right", va="top", color="#1B5E20",
               bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1B5E20",
                         alpha=0.85, lw=0.6))

    for spine in ax_ins.spines.values():
        spine.set_edgecolor('#666666')
        spine.set_linewidth(0.8)
    ax_ins.patch.set_alpha(0.92)


# =====================================================================
# 6. Main
# =====================================================================
def load_best_model_v3(scenario_name: str, device):
    """
    Load best v3 checkpoint + test data for eval-only mode.

    Returns
    -------
    model       : TCN_SOC_V3 with best weights loaded
    test_loader : DataLoader for test split
    """
    best_path = os.path.join(OUTPUT_MOD, f"hybrid_tcn_v3_{scenario_name}.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"No best checkpoint: {best_path}")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})

    model = TCN_SOC_V3(
        num_inputs=cfg.get('num_inputs', NUM_INPUTS),
        num_filters=cfg.get('num_filters', NUM_FILTERS),
        kernel_size=cfg.get('kernel_size', KERNEL_SIZE),
        dropout=cfg.get('dropout', DROPOUT),
        dilation_rates=cfg.get('dilation_rates', DILATION_RATES),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    log_print(f"  Loaded best weights from epoch {ckpt.get('epoch', '?')}")

    # Load test data only
    d = os.path.join(DATA_PROC, f"v3_{scenario_name}")
    X = torch.from_numpy(np.load(os.path.join(d, "X_test.npy")).astype(np.float32))
    y = torch.from_numpy(np.load(os.path.join(d, "y_test.npy")).astype(np.float32))
    I = torch.from_numpy(np.load(os.path.join(d, "I_unscaled_test.npy")).astype(np.float32))
    ds = TensorDataset(X, y, I)
    test_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=(device.type == "cuda"))

    return model, test_loader


def run_eval_and_plot(scenario_name: str, device):
    """
    Eval-only pipeline: load best checkpoint, evaluate, plot, per-temp RMSE,
    and extract −20°C subset metrics for ablation study.
    """
    log_print(f"\n{'='*65}")
    log_print(f"  EVAL-ONLY v3: {scenario_name}")
    log_print(f"{'='*65}")

    model, test_loader = load_best_model_v3(scenario_name, device)
    metrics, yp, yt, I_all = evaluate_model_v3(model, test_loader, device, scenario_name)

    # Load temp labels
    tl_path = os.path.join(DATA_PROC, f"v3_{scenario_name}", "temp_labels_test.npy")
    temp_labels = None
    if os.path.exists(tl_path):
        temp_labels = np.load(tl_path, allow_pickle=True)
        yp_last = yp[:, -1]; yt_last = yt[:, -1]
        if len(temp_labels) == len(yp_last):
            per_temp = compute_per_temp_rmse(yp_last, yt_last, temp_labels)
            log_print(f"    Per-Temperature RMSE (last-step):")
            for t, m in per_temp.items():
                log_print(f"      {t:>8s}: RMSE={m['rmse_pct']:6.2f}%  N={m['n_samples']:,}")

        # ── −20°C Subset Full-Sequence Metrics (Ablation) ────────────
        mask_n20 = (temp_labels == 'n20degC')
        n_n20 = int(mask_n20.sum())
        if n_n20 > 0:
            yp_n20 = yp[mask_n20]   # (N_n20, 100)
            yt_n20 = yt[mask_n20]   # (N_n20, 100)
            I_n20 = I_all[mask_n20] # (N_n20, 100)

            err_n20 = yp_n20 - yt_n20
            rmse_n20 = np.sqrt(np.mean(err_n20 ** 2))
            mae_n20 = np.mean(np.abs(err_n20))
            maxe_n20 = np.max(np.abs(err_n20))
            r2_n20 = float(r2_score(yt_n20.flatten(), yp_n20.flatten()))

            pvr_n20, nv_n20, nd_n20 = compute_pvr(yp_n20, I_n20)

            log_print(f"\n    {'='*55}")
            log_print(f"    -20degC SUBSET METRICS (N={n_n20:,}, "
                      f"predictions={n_n20 * yp_n20.shape[1]:,})")
            log_print(f"    {'='*55}")
            log_print(f"      RMSE : {rmse_n20*100:.4f}%")
            log_print(f"      MAE  : {mae_n20*100:.4f}%")
            log_print(f"      MaxE : {maxe_n20*100:.4f}%")
            log_print(f"      R2   : {r2_n20:.6f}")
            log_print(f"      PVR  : {pvr_n20:.2f}% ({nv_n20:,} / {nd_n20:,})")
            log_print(f"    {'='*55}")

            metrics['n20_rmse_pct'] = rmse_n20 * 100
            metrics['n20_mae_pct'] = mae_n20 * 100
            metrics['n20_maxe_pct'] = maxe_n20 * 100
            metrics['n20_r2'] = r2_n20
            metrics['n20_pvr'] = pvr_n20
            metrics['n20_n_samples'] = n_n20

    # Plot (pass temp_labels + scenario_tag for inset rendering)
    s_tag = 'A' if scenario_name.upper().endswith('A') else 'B'
    os.makedirs(OUTPUT_FIG, exist_ok=True)
    plot_soc_v3(yt, yp, scenario_name.replace('_', ' ').title(),
                os.path.join(OUTPUT_FIG, f"v3_soc_{scenario_name}.png"),
                temp_labels=temp_labels, scenario_tag=s_tag)

    del model, test_loader
    return metrics, yp, yt


def main():
    parser = argparse.ArgumentParser(description="Hybrid Physics-ML Train v3")
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training; load best weights, evaluate, and plot')
    args = parser.parse_args()

    log_print("=" * 65)
    log_print("  v3: Hybrid Physics-ML Training")
    log_print("  V_proxy + Hard Constraint | Pure MSE | PVR=0% guaranteed")
    if args.eval_only:
        log_print("  [MODE] EVAL-ONLY — loading saved checkpoints")
    log_print("=" * 65)
    log_print(f"  Device: {DEVICE}  |  Batch: {BATCH_SIZE}  |  Epochs: {EPOCHS}")

    # ── EVAL-ONLY MODE ───────────────────────────────────────────────
    if args.eval_only:
        metrics_a, yp_a, yt_a = run_eval_and_plot("scenario_A", DEVICE)
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        metrics_b, yp_b, yt_b = run_eval_and_plot("scenario_B", DEVICE)

        log_print(f"\n{'='*65}")
        log_print(f"  v3 FINAL COMPARISON (eval-only)")
        log_print(f"{'='*65}")
        log_print(f"  {'Metric':<18} {'Scenario A':<18} {'Scenario B':<18}")
        log_print(f"  {'-'*54}")
        for key in ['rmse_full_pct', 'mae_full_pct', 'max_error_full_pct',
                    'pvr', 'r2_full']:
            va = metrics_a.get(key, 0)
            vb = metrics_b.get(key, 0)
            fmt = ".4f" if 'r2' in key else ".2f"
            log_print(f"  {key:<18} {va:<18{fmt}} {vb:<18{fmt}}")
        log_print(f"\n  Figures: {OUTPUT_FIG}")
        return

    # ── FULL TRAINING MODE ───────────────────────────────────────────
    all_metrics = {}
    bs = BATCH_SIZE

    # ── SCENARIO A ───────────────────────────────────────────────────
    try:
        model_a, hist_a, test_a = train_scenario_v3("scenario_A", bs, args.resume)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            bs //= 2
            log_print(f"  OOM! Reducing batch to {bs}")
            model_a, hist_a, test_a = train_scenario_v3("scenario_A", bs, args.resume)
        else:
            raise

    metrics_a, yp_a, yt_a, Ia = evaluate_model_v3(model_a, test_a, DEVICE, "scenario_A")
    all_metrics["scenario_A"] = metrics_a

    # Per-temp RMSE
    data_dir_a = os.path.join(DATA_PROC, "v3_scenario_A")
    tl_path = os.path.join(data_dir_a, "temp_labels_test.npy")
    if os.path.exists(tl_path):
        temp_labels = np.load(tl_path, allow_pickle=True)
        yp_last = yp_a[:, -1]; yt_last = yt_a[:, -1]
        if len(temp_labels) == len(yp_last):
            per_temp = compute_per_temp_rmse(yp_last, yt_last, temp_labels)
            log_print(f"    Per-Temperature RMSE:")
            for t, m in per_temp.items():
                log_print(f"      {t:>8s}: RMSE={m['rmse_pct']:6.2f}%  N={m['n_samples']:,}")

    del model_a, test_a
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ── SCENARIO B ───────────────────────────────────────────────────
    try:
        model_b, hist_b, test_b = train_scenario_v3("scenario_B", bs, args.resume)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            bs //= 2
            model_b, hist_b, test_b = train_scenario_v3("scenario_B", bs, args.resume)
        else:
            raise

    metrics_b, yp_b, yt_b, Ib = evaluate_model_v3(model_b, test_b, DEVICE, "scenario_B")
    all_metrics["scenario_B"] = metrics_b
    del model_b, test_b

    # ── PLOTS ────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_FIG, exist_ok=True)
    plot_soc_v3(yt_a, yp_a, "Scenario A (Zero-Shot)",
                os.path.join(OUTPUT_FIG, "v3_soc_scenario_A.png"))
    plot_soc_v3(yt_b, yp_b, "Scenario B (In-Distribution)",
                os.path.join(OUTPUT_FIG, "v3_soc_scenario_B.png"))

    # ── COMPARISON TABLE ─────────────────────────────────────────────
    log_print(f"\n{'='*65}")
    log_print(f"  v3 FINAL COMPARISON")
    log_print(f"{'='*65}")
    log_print(f"  {'Metric':<18} {'Scenario A':<18} {'Scenario B':<18}")
    log_print(f"  {'-'*54}")
    for key in ['rmse_full_pct', 'mae_full_pct', 'pvr', 'r2_full']:
        va = metrics_a.get(key, 0)
        vb = metrics_b.get(key, 0)
        fmt = ".4f" if 'r2' in key else ".2f"
        log_print(f"  {key:<18} {va:<18{fmt}} {vb:<18{fmt}}")

    log_print(f"\n  All artifacts saved.")
    log_print(f"  Models : {OUTPUT_MOD}")
    log_print(f"  Logs   : {LOG_DIR}")
    log_print(f"  Figures: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
