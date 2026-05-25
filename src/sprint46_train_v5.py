"""
sprint46_train_v5.py — Sprint 46: Hard-Coulomb LSTM (V5) Training
=================================================================
Trains HardCoulombLSTM on V3 preprocessed data with Coulomb-bounded
delta constraint. All artifacts use '_v5_coulomb_' safe namespace.

Usage: python src/sprint46_train_v5.py [--resume] [--eval-only]
"""

import os
import sys
import time
import csv
import json
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
    NUM_INPUTS, BATCH_SIZE, LEARNING_RATE, EPOCHS, RANDOM_SEED,
)
from model_v5_coulomb import HardCoulombLSTM, count_parameters
from evaluate import compute_pvr, compute_per_temp_rmse

# =====================================================================
# 0. Reproducibility & Device
# =====================================================================
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10

# V5 hyperparams
HIDDEN_SIZE    = 64
NUM_LAYERS     = 2
DROPOUT        = 0.2
SAFETY_FACTOR  = 1.5   # Coulomb envelope headroom


def log_print(msg: str):
    print(msg, flush=True)


# =====================================================================
# 1. Data Loading (reuses v3 preprocessed data)
# =====================================================================
def load_scenario(scenario_name: str):
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


def make_dataloaders(data: dict, batch_size: int):
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
# 2. Training Loop — Pure MSE
# =====================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, tag, epoch, total_epochs):
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc=f"  [{tag}] Ep {epoch:3d}/{total_epochs} Train",
                leave=False, dynamic_ncols=True)
    for X_b, y_b, I_b in pbar:
        X_b = X_b.to(device)
        y_t = y_b.unsqueeze(-1).to(device)
        I_b = I_b.to(device)
        optimizer.zero_grad()
        y_p = model(X_b, I_b)
        loss = criterion(y_p, y_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item(); n += 1
        pbar.set_postfix(loss=f"{loss.item():.6f}")
    return total_loss / max(n, 1)


def validate(model, loader, criterion, device, tag, epoch, total_epochs):
    model.eval()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc=f"  [{tag}] Ep {epoch:3d}/{total_epochs} Val  ",
                leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for X_b, y_b, I_b in pbar:
            X_b = X_b.to(device)
            y_t = y_b.unsqueeze(-1).to(device)
            I_b = I_b.to(device)
            y_p = model(X_b, I_b)
            loss = criterion(y_p, y_t)
            total_loss += loss.item(); n += 1
            pbar.set_postfix(val_loss=f"{loss.item():.6f}")
    return total_loss / max(n, 1)


# =====================================================================
# 3. Full Training Pipeline
# =====================================================================
def train_scenario(scenario_name: str, batch_size: int = BATCH_SIZE,
                   resume: bool = False):
    log_print(f"\n{'='*65}")
    log_print(f"  TRAINING V5-COULOMB: {scenario_name}")
    log_print(f"  Hard-Coulomb LSTM | Magnitude-Bounded | Pure MSE")
    log_print(f"{'='*65}")

    data = load_scenario(scenario_name)
    train_loader, val_loader, test_loader = make_dataloaders(data, batch_size)
    del data

    log_print(f"  Batches: train={len(train_loader)}, "
              f"val={len(val_loader)}, test={len(test_loader)}")
    log_print(f"  Device: {DEVICE}")

    model = HardCoulombLSTM(
        num_inputs=NUM_INPUTS, hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS, dropout=DROPOUT,
        safety_factor=SAFETY_FACTOR,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)

    log_print(f"  Parameters: {count_parameters(model):,}")
    log_print(f"  Gamma (SOC/A/s): {model.hard_constraint.gamma:.6e}")
    log_print(f"  Safety factor: {SAFETY_FACTOR}")
    log_print(f"  Loss: Pure MSELoss")
    log_print(f"  Scheduler: CosineAnnealingLR (T_max={EPOCHS})")
    log_print(f"  Early Stopping: patience={PATIENCE}")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MOD, exist_ok=True)

    log_path    = os.path.join(LOG_DIR,    f"training_log_v5_coulomb_{scenario_name}.csv")
    best_path   = os.path.join(OUTPUT_MOD, f"hybrid_v5_coulomb_{scenario_name}.pt")
    latest_path = os.path.join(OUTPUT_MOD, f"hybrid_v5_coulomb_{scenario_name}_latest.pt")

    history = []
    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 1

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

    log_print(f"\n  Starting training for {EPOCHS} epochs...\n")
    t_start = time.time()

    for epoch in range(start_epoch, EPOCHS + 1):
        t_ep = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            scenario_name, epoch, EPOCHS)
        val_loss = validate(
            model, val_loader, criterion, DEVICE,
            scenario_name, epoch, EPOCHS)

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
                    "version": "v5_coulomb", "num_inputs": NUM_INPUTS,
                    "hidden_size": HIDDEN_SIZE, "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT, "safety_factor": SAFETY_FACTOR,
                }
            }, best_path)
            improved = " ** BEST **"
        else:
            patience_counter += 1

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

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    log_print(f"  Log saved: {log_path}")

    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        log_print(f"  Loaded best weights from epoch {ckpt.get('epoch', '?')}")

    return model, history, test_loader


# =====================================================================
# 4. Evaluation
# =====================================================================
def evaluate_model(model, test_loader, device, scenario_name: str):
    model.eval()
    all_preds, all_trues, all_currents = [], [], []
    with torch.no_grad():
        for X_b, y_b, I_b in tqdm(test_loader, desc=f"  Eval {scenario_name}", leave=False):
            X_b = X_b.to(device); I_b = I_b.to(device)
            y_p = model(X_b, I_b)
            all_preds.append(y_p.cpu().numpy())
            all_trues.append(y_b.numpy())
            all_currents.append(I_b.cpu().numpy())

    yp = np.concatenate(all_preds, axis=0).squeeze(-1)
    yt = np.concatenate(all_trues, axis=0)
    Ia = np.concatenate(all_currents, axis=0)

    err = yp - yt
    rmse_full = np.sqrt(np.mean(err**2))
    mae_full  = np.mean(np.abs(err))
    maxe_full = np.max(np.abs(err))
    r2_full   = float(r2_score(yt.flatten(), yp.flatten()))

    rmse_last = np.sqrt(np.mean((yp[:, -1] - yt[:, -1])**2))
    r2_last   = float(r2_score(yt[:, -1], yp[:, -1]))

    pvr, n_viol, n_dis = compute_pvr(yp, Ia)

    metrics = {
        "scenario": scenario_name,
        "rmse_full_pct": rmse_full * 100, "mae_full_pct": mae_full * 100,
        "max_error_full_pct": maxe_full * 100, "r2_full": r2_full,
        "rmse_last_pct": rmse_last * 100, "r2_last": r2_last,
        "pvr": pvr, "pvr_violations": n_viol, "pvr_discharge_steps": n_dis,
        "n_samples": len(yt),
    }

    log_print(f"\n  Results ({scenario_name}):")
    log_print(f"    [Full-Seq]  RMSE: {rmse_full*100:.4f}%  R2={r2_full:.6f}")
    log_print(f"    [Full-Seq]  MAE : {mae_full*100:.4f}%")
    log_print(f"    [Full-Seq]  MaxE: {maxe_full*100:.4f}%")
    log_print(f"    [Last-Step] RMSE: {rmse_last*100:.4f}%  R2={r2_last:.6f}")
    log_print(f"    PVR: {pvr:.2f}% ({n_viol:,} / {n_dis:,})")

    return metrics, yp, yt, Ia


# =====================================================================
# 5. Plotting (safe filenames)
# =====================================================================
def plot_soc(y_true, y_pred, scenario_name, save_path, n_points=5000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
        "axes.titlesize": 12, "legend.fontsize": 9, "figure.dpi": 300,
        "savefig.dpi": 300, "savefig.bbox": "tight", "axes.grid": True,
        "grid.alpha": 0.25, "lines.linewidth": 0.9,
    })

    yt1d = np.ravel(y_true); yp1d = np.ravel(y_pred)
    total = len(yt1d)
    if total > n_points:
        idx = np.linspace(0, total - 1, n_points, dtype=int)
        y_t, y_p = yt1d[idx], yp1d[idx]
    else:
        idx = np.arange(total); y_t, y_p = yt1d, yp1d

    errors = np.abs(y_t - y_p)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), height_ratios=[3, 1.5, 1.5])
    fig.suptitle(f"Hard-Coulomb LSTM (v5) -- {scenario_name}",
                 fontweight="bold", fontsize=13)

    axes[0].plot(idx, y_t*100, color="#1565C0", lw=0.8, alpha=0.9, label="True SOC")
    axes[0].plot(idx, y_p*100, color="#E65100", lw=0.8, alpha=0.75,
                 label="Predicted SOC", linestyle="--")
    axes[0].set_ylabel("SOC (%)"); axes[0].set_xlabel("Time Step Index")
    axes[0].legend(loc="upper right", framealpha=0.9)

    rmse_val = np.sqrt(np.mean((y_t - y_p)**2)) * 100
    axes[1].fill_between(idx, errors*100, color="#FF9800", alpha=0.45)
    axes[1].axhline(y=rmse_val, color="#C62828", ls=":", lw=1.0,
                    label=f"RMSE = {rmse_val:.2f}%")
    axes[1].set_ylabel("Abs. Error (%)"); axes[1].set_xlabel("Time Step Index")
    axes[1].legend(loc="upper right", framealpha=0.9)

    sc = axes[2].scatter(y_t*100, y_p*100, c=errors*100, cmap="YlOrRd",
                         vmin=0, vmax=60, s=1, alpha=0.5, edgecolors="none",
                         rasterized=True)
    axes[2].plot([0,100],[0,100], "k--", lw=0.8, alpha=0.5, label="Ideal (y=x)")
    axes[2].set_xlabel("True SOC (%)"); axes[2].set_ylabel("Predicted SOC (%)")
    axes[2].set_aspect("equal"); axes[2].set_xlim(0,100); axes[2].set_ylim(0,100)
    axes[2].legend(loc="lower right", framealpha=0.9)
    fig.colorbar(sc, ax=axes[2], shrink=0.6, pad=0.02).set_label("Abs. Error (%)", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    fig.savefig(save_path.replace(".png", ".pdf"))
    plt.close(fig)
    log_print(f"  Figure saved: {save_path}")


# =====================================================================
# 6. Eval-Only Pipeline
# =====================================================================
def run_eval_and_plot(scenario_name: str, device):
    log_print(f"\n{'='*65}")
    log_print(f"  EVAL-ONLY V5-COULOMB: {scenario_name}")
    log_print(f"{'='*65}")

    best_path = os.path.join(OUTPUT_MOD, f"hybrid_v5_coulomb_{scenario_name}.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"No checkpoint: {best_path}")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    model = HardCoulombLSTM(
        num_inputs=cfg.get('num_inputs', NUM_INPUTS),
        hidden_size=cfg.get('hidden_size', HIDDEN_SIZE),
        num_layers=cfg.get('num_layers', NUM_LAYERS),
        dropout=cfg.get('dropout', DROPOUT),
        safety_factor=cfg.get('safety_factor', SAFETY_FACTOR),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    log_print(f"  Loaded best weights from epoch {ckpt.get('epoch', '?')}")

    d = os.path.join(DATA_PROC, f"v3_{scenario_name}")
    X = torch.from_numpy(np.load(os.path.join(d, "X_test.npy")).astype(np.float32))
    y = torch.from_numpy(np.load(os.path.join(d, "y_test.npy")).astype(np.float32))
    I = torch.from_numpy(np.load(os.path.join(d, "I_unscaled_test.npy")).astype(np.float32))
    test_loader = DataLoader(TensorDataset(X, y, I), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    metrics, yp, yt, Ia = evaluate_model(model, test_loader, device, scenario_name)

    # Per-temp RMSE
    tl_path = os.path.join(d, "temp_labels_test.npy")
    if os.path.exists(tl_path):
        temp_labels = np.load(tl_path, allow_pickle=True)
        yp_last = yp[:, -1]; yt_last = yt[:, -1]
        if len(temp_labels) == len(yp_last):
            per_temp = compute_per_temp_rmse(yp_last, yt_last, temp_labels)
            log_print(f"    Per-Temperature RMSE (last-step):")
            for t, m in per_temp.items():
                log_print(f"      {t:>8s}: RMSE={m['rmse_pct']:6.2f}%  N={m['n_samples']:,}")

    os.makedirs(OUTPUT_FIG, exist_ok=True)
    plot_soc(yt, yp, scenario_name.replace('_', ' ').title(),
             os.path.join(OUTPUT_FIG, f"v5_coulomb_soc_{scenario_name}.png"))

    del model, test_loader
    return metrics, yp, yt


# =====================================================================
# 7. Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Sprint 46: Hard-Coulomb LSTM (v5)")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    log_print("=" * 65)
    log_print("  Sprint 46: Hard-Coulomb LSTM (V5)")
    log_print("  Coulomb-bounded delta + Hard Constraint | Pure MSE")
    if args.eval_only:
        log_print("  [MODE] EVAL-ONLY")
    log_print("=" * 65)
    log_print(f"  Device: {DEVICE}  |  Batch: {BATCH_SIZE}  |  Epochs: {EPOCHS}")
    log_print(f"  Safety factor: {SAFETY_FACTOR}  |  "
              f"Gamma: {1.0/(3.0*3600):.6e} SOC/A/s")

    if args.eval_only:
        ma, _, _ = run_eval_and_plot("scenario_A", DEVICE)
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
        mb, _, _ = run_eval_and_plot("scenario_B", DEVICE)

        log_print(f"\n{'='*65}")
        log_print(f"  V5-COULOMB FINAL COMPARISON (eval-only)")
        log_print(f"{'='*65}")
        log_print(f"  {'Metric':<18} {'Scenario A':<18} {'Scenario B':<18}")
        log_print(f"  {'-'*54}")
        for key in ['rmse_full_pct','mae_full_pct','max_error_full_pct','pvr','r2_full']:
            va, vb = ma.get(key, 0), mb.get(key, 0)
            fmt = ".4f" if 'r2' in key else ".2f"
            log_print(f"  {key:<18} {va:<18{fmt}} {vb:<18{fmt}}")
        return

    # ── FULL TRAINING ──
    all_metrics = {}
    bs = BATCH_SIZE

    for scenario in ["scenario_A", "scenario_B"]:
        try:
            model, hist, test_loader = train_scenario(scenario, bs, args.resume)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache(); bs //= 2
                log_print(f"  OOM! Reducing batch to {bs}")
                model, hist, test_loader = train_scenario(scenario, bs, args.resume)
            else:
                raise

        metrics, yp, yt, Ia = evaluate_model(model, test_loader, DEVICE, scenario)
        all_metrics[scenario] = metrics

        tl_path = os.path.join(DATA_PROC, f"v3_{scenario}", "temp_labels_test.npy")
        if os.path.exists(tl_path):
            tl = np.load(tl_path, allow_pickle=True)
            yp_last = yp[:, -1]; yt_last = yt[:, -1]
            if len(tl) == len(yp_last):
                pt = compute_per_temp_rmse(yp_last, yt_last, tl)
                log_print(f"    Per-Temperature RMSE:")
                for t, m in pt.items():
                    log_print(f"      {t:>8s}: RMSE={m['rmse_pct']:6.2f}%  N={m['n_samples']:,}")

        os.makedirs(OUTPUT_FIG, exist_ok=True)
        plot_soc(yt, yp, scenario.replace('_', ' ').title(),
                 os.path.join(OUTPUT_FIG, f"v5_coulomb_soc_{scenario}.png"))

        del model, test_loader
        if DEVICE.type == "cuda": torch.cuda.empty_cache()

    results_path = os.path.join(LOG_DIR, "sprint46_results_v5_coulomb.json")
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    log_print(f"\n  Results JSON: {results_path}")

    log_print(f"\n{'='*65}")
    log_print(f"  V5-COULOMB FINAL COMPARISON")
    log_print(f"{'='*65}")
    log_print(f"  {'Metric':<18} {'Scenario A':<18} {'Scenario B':<18}")
    log_print(f"  {'-'*54}")
    ma = all_metrics.get("scenario_A", {})
    mb = all_metrics.get("scenario_B", {})
    for key in ['rmse_full_pct','mae_full_pct','max_error_full_pct','pvr','r2_full']:
        va, vb = ma.get(key, 0), mb.get(key, 0)
        fmt = ".4f" if 'r2' in key else ".2f"
        log_print(f"  {key:<18} {va:<18{fmt}} {vb:<18{fmt}}")

    log_print(f"\n  All V5-Coulomb artifacts saved (V3/V4 untouched).")
    log_print(f"  Models : {OUTPUT_MOD}")
    log_print(f"  Logs   : {LOG_DIR}")
    log_print(f"  Figures: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
