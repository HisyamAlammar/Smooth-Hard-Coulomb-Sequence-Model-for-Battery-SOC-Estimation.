"""
Sprint 5 -- Seq2Seq Model Validation & Mini-Ablation Study
=====================================================
Fair comparison: LSTM vs TCN vs PI-TCN across 2 scenarios.
All models output (B, 100, 1) Seq2Seq predictions.

Rules enforced:
  - Identical features, splits, normalization for all models
  - All models produce Seq2Seq output (B, T, 1)
  - Chronological evaluation (no shuffle)
  - Intra-window PVR for physics violation measurement
  - Every metric verified and printed

Created: 2026-04-08
Updated: 2026-05-10 -- Sprint 5 (Seq2Seq pivot)
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.parametrizations import weight_norm
from tqdm import tqdm
from sklearn.metrics import r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from config import (DATA_PROC, OUTPUT_MOD, OUTPUT_FIG, LOG_DIR,
                     BATCH_SIZE, LEARNING_RATE, EPOCHS, LAMBDA_PHYS,
                     RANDOM_SEED, NUM_INPUTS, NUM_FILTERS, KERNEL_SIZE,
                     DROPOUT, DILATION_RATES)
from model import (TCN_SOC_Estimator, PhysicsInformedLoss,
                    TemporalBlock, Chomp1d, count_parameters)
from train import validate_mse_only

os.makedirs(OUTPUT_MOD, exist_ok=True)
os.makedirs(OUTPUT_FIG, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10
MAX_LAMBDA = 5.0           # Peak physics loss weight (curriculum schedule)
LAMBDA_WARMUP_EPOCHS = 30  # Epochs to ramp lambda from 0 → MAX_LAMBDA


# =====================================================================
# LSTM BASELINE MODEL
# =====================================================================
class LSTM_SOC_Estimator(nn.Module):
    """
    LSTM baseline for SOC estimation (Seq2Seq).
    Architecture matched to have similar parameter count as TCN.
    Input(B,100,5) -> LSTM(hidden=64, 2 layers) -> FC per-timestep -> Sigmoid
    Output: (B, 100, 1)
    """
    def __init__(self, num_inputs=5, hidden_size=64, num_layers=2,
                 dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_inputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, T=100, C=5)
        out, _ = self.lstm(x)       # (B, T, hidden)
        # Apply FC per-timestep (Seq2Seq)
        return self.fc(out)         # (B, T, 1)


# =====================================================================
# VANILLA TCN (no LayerNorm, no physics loss) — fair baseline
# =====================================================================
class VanillaTemporalBlock(nn.Module):
    """TCN block WITHOUT LayerNorm — standard architecture."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None)
        self.relu_out = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')

    def forward(self, x):
        out = self.dropout1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.dropout2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class VanillaTCN_SOC(nn.Module):
    """Vanilla TCN -- Seq2Seq, NO LayerNorm. Fair baseline."""
    def __init__(self, num_inputs=5, num_filters=64, kernel_size=7,
                 dropout=0.2, dilation_rates=None):
        super().__init__()
        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 8]
        layers = []
        for i, d in enumerate(dilation_rates):
            ic = num_inputs if i == 0 else num_filters
            layers.append(VanillaTemporalBlock(
                ic, num_filters, kernel_size, 1, d, dropout))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(num_filters, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        x = self.tcn(x)        # (B, filters, T)
        x = x.transpose(1, 2)  # (B, T, filters)
        return self.fc(x)      # (B, T, 1)


# =====================================================================
# DATASET PREPARATION (Sprint 4.4 splits)
# =====================================================================
def prepare_datasets_sprint44():
    """
    Generate Scenario A_44 and Scenario B_44 with Sprint 4.4 split strategy.
    A: Train 25+10+0, Test -20 only
    B: Train ALL (incl -20), Test -20 (chronological split)
    """
    from preprocessing import (build_ocv_soc_lookup, engineer_features,
                                read_csv, build_sequences, FEATURE_COLS)
    import glob

    DATA_RAW = os.path.join(BASE_DIR, "data", "raw", "LG Dataset",
                            "LG_HG2_Original_Dataset")

    all_temps = ['0degC', '10degC', '25degC', '40degC', 'n10degC', 'n20degC']
    keywords = ['udds', 'la92', 'hwfet', 'us06', 'mixed']

    temp_data = {}
    for temp in all_temps:
        temp_dir = os.path.join(DATA_RAW, temp)
        if not os.path.exists(temp_dir):
            continue
        ocv_lookup, q_actual = build_ocv_soc_lookup(temp)

        X_list, y_list = [], []
        csvs = sorted(glob.glob(os.path.join(temp_dir, '*.csv')))
        for c in csvs:
            fname = os.path.basename(c).lower()
            if not any(k in fname for k in keywords):
                continue
            try:
                df = read_csv(c)
                df, _ = engineer_features(df, q_actual, ocv_lookup)
                X, y = build_sequences(df, window=100, stride=10)
                if X is not None:
                    X_list.append(X)
                    y_list.append(y)
            except Exception:
                pass

        if X_list:
            temp_data[temp] = {
                'X': np.concatenate(X_list, axis=0),
                'y': np.concatenate(y_list, axis=0),
            }
            print(f"    [{temp}] {temp_data[temp]['X'].shape[0]:,} sequences")

    # Physics-Informed Scaling
    PHYS_MIN = np.array([2.0, -20.0, -20.0, -2.0, -20.0], dtype=np.float32)
    PHYS_MAX = np.array([4.25, 20.0, 50.0, 2.0, 20.0], dtype=np.float32)
    p_min = PHYS_MIN.reshape(1, 1, 5)
    p_rng = (PHYS_MAX - PHYS_MIN).reshape(1, 1, 5)

    for temp in temp_data:
        temp_data[temp]['X'] = (temp_data[temp]['X'] - p_min) / p_rng

    # ---- Scenario A_44: Train 25+10+0, Val from train (last 10%), Test -20 ----
    train_temps_a = ['25degC', '10degC', '0degC']
    X_train_a = np.concatenate([temp_data[t]['X'] for t in train_temps_a
                                 if t in temp_data])
    y_train_a = np.concatenate([temp_data[t]['y'] for t in train_temps_a
                                 if t in temp_data])

    # Shuffle train
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_train_a))
    X_train_a, y_train_a = X_train_a[idx], y_train_a[idx]

    # Split 90/10 for train/val
    n_val = int(0.1 * len(X_train_a))
    X_val_a = X_train_a[-n_val:]
    y_val_a = y_train_a[-n_val:]
    X_train_a = X_train_a[:-n_val]
    y_train_a = y_train_a[:-n_val]

    X_test_a = temp_data['n20degC']['X']
    y_test_a = temp_data['n20degC']['y']

    # ---- Scenario B_44: Train ALL, test -20 (chronological 70/10/20) ----
    # For -20: last 20% is test, second-to-last 10% is val contribution
    X_n20 = temp_data['n20degC']['X']
    y_n20 = temp_data['n20degC']['y']
    n_tr = int(0.7 * len(X_n20))
    n_vl = int(0.1 * len(X_n20))

    X_train_parts = [temp_data[t]['X'] for t in all_temps if t in temp_data and t != 'n20degC']
    y_train_parts = [temp_data[t]['y'] for t in all_temps if t in temp_data and t != 'n20degC']
    X_train_parts.append(X_n20[:n_tr])
    y_train_parts.append(y_n20[:n_tr])

    X_train_b = np.concatenate(X_train_parts)
    y_train_b = np.concatenate(y_train_parts)
    idx = rng.permutation(len(X_train_b))
    X_train_b, y_train_b = X_train_b[idx], y_train_b[idx]

    X_val_b = X_n20[n_tr:n_tr+n_vl]
    y_val_b = y_n20[n_tr:n_tr+n_vl]
    X_test_b = X_n20[n_tr+n_vl:]
    y_test_b = y_n20[n_tr+n_vl:]

    datasets = {
        'A': (X_train_a, y_train_a, X_val_a, y_val_a, X_test_a, y_test_a),
        'B': (X_train_b, y_train_b, X_val_b, y_val_b, X_test_b, y_test_b),
    }

    return datasets


# =====================================================================
# TRAINING FUNCTION (generic for all 3 models)
# =====================================================================
def train_model(model, train_loader, val_loader, model_name, scenario_name,
                use_physics_loss=False, seed=42, resume=False):
    """Train a model with early stopping, checkpointing, and terminal tracking."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # Fix R5: CosineAnnealingLR — immune to curriculum-lambda inflation.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)

    if use_physics_loss:
        criterion = PhysicsInformedLoss(lambda_phys=0.0)  # starts at 0; warmed up each epoch
    else:
        criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_mse': [], 'lambda_phys': []}
    best_state = None
    start_epoch = 1

    tag = f"{model_name}_{scenario_name}"
    latest_ckpt_path = os.path.join(OUTPUT_MOD, f"{tag}_latest.pt")
    best_ckpt_path = os.path.join(OUTPUT_MOD, f"{tag}_best.pt")

    # ---- Resume Logic ----
    if resume and os.path.exists(latest_ckpt_path):
        print(f"    [{tag}] Found checkpoint, loading...")
        checkpoint = torch.load(latest_ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        history = checkpoint.get('history', {'train_loss': [], 'val_loss': []})
        start_epoch = checkpoint['epoch'] + 1
        print(f"    [{tag}] Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS + 1):
        # ---- Curriculum Lambda Schedule ----
        if use_physics_loss:
            lambda_curr = min(MAX_LAMBDA, MAX_LAMBDA * (epoch / LAMBDA_WARMUP_EPOCHS))
            criterion.lambda_phys = lambda_curr
        else:
            lambda_curr = 0.0

        # ---- Train (Seq2Seq) ----
        model.train()
        train_loss = 0.0
        
        pbar_train = tqdm(train_loader, desc=f"    [{tag}] Ep {epoch}/{EPOCHS} Train", 
                          leave=False, dynamic_ncols=True)
        for X_batch, y_batch in pbar_train:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            y_target = y_batch.unsqueeze(-1)  # (B, 100) -> (B, 100, 1)
            optimizer.zero_grad()
            y_pred = model(X_batch)            # (B, 100, 1)

            if use_physics_loss:
                current_seq = X_batch[:, :, 1] * 40.0 - 20.0  # (B, 100)
                loss = criterion(y_pred, y_target, current_seq)
            else:
                loss = criterion(y_pred, y_target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            pbar_train.set_postfix(loss=f"{loss.item():.6f}", λ=f"{lambda_curr:.3f}")
            
        train_loss /= len(train_loader)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        pbar_val = tqdm(val_loader, desc=f"    [{tag}] Ep {epoch}/{EPOCHS} Val  ", 
                        leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for X_batch, y_batch in pbar_val:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                y_target = y_batch.unsqueeze(-1)
                y_pred = model(X_batch)
                if use_physics_loss:
                    current_seq = X_batch[:, :, 1] * 40.0 - 20.0
                    loss = criterion(y_pred, y_target, current_seq)
                else:
                    loss = criterion(y_pred, y_target)
                val_loss += loss.item()
                pbar_val.set_postfix(loss=f"{loss.item():.6f}", λ=f"{lambda_curr:.3f}")
        val_loss /= len(val_loader)

        # Fix F2: compute pure MSE val loss (curriculum-independent)
        # Used for early stopping and best-model selection
        val_mse = validate_mse_only(model, val_loader, DEVICE)

        # CosineAnnealingLR steps every epoch (no plateau argument)
        scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mse'].append(val_mse)
        if use_physics_loss:
            history['lambda_phys'].append(lambda_curr)

        # Fix F2: use pure MSE for best-model and patience (not composite loss)
        is_best = val_mse < best_val_loss
        if is_best:
            best_val_loss = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = "** BEST **"
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, best_ckpt_path)
        else:
            patience_counter += 1
            marker = ""

        # Print clean epoch summary
        lam_str = f" | λ={lambda_curr:.3f}" if use_physics_loss else ""
        print(f"    [{tag}] Epoch {epoch:3d}/{EPOCHS} | "
              f"Train: {train_loss:.6f} | Val(MSE): {val_mse:.6f} | "
              f"Val(+λ): {val_loss:.6f}{lam_str} {marker}",
              flush=True)
              
        # ---- Save Latest Checkpoint Every Epoch ----
        if use_physics_loss:
            history['lambda_phys'].append(lambda_curr)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'history': history
        }, latest_ckpt_path)

        if patience_counter >= PATIENCE:
            print(f"    [{tag}] Early stopping at epoch {epoch}", flush=True)
            break

    # Restore best weights for evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
    elif os.path.exists(best_ckpt_path):
        # Fallback if best_state wasn't captured in memory (e.g. resumed and immediately early stopped)
        best_ckpt = torch.load(best_ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])

    return model, history


# =====================================================================
# EVALUATION (chronological, no shuffle)
# =====================================================================
def evaluate_model(model, X_test, y_test, model_name, scenario_name):
    """Evaluate Seq2Seq model: RMSE, MAE, MaxError, intra-window PVR."""
    model.eval()
    all_preds = []

    dataset = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).float())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(DEVICE)
            y_pred = model(X_batch)              # (B, 100, 1)
            all_preds.append(y_pred.cpu().numpy())

    # (N, 100, 1) -> (N, 100)
    y_pred_all = np.concatenate(all_preds).squeeze(-1)  # (N, 100)
    y_true_all = y_test                                  # (N, 100)

    # Full-sequence metrics
    errors_full = y_pred_all - y_true_all
    rmse = np.sqrt(np.mean(errors_full**2))
    mae = np.mean(np.abs(errors_full))
    max_err = np.max(np.abs(errors_full))
    r2_full = float(r2_score(y_true_all.flatten(), y_pred_all.flatten()))

    # Last-step metrics
    errors_last = y_pred_all[:, -1] - y_true_all[:, -1]
    rmse_last = np.sqrt(np.mean(errors_last**2))
    mae_last = np.mean(np.abs(errors_last))
    max_err_last = np.max(np.abs(errors_last))
    r2_last = float(r2_score(y_true_all[:, -1], y_pred_all[:, -1]))

    # Intra-window PVR
    current_scaled = X_test[:, :, 1]              # (N, 100)
    current_real = current_scaled * 40.0 - 20.0   # unscale

    # delta_soc within each window
    dsoc_pred = y_pred_all[:, 1:] - y_pred_all[:, :-1]  # (N, 99)
    dis_mask = current_real[:, 1:] < -0.05               # (N, 99)
    n_dis = int(dis_mask.sum())
    if n_dis > 0:
        violations = int((dsoc_pred[dis_mask] > 0).sum())
        pvr = violations / n_dis * 100
    else:
        pvr = 0.0
        violations = 0

    return {
        'model': model_name,
        'scenario': scenario_name,
        'rmse': rmse,
        'rmse_pct': rmse * 100,
        'mae': mae,
        'mae_pct': mae * 100,
        'max_error': max_err,
        'max_error_pct': max_err * 100,
        'r2_full': r2_full,
        'rmse_last_pct': rmse_last * 100,
        'mae_last_pct': mae_last * 100,
        'max_error_last_pct': max_err_last * 100,
        'r2_last': r2_last,
        'pvr': pvr,
        'n_violations': violations,
        'n_discharge': n_dis,
        'n_samples': len(y_true_all),
        'y_pred': y_pred_all,
        'y_true': y_true_all,
    }


# =====================================================================
# FAILURE ANALYSIS — top 3 worst sequences
# =====================================================================
def failure_analysis(metrics, X_test):
    """Identify top 3 largest error windows and their characteristics."""
    y_pred = metrics['y_pred']   # (N, 100)
    y_true = metrics['y_true']   # (N, 100)
    # Per-window mean absolute error
    per_window_mae = np.mean(np.abs(y_pred - y_true), axis=1)  # (N,)

    top3_idx = np.argsort(per_window_mae)[-3:][::-1]

    print(f"\n    Failure Analysis ({metrics['model']}, {metrics['scenario']}):")
    for rank, idx in enumerate(top3_idx):
        # Unscale features at this index
        v_scaled = X_test[idx, :, 0]
        i_scaled = X_test[idx, :, 1]
        t_scaled = X_test[idx, :, 2]

        v_real = v_scaled * (4.25 - 2.0) + 2.0
        i_real = i_scaled * 40.0 - 20.0
        t_real = t_scaled * 70.0 - 20.0

        # Use last-step error for reporting
        last_err = abs(y_pred[idx, -1] - y_true[idx, -1])
        print(f"      #{rank+1}: idx={idx:,}")
        print(f"        Window MAE: {per_window_mae[idx]:.4f} "
              f"({per_window_mae[idx]*100:.2f}%)")
        print(f"        Last-step : err={last_err:.4f}, "
              f"true={y_true[idx, -1]:.4f}, pred={y_pred[idx, -1]:.4f}")
        print(f"        V_mean   : {v_real.mean():.3f}V")
        print(f"        I_mean   : {i_real.mean():.3f}A")
        print(f"        T_mean   : {t_real.mean():.1f}degC")
        soc_last = y_true[idx, -1]
        soc_region = "HIGH" if soc_last > 0.7 else (
            "MID" if soc_last > 0.3 else "LOW")
        print(f"        SOC region: {soc_region}")


# =====================================================================
# MAIN EXECUTION
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Sprint 5 Ablation Study")
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoints')
    args = parser.parse_args()

    print("=" * 70)
    print("  SPRINT 5 -- Seq2Seq MODEL VALIDATION & MINI-ABLATION")
    print("  LSTM vs TCN vs PI-TCN | Seq2Seq | 2 Scenarios")
    if args.resume:
        print("  [MODE] RESUMING FROM LATEST CHECKPOINTS")
    print(f"  Lambda schedule: 0.0 → {MAX_LAMBDA} over {LAMBDA_WARMUP_EPOCHS} epochs")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Seed: {RANDOM_SEED}")
    print(f"  Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}, Epochs: {EPOCHS}")
    print(f"  Lambda_phys: {LAMBDA_PHYS}")

    # ── STEP 1: Dataset Preparation ──────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  STEP 1 - DATASET PREPARATION")
    print("=" * 70)

    datasets = prepare_datasets_sprint44()

    for sc in ['A', 'B']:
        X_tr, y_tr, X_vl, y_vl, X_te, y_te = datasets[sc]
        print(f"\n  Scenario {sc}:")
        print(f"    Train: {X_tr.shape}, y: [{y_tr.min():.4f}, {y_tr.max():.4f}]")
        print(f"    Val:   {X_vl.shape}")
        print(f"    Test:  {X_te.shape}, y: [{y_te.min():.4f}, {y_te.max():.4f}]")

    # ── STEP 2: Build Models ─────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  STEP 2 - MODEL CONSTRUCTION")
    print("=" * 70)

    def build_models(seed=42):
        torch.manual_seed(seed)
        models = {
            'LSTM': LSTM_SOC_Estimator(
                num_inputs=NUM_INPUTS, hidden_size=NUM_FILTERS,
                num_layers=2, dropout=DROPOUT).to(DEVICE),
            'TCN': VanillaTCN_SOC(
                num_inputs=NUM_INPUTS, num_filters=NUM_FILTERS,
                kernel_size=KERNEL_SIZE, dropout=DROPOUT,
                dilation_rates=DILATION_RATES).to(DEVICE),
            'PI-TCN': TCN_SOC_Estimator(
                num_inputs=NUM_INPUTS, num_filters=NUM_FILTERS,
                kernel_size=KERNEL_SIZE, dropout=DROPOUT,
                dilation_rates=DILATION_RATES).to(DEVICE),
        }
        return models

    models_check = build_models()
    for name, model in models_check.items():
        n_params = count_parameters(model)
        print(f"  {name:>8s}: {n_params:,} parameters")
    del models_check

    # ── STEP 3 & 4: Train + Evaluate All Combinations ────────────────
    print(f"\n{'=' * 70}")
    print("  STEPS 3-4: TRAINING & EVALUATION")
    print("=" * 70)

    all_results = []

    for scenario in ['A', 'B']:
        X_tr, y_tr, X_vl, y_vl, X_te, y_te = datasets[scenario]

        train_ds = TensorDataset(
            torch.from_numpy(X_tr).float(),
            torch.from_numpy(y_tr).float())
        val_ds = TensorDataset(
            torch.from_numpy(X_vl).float(),
            torch.from_numpy(y_vl).float())

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=0,
                                   pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=0,
                                 pin_memory=True)

        for model_name in ['LSTM', 'TCN', 'PI-TCN']:
            print(f"\n  --- {model_name} / Scenario {scenario} ---", flush=True)
            t0 = time.time()

            # Fresh model for each experiment
            models = build_models(RANDOM_SEED)
            model = models[model_name]

            use_phys = (model_name == 'PI-TCN')

            model, history = train_model(
                model, train_loader, val_loader,
                model_name, f"Scenario_{scenario}",
                use_physics_loss=use_phys, seed=RANDOM_SEED, 
                resume=args.resume)

            elapsed = time.time() - t0
            print(f"    Training time: {elapsed:.1f}s ({elapsed/60:.1f}min)", flush=True)

            # Evaluate (chronological, no shuffle)
            metrics = evaluate_model(model, X_te, y_te, model_name,
                                      f"Scenario_{scenario}")

            print(f"    RMSE: {metrics['rmse_pct']:.4f}%  R²(full)={metrics['r2_full']:.6f}")
            print(f"    MAE:  {metrics['mae_pct']:.4f}%")
            print(f"    MaxE: {metrics['max_error_pct']:.4f}%")
            print(f"    Last: RMSE={metrics['rmse_last_pct']:.4f}%  R²={metrics['r2_last']:.6f}")
            print(f"    PVR:  {metrics['pvr']:.2f}% "
                  f"({metrics['n_violations']:,}/{metrics['n_discharge']:,})", flush=True)

            # Failure analysis
            failure_analysis(metrics, X_te)

            # Save metrics (without large arrays)
            result_row = {k: v for k, v in metrics.items()
                          if k not in ('y_pred', 'y_true')}
            all_results.append(result_row)

            # Free GPU memory
            del model, models
            torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # ── STEP 5: Reproducibility (3 seeds for best model) ─────────────
    print(f"\n{'=' * 70}")
    print("  STEP 5 - REPRODUCIBILITY (PI-TCN, 3 Seeds)")
    print("=" * 70)

    seeds = [42, 123, 2026]
    repro_results = {'A': [], 'B': []}

    for scenario in ['A', 'B']:
        X_tr, y_tr, X_vl, y_vl, X_te, y_te = datasets[scenario]

        train_ds = TensorDataset(
            torch.from_numpy(X_tr).float(),
            torch.from_numpy(y_tr).float())
        val_ds = TensorDataset(
            torch.from_numpy(X_vl).float(),
            torch.from_numpy(y_vl).float())
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=0, pin_memory=True)

        for seed in seeds:
            print(f"\n  PI-TCN / Scenario {scenario} / Seed {seed}", flush=True)
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = TCN_SOC_Estimator(
                num_inputs=NUM_INPUTS, num_filters=NUM_FILTERS,
                kernel_size=KERNEL_SIZE, dropout=DROPOUT,
                dilation_rates=DILATION_RATES).to(DEVICE)

            model, _ = train_model(
                model, train_loader, val_loader,
                'PI-TCN', f'Scenario_{scenario}_Seed_{seed}',
                use_physics_loss=True, seed=seed,
                resume=args.resume)

            metrics = evaluate_model(model, X_te, y_te, 'PI-TCN',
                                      f'Scenario_{scenario}')

            print(f"    RMSE: {metrics['rmse_pct']:.4f}%, "
                  f"PVR: {metrics['pvr']:.2f}%", flush=True)

            repro_results[scenario].append({
                'seed': seed,
                'rmse': metrics['rmse_pct'],
                'pvr': metrics['pvr'],
            })

            del model
            torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    # Print reproducibility summary
    for scenario in ['A', 'B']:
        rmses = [r['rmse'] for r in repro_results[scenario]]
        pvrs = [r['pvr'] for r in repro_results[scenario]]
        print(f"\n  Reproducibility Scenario {scenario} (PI-TCN):")
        print(f"    RMSE: {np.mean(rmses):.4f}% +/- {np.std(rmses):.4f}%")
        print(f"    PVR:  {np.mean(pvrs):.2f}% +/- {np.std(pvrs):.2f}%")

    # ── RESULTS TABLE ────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  FINAL RESULTS TABLE")
    print("=" * 70)

    print(f"\n  {'Model':<8s} | {'Scenario':<12s} | {'RMSE(%)':<8s} | "
          f"{'R²(full)':<10s} | {'MAE(%)':<8s} | {'MaxErr(%)':<10s} | {'PVR(%)':<8s}")
    print(f"  {'-'*8} | {'-'*12} | {'-'*8} | {'-'*10} | {'-'*8} | {'-'*10} | {'-'*8}")

    for r in all_results:
        print(f"  {r['model']:<8s} | {r['scenario']:<12s} | "
              f"{r['rmse_pct']:>8.4f} | {r.get('r2_full', float('nan')):>10.6f} | "
              f"{r['mae_pct']:>8.4f} | "
              f"{r['max_error_pct']:>10.4f} | {r['pvr']:>8.2f}")

    # ── STEP 7: Interpretation ───────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  STEP 7 - SCIENTIFIC INTERPRETATION")
    print("=" * 70)

    # Find PI-TCN vs TCN PVR comparison
    for scenario in ['A', 'B']:
        sc_tag = f"Scenario_{scenario}"
        tcn_r = next((r for r in all_results
                      if r['model'] == 'TCN' and r['scenario'] == sc_tag), None)
        pitcn_r = next((r for r in all_results
                        if r['model'] == 'PI-TCN' and r['scenario'] == sc_tag), None)
        lstm_r = next((r for r in all_results
                       if r['model'] == 'LSTM' and r['scenario'] == sc_tag), None)

        if tcn_r and pitcn_r:
            pvr_drop = tcn_r['pvr'] - pitcn_r['pvr']
            rmse_diff = pitcn_r['rmse_pct'] - tcn_r['rmse_pct']
            print(f"\n  Scenario {scenario}:")
            print(f"    1. Does physics loss reduce PVR?")
            if pvr_drop > 0:
                print(f"       YES: PVR dropped {pvr_drop:.2f}pp "
                      f"(TCN={tcn_r['pvr']:.2f}% -> PI-TCN={pitcn_r['pvr']:.2f}%)")
            else:
                print(f"       NO: PVR did not decrease "
                      f"(TCN={tcn_r['pvr']:.2f}% vs PI-TCN={pitcn_r['pvr']:.2f}%)")

            print(f"    2. RMSE trade-off:")
            if rmse_diff > 0.5:
                print(f"       RMSE increased by {rmse_diff:.2f}pp (accuracy vs physics trade-off)")
            elif rmse_diff < -0.5:
                print(f"       RMSE DECREASED by {abs(rmse_diff):.2f}pp (physics helps accuracy!)")
            else:
                print(f"       RMSE change negligible ({rmse_diff:+.2f}pp)")

            if lstm_r:
                tcn_gain = lstm_r['rmse_pct'] - tcn_r['rmse_pct']
                print(f"    3. TCN vs LSTM advantage: {tcn_gain:+.2f}pp RMSE")

    # Save results to JSON
    results_path = os.path.join(LOG_DIR, 'sprint44_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'results': all_results,
            'reproducibility': repro_results,
        }, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")

    print(f"\n{'=' * 70}")
    print("  SPRINT 5 Seq2Seq ABLATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print(f"\n\nFATAL ERROR: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
