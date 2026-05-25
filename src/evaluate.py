"""
evaluate.py -- Sprint 5: Seq2Seq Evaluation & Intra-Window PVR
========================================================================
Standalone evaluation module for PI-TCN SOC Estimator (Seq2Seq).

Metrics:
  1. Full-Sequence RMSE, MAE, Max Error (all 100 timesteps)
  2. Last-Step RMSE, MAE, Max Error (t=-1, backward-compatible)
  3. RMSE per temperature (Scenario A breakdown)
  4. Intra-Window PVR -- key metric for PI paper

Created : 2026-04-08
Updated : 2026-05-10 -- Sprint 5 (Seq2Seq pivot)
"""

import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_PROC, OUTPUT_MOD, OUTPUT_FIG, LOG_DIR
from model import TCN_SOC_Estimator, count_parameters


# =====================================================================
# 1. Physics Violation Rate (PVR)
# =====================================================================
def compute_pvr(y_pred, current_values):
    """
    Compute Physics Violation Rate (intra-window, Seq2Seq).

    Supports two input modes:
      - 2D: y_pred (N, T), current_values (N, T) -- intra-window
      - 1D: y_pred (N,), current_values (N,) -- inter-sample (legacy)

    PVR = (violations / total_discharge_steps) * 100%
    A violation: delta_SOC > 0 when I < -0.05 A (discharging)

    Parameters
    ----------
    y_pred         : ndarray (N, T) or (N,)
    current_values : ndarray (N, T) or (N,)

    Returns
    -------
    pvr : float, percentage [0, 100]
    n_violations : int
    n_discharge  : int
    """
    if y_pred.ndim == 2 and current_values.ndim == 2:
        # Intra-window mode: delta across timesteps within each window
        # delta_soc: (N, T-1), current at destination: (N, T-1)
        delta_soc = y_pred[:, 1:] - y_pred[:, :-1]      # (N, T-1)
        discharge_mask = current_values[:, 1:] < -0.05   # (N, T-1)
    else:
        # Legacy 1D inter-sample mode
        delta_soc = np.diff(y_pred)
        discharge_mask = current_values[1:] < -0.05

    n_discharge = int(discharge_mask.sum())
    if n_discharge == 0:
        return 0.0, 0, 0

    violations = (delta_soc > 0) & discharge_mask
    n_violations = int(violations.sum())

    pvr = (n_violations / n_discharge) * 100.0
    return pvr, n_violations, n_discharge


# =====================================================================
# 2. Standard Metrics
# =====================================================================
def compute_metrics(y_pred, y_true):
    """Compute RMSE, MAE, Max Error, and R² score."""
    errors = y_pred - y_true
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    max_err = np.max(np.abs(errors))
    r2 = float(r2_score(y_true.flatten(), y_pred.flatten()))
    return {
        'rmse': rmse,
        'rmse_pct': rmse * 100,
        'mae': mae,
        'mae_pct': mae * 100,
        'max_error': max_err,
        'max_error_pct': max_err * 100,
        'r2': r2,
        'n_samples': len(y_true),
    }


# =====================================================================
# 3. Per-Temperature RMSE Breakdown
# =====================================================================
def compute_per_temp_rmse(y_pred, y_true, temp_labels):
    """
    Compute RMSE for each temperature group.

    Parameters
    ----------
    y_pred, y_true : 1D arrays
    temp_labels    : 1D array of strings (e.g., '40degC', 'n10degC')

    Returns
    -------
    dict: {temp_name: {'rmse': float, 'rmse_pct': float, 'n': int}}
    """
    results = {}
    unique_temps = np.unique(temp_labels)

    for temp in sorted(unique_temps):
        mask = temp_labels == temp
        if mask.sum() == 0:
            continue

        errors = y_pred[mask] - y_true[mask]
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(np.abs(errors))

        results[temp] = {
            'rmse': rmse,
            'rmse_pct': rmse * 100,
            'mae': mae,
            'mae_pct': mae * 100,
            'n_samples': int(mask.sum()),
        }

    return results


# =====================================================================
# 4. Full Evaluation Pipeline
# =====================================================================
def evaluate_scenario(scenario_name, device=None):
    """
    Load best model checkpoint and evaluate on test set (Seq2Seq).

    Returns metrics dict with dual metrics, intra-window PVR, and predictions.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data_dir = os.path.join(DATA_PROC, scenario_name)
    X_test = np.load(os.path.join(data_dir, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(data_dir, "y_test.npy")).astype(np.float32)

    # Load temp labels if available
    temp_labels_path = os.path.join(data_dir, "temp_labels_test.npy")
    temp_labels = np.load(temp_labels_path, allow_pickle=True) if os.path.exists(temp_labels_path) else None

    print(f"\n  Evaluating {scenario_name}:")
    print(f"    Test set: X={X_test.shape}, y={y_test.shape}")

    # Load model
    model_path = os.path.join(OUTPUT_MOD, f"pi_tcn_{scenario_name}.pt")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    config = checkpoint.get('config', {})
    model = TCN_SOC_Estimator(
        num_inputs=config.get('num_inputs', 5),
        num_filters=config.get('num_filters', 64),
        kernel_size=config.get('kernel_size', 7),
        dropout=config.get('dropout', 0.2),
        dilation_rates=config.get('dilation_rates', [1, 2, 4, 8]),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"    Model loaded from epoch {checkpoint.get('epoch', '?')}")

    # Inference (Seq2Seq)
    batch_size = 1024
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            X_batch = torch.from_numpy(X_test[i:i+batch_size]).to(device)
            y_pred = model(X_batch)              # (B, 100, 1)
            all_preds.append(y_pred.cpu().numpy())

    # y_pred_all: (N, 100, 1) -> (N, 100)
    y_pred_all = np.concatenate(all_preds, axis=0).squeeze(-1)  # (N, 100)
    y_true_all = y_test                                         # (N, 100)

    # --- Dual metrics ---
    # Full-sequence
    errors_full = y_pred_all - y_true_all
    rmse_full = np.sqrt(np.mean(errors_full ** 2))
    mae_full = np.mean(np.abs(errors_full))
    max_err_full = np.max(np.abs(errors_full))
    r2_full = float(r2_score(y_true_all.flatten(), y_pred_all.flatten()))

    # Last-step (backward-compatible)
    y_pred_last = y_pred_all[:, -1]
    y_true_last = y_true_all[:, -1]
    errors_last = y_pred_last - y_true_last
    rmse_last = np.sqrt(np.mean(errors_last ** 2))
    mae_last = np.mean(np.abs(errors_last))
    max_err_last = np.max(np.abs(errors_last))
    r2_last = float(r2_score(y_true_last, y_pred_last))

    metrics = compute_metrics(y_pred_last, y_true_last)
    metrics.update({
        'rmse_full_pct': rmse_full * 100,
        'mae_full_pct': mae_full * 100,
        'max_error_full_pct': max_err_full * 100,
        'r2_full': r2_full,
        'r2_last': r2_last,
    })

    print(f"    [Full-Seq]  RMSE: {rmse_full*100:.4f}%  R²={r2_full:.6f}")
    print(f"    [Full-Seq]  MAE : {mae_full*100:.4f}%")
    print(f"    [Full-Seq]  MaxE: {max_err_full*100:.4f}%")
    print(f"    [Last-Step] RMSE: {rmse_last*100:.4f}%  R²={r2_last:.6f}")
    print(f"    [Last-Step] MAE : {mae_last*100:.4f}%")
    print(f"    [Last-Step] MaxE: {max_err_last*100:.4f}%")

    # Intra-window PVR
    # Full current sequence: (N, 100), unscaled
    current_scaled = X_test[:, :, 1]              # (N, 100)
    current_real = current_scaled * 40.0 - 20.0   # unscale

    pvr, n_viol, n_dis = compute_pvr(y_pred_all, current_real)
    print(f"    PVR       : {pvr:.2f}% ({n_viol:,} violations / "
          f"{n_dis:,} discharge steps)")

    # Per-temperature breakdown (using last-step)
    if temp_labels is not None and len(temp_labels) == len(y_pred_last):
        print(f"\n    Per-Temperature RMSE (last-step, {scenario_name}):")
        per_temp = compute_per_temp_rmse(y_pred_last, y_true_last, temp_labels)
        for temp, m in per_temp.items():
            print(f"      {temp:>8s}: RMSE={m['rmse_pct']:6.2f}%, "
                  f"MAE={m['mae_pct']:6.2f}%, N={m['n_samples']:,}")
    else:
        per_temp = None

    return {
        'metrics': metrics,
        'pvr': pvr,
        'pvr_violations': n_viol,
        'pvr_discharge': n_dis,
        'per_temp': per_temp,
        'y_pred': y_pred_all,
        'y_true': y_true_all,
    }


# =====================================================================
# 5. Standalone Execution
# =====================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  PI-TCN Evaluation -- Seq2Seq Intra-Window PVR")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    for scenario in ['scenario_A', 'scenario_B']:
        model_path = os.path.join(OUTPUT_MOD, f"pi_tcn_{scenario}.pt")
        if not os.path.exists(model_path):
            print(f"\n  [{scenario}] No model found, skipping.")
            continue

        results = evaluate_scenario(scenario, device)
