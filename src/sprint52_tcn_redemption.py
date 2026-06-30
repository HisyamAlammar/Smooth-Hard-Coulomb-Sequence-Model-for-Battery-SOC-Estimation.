"""
sprint52_tcn_redemption.py -- Leak-Free Hard-Coulomb TCN Redemption
===================================================================

This script puts the TCN backbone through the same post-leakage-fix protocol
used for the Sprint 48 LSTM and the Sprint 50 contextual-anchor ablation.

Milestone 1: clean v4 baseline
  - Train HardCoulombTCN on v4 Scenario A and Scenario B.
  - Replace the TCN module's local constraint with the canonical
    model_v5_coulomb.SmoothHardCoulombConstraint used by the LSTM pipeline.

Milestone 2: contextual history-only anchor
  - Train ContextualHardCoulombTCN on v5_contextual Scenario A.
  - Use the Sprint 50 "history_only" masking protocol: OCV/rest context is
    zeroed, history context is retained.
  - Report -20°C OOD t=0 RMSE, MaxE, and PVR.

No Sprint 48/49/50 files are overwritten. All artifacts are written to:

    outputs/v8_tcn_redemption/sprint52/
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    BATCH_SIZE,
    DILATION_RATES,
    DROPOUT,
    EPOCHS,
    KERNEL_SIZE,
    LEARNING_RATE,
    NUM_FILTERS,
    NUM_INPUTS,
    Q_NOMINAL,
    RANDOM_SEED,
)
from model_v5_coulomb import SmoothHardCoulombConstraint as CanonicalHardCoulombConstraint  # noqa: E402
from model_v5_coulomb_tcn import HardCoulombTCN, TemporalBlock  # noqa: E402
from preprocessing_v5_contextual import ANCHOR_CTX_COLS, HISTORY_CTX_INDICES, OCV_CTX_INDICES  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

WINDOW = 100
SAFETY_FACTOR = 1.5
PATIENCE = 12
TARGET_TEMP = "n20degC"
DISCHARGE_THRESHOLD_A = -0.05

V4_DATA_DIRS = {
    "scenario_A": BASE_DIR / "data" / "processed" / "v4_scenario_A",
    "scenario_B": BASE_DIR / "data" / "processed" / "v4_scenario_B",
}
V5_CONTEXTUAL_DIR = BASE_DIR / "data" / "processed" / "v5_contextual" / "scenario_A"
OUTPUT_DIR = BASE_DIR / "outputs" / "v8_tcn_redemption" / "sprint52"


@dataclass(frozen=True)
class CleanSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    I_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    I_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    I_test: np.ndarray
    temp_test: np.ndarray | None


@dataclass(frozen=True)
class ContextualSplits:
    X_train: np.ndarray
    A_train: np.ndarray
    I_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    A_val: np.ndarray
    I_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    A_test: np.ndarray
    I_test: np.ndarray
    y_test: np.ndarray
    temp_test: np.ndarray


class ContextualHardCoulombTCN(nn.Module):
    """
    TCN sequence backbone with a Sprint-50-style contextual anchor head.

    The sequence path remains TCN -> delta_head -> canonical Hard-Coulomb
    constraint. The anchor path receives h[:, 0, :] plus a static contextual
    feature vector. For the Sprint 52 history-only experiment, callers pass a
    full 14-D anchor vector with OCV/rest indices zeroed and history retained.
    """

    def __init__(
        self,
        num_inputs: int = NUM_INPUTS,
        anchor_ctx_dim: int = len(ANCHOR_CTX_COLS),
        num_filters: int = NUM_FILTERS,
        kernel_size: int = KERNEL_SIZE,
        dropout: float = DROPOUT,
        dilation_rates: Iterable[int] | None = None,
        q_nominal: float = Q_NOMINAL,
        safety_factor: float = SAFETY_FACTOR,
    ) -> None:
        super().__init__()
        self.num_inputs = num_inputs
        self.anchor_ctx_dim = anchor_ctx_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_rates = list(dilation_rates if dilation_rates is not None else DILATION_RATES)

        layers: List[nn.Module] = []
        for block_idx, dilation in enumerate(self.dilation_rates):
            in_channels = num_inputs if block_idx == 0 else num_filters
            layers.append(
                TemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=num_filters,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*layers)

        self.delta_head = nn.Sequential(
            nn.Linear(num_filters, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.anchor_ctx_encoder = nn.Sequential(
            nn.Linear(anchor_ctx_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
        )

        self.anchor_head = nn.Sequential(
            nn.Linear(num_filters + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.hard_constraint = CanonicalHardCoulombConstraint(
            q_nominal=q_nominal,
            safety_factor=safety_factor,
        )
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(self.dilation_rates)
        self._init_heads()

    def _init_heads(self) -> None:
        for module in [self.delta_head, self.anchor_ctx_encoder, self.anchor_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        nn.init.normal_(self.delta_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.normal_(self.anchor_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.anchor_head[-1].bias)

    def forward(
        self,
        x_seq: torch.Tensor,
        current_seq: torch.Tensor,
        anchor_ctx: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.tcn(x_seq.transpose(1, 2)).transpose(1, 2)
        delta_logits = self.delta_head(hidden)
        context_embedding = self.anchor_ctx_encoder(anchor_ctx)
        anchor_input = torch.cat([hidden[:, 0, :], context_embedding], dim=-1)
        anchor_logit = self.anchor_head(anchor_input)
        soc_pred, _delta = self.hard_constraint(delta_logits, current_seq, anchor_logit)
        return soc_pred


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_float_array(data_dir: Path, name: str) -> np.ndarray:
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required array: {path}")
    return np.load(path).astype(np.float32, copy=False)


def assert_seq_contract(X_seq: np.ndarray, y_seq: np.ndarray, current_seq: np.ndarray, split: str) -> None:
    if X_seq.ndim != 3 or X_seq.shape[1:] != (WINDOW, NUM_INPUTS):
        raise ValueError(f"{split}: expected X shape (N, {WINDOW}, {NUM_INPUTS}), got {X_seq.shape}")
    if y_seq.ndim != 2 or y_seq.shape[1] != WINDOW:
        raise ValueError(f"{split}: expected y shape (N, {WINDOW}), got {y_seq.shape}")
    if current_seq.shape != y_seq.shape:
        raise ValueError(f"{split}: expected I shape {y_seq.shape}, got {current_seq.shape}")
    if X_seq.shape[0] != y_seq.shape[0]:
        raise ValueError(f"{split}: mismatched row counts X={X_seq.shape}, y={y_seq.shape}")


def assert_anchor_contract(anchor_ctx: np.ndarray, X_seq: np.ndarray, split: str) -> None:
    if anchor_ctx.ndim != 2 or anchor_ctx.shape[1] != len(ANCHOR_CTX_COLS):
        raise ValueError(f"{split}: expected A shape (N, {len(ANCHOR_CTX_COLS)}), got {anchor_ctx.shape}")
    if anchor_ctx.shape[0] != X_seq.shape[0]:
        raise ValueError(f"{split}: mismatched A/X rows: A={anchor_ctx.shape}, X={X_seq.shape}")


def load_clean_splits(scenario_key: str) -> CleanSplits:
    data_dir = V4_DATA_DIRS[scenario_key]
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing leak-free v4 directory: {data_dir}")

    temp_path = data_dir / "temp_labels_test.npy"
    splits = CleanSplits(
        X_train=load_float_array(data_dir, "X_train.npy"),
        y_train=load_float_array(data_dir, "y_train.npy"),
        I_train=load_float_array(data_dir, "I_unscaled_train.npy"),
        X_val=load_float_array(data_dir, "X_val.npy"),
        y_val=load_float_array(data_dir, "y_val.npy"),
        I_val=load_float_array(data_dir, "I_unscaled_val.npy"),
        X_test=load_float_array(data_dir, "X_test.npy"),
        y_test=load_float_array(data_dir, "y_test.npy"),
        I_test=load_float_array(data_dir, "I_unscaled_test.npy"),
        temp_test=np.load(temp_path, allow_pickle=True) if temp_path.exists() else None,
    )
    for split_name in ["train", "val", "test"]:
        assert_seq_contract(
            getattr(splits, f"X_{split_name}"),
            getattr(splits, f"y_{split_name}"),
            getattr(splits, f"I_{split_name}"),
            f"{scenario_key}/{split_name}",
        )
    return splits


def load_contextual_splits() -> ContextualSplits:
    if not V5_CONTEXTUAL_DIR.exists():
        raise FileNotFoundError(
            f"Missing v5 contextual directory: {V5_CONTEXTUAL_DIR}. "
            "Run src/preprocessing_v5_contextual.py first."
        )
    temp_path = V5_CONTEXTUAL_DIR / "temp_labels_test.npy"
    if not temp_path.exists():
        raise FileNotFoundError(f"Missing test temperature labels: {temp_path}")

    splits = ContextualSplits(
        X_train=load_float_array(V5_CONTEXTUAL_DIR, "X_train.npy"),
        A_train=load_float_array(V5_CONTEXTUAL_DIR, "A_anchor_train.npy"),
        I_train=load_float_array(V5_CONTEXTUAL_DIR, "I_unscaled_train.npy"),
        y_train=load_float_array(V5_CONTEXTUAL_DIR, "y_train.npy"),
        X_val=load_float_array(V5_CONTEXTUAL_DIR, "X_val.npy"),
        A_val=load_float_array(V5_CONTEXTUAL_DIR, "A_anchor_val.npy"),
        I_val=load_float_array(V5_CONTEXTUAL_DIR, "I_unscaled_val.npy"),
        y_val=load_float_array(V5_CONTEXTUAL_DIR, "y_val.npy"),
        X_test=load_float_array(V5_CONTEXTUAL_DIR, "X_test.npy"),
        A_test=load_float_array(V5_CONTEXTUAL_DIR, "A_anchor_test.npy"),
        I_test=load_float_array(V5_CONTEXTUAL_DIR, "I_unscaled_test.npy"),
        y_test=load_float_array(V5_CONTEXTUAL_DIR, "y_test.npy"),
        temp_test=np.load(temp_path, allow_pickle=True),
    )
    for split_name in ["train", "val", "test"]:
        X_seq = getattr(splits, f"X_{split_name}")
        assert_seq_contract(
            X_seq,
            getattr(splits, f"y_{split_name}"),
            getattr(splits, f"I_{split_name}"),
            f"v5_contextual/{split_name}",
        )
        assert_anchor_contract(getattr(splits, f"A_{split_name}"), X_seq, f"v5_contextual/{split_name}")
    return splits


def history_only_context(anchor_ctx: np.ndarray) -> np.ndarray:
    masked = anchor_ctx.copy()
    masked[:, OCV_CTX_INDICES] = 0.0
    return masked.astype(np.float32, copy=False)


def make_clean_loader(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    current_seq: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_seq), torch.from_numpy(current_seq))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        generator=generator if shuffle else None,
    )


def make_contextual_loader(
    X_seq: np.ndarray,
    anchor_ctx: np.ndarray,
    current_seq: np.ndarray,
    y_seq: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(X_seq),
        torch.from_numpy(anchor_ctx),
        torch.from_numpy(current_seq),
        torch.from_numpy(y_seq),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        generator=generator if shuffle else None,
    )


def build_clean_tcn(device: torch.device) -> HardCoulombTCN:
    model = HardCoulombTCN(
        num_inputs=NUM_INPUTS,
        num_filters=NUM_FILTERS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
        dilation_rates=list(DILATION_RATES),
        safety_factor=SAFETY_FACTOR,
    )
    model.hard_constraint = CanonicalHardCoulombConstraint(
        q_nominal=Q_NOMINAL,
        safety_factor=SAFETY_FACTOR,
    )
    return model.to(device)


def build_contextual_tcn(device: torch.device) -> ContextualHardCoulombTCN:
    return ContextualHardCoulombTCN(
        num_inputs=NUM_INPUTS,
        anchor_ctx_dim=len(ANCHOR_CTX_COLS),
        num_filters=NUM_FILTERS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
        dilation_rates=list(DILATION_RATES),
        safety_factor=SAFETY_FACTOR,
    ).to(device)


def forward_batch(model: nn.Module, batch: Tuple[torch.Tensor, ...], mode: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if mode == "clean":
        X_batch, y_batch, I_batch = batch
        X_batch = X_batch.to(device, non_blocking=True)
        I_batch = I_batch.to(device, non_blocking=True)
        y_target = y_batch.unsqueeze(-1).to(device, non_blocking=True)
        return model(X_batch, I_batch), y_target

    if mode == "contextual":
        X_batch, A_batch, I_batch, y_batch = batch
        X_batch = X_batch.to(device, non_blocking=True)
        A_batch = A_batch.to(device, non_blocking=True)
        I_batch = I_batch.to(device, non_blocking=True)
        y_target = y_batch.unsqueeze(-1).to(device, non_blocking=True)
        return model(X_batch, I_batch, A_batch), y_target

    raise ValueError(f"Unknown mode: {mode}")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mode: str,
    description: str,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    progress = tqdm(loader, desc=description, leave=False, dynamic_ncols=True)
    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        y_pred, y_target = forward_batch(model, batch, mode, device)
        loss = criterion(y_pred, y_target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
        progress.set_postfix(loss=f"{loss.item():.6f}")
    return total_loss / max(batches, 1)


def validate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mode: str,
    description: str,
) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    progress = tqdm(loader, desc=description, leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for batch in progress:
            y_pred, y_target = forward_batch(model, batch, mode, device)
            loss = criterion(y_pred, y_target)
            total_loss += float(loss.item())
            batches += 1
            progress.set_postfix(val_loss=f"{loss.item():.6f}")
    return total_loss / max(batches, 1)


def write_history(path: Path, history: List[Dict[str, Any]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def train_model(
    model: nn.Module,
    mode: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_name: str,
    label: str,
    epochs: int,
    learning_rate: float,
    patience: int,
    device: torch.device,
    config: Dict[str, Any],
) -> Tuple[nn.Module, Dict[str, Any]]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)

    best_state: Dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    wait = 0
    history: List[Dict[str, Any]] = []

    print(f"\nTraining {label}")
    print(f"  Trainable params     : {count_parameters(model):,}")
    print(f"  Receptive field      : {getattr(model, 'receptive_field', 'n/a')} steps")
    print("  Constraint source    : model_v5_coulomb.SmoothHardCoulombConstraint")
    print(f"  Safety factor        : {SAFETY_FACTOR}")

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            mode,
            f"{label} epoch {epoch}/{epochs} train",
        )
        val_loss = validate_loss(
            model,
            val_loader,
            criterion,
            device,
            mode,
            f"{label} epoch {epoch}/{epochs} val",
        )
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": round(time.time() - start, 2),
        }
        history.append(record)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            wait = 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            wait += 1

        marker = " BEST" if improved else ""
        print(
            f"  Epoch {epoch:03d}/{epochs} | train={train_loss:.6f} | "
            f"val={val_loss:.6f} | lr={record['lr']:.2e} | {record['time_sec']:.1f}s{marker}"
        )
        if wait >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / checkpoint_name
    torch.save(
        {
            "format_version": "sprint52_tcn_redemption",
            "label": label,
            "mode": mode,
            "model_state_dict": model.state_dict(),
            "best_val_loss": best_val,
            "history": history,
            "config": config,
        },
        checkpoint_path,
    )
    history_path = OUTPUT_DIR / checkpoint_name.replace(".pt", "_history.csv")
    write_history(history_path, history)
    return model, {
        "label": label,
        "mode": mode,
        "best_val_loss": best_val,
        "epochs_ran": len(history),
        "checkpoint": str(checkpoint_path.relative_to(BASE_DIR)),
        "history": str(history_path.relative_to(BASE_DIR)),
    }


def compute_pvr(y_pred: np.ndarray, I_unscaled: np.ndarray) -> Dict[str, float | int]:
    if y_pred.shape != I_unscaled.shape:
        raise ValueError(f"PVR shape mismatch: y_pred={y_pred.shape}, I={I_unscaled.shape}")
    delta_soc = y_pred[:, 1:] - y_pred[:, :-1]
    discharge_mask = I_unscaled[:, 1:] < DISCHARGE_THRESHOLD_A
    violations = (delta_soc > 0.0) & discharge_mask
    discharge_steps = int(discharge_mask.sum())
    violation_count = int(violations.sum())
    return {
        "pvr_pct": 0.0 if discharge_steps == 0 else float(violation_count / discharge_steps * 100.0),
        "violations": violation_count,
        "discharge_steps": discharge_steps,
    }


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_flat = y_true.reshape(-1).astype(np.float64)
    pred_flat = y_pred.reshape(-1).astype(np.float64)
    ss_res = float(np.square(true_flat - pred_flat).sum())
    ss_tot = float(np.square(true_flat - true_flat.mean()).sum())
    return 0.0 if ss_tot <= 0.0 else float(1.0 - ss_res / ss_tot)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, I_unscaled: np.ndarray) -> Dict[str, Any]:
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    t0_errors = y_pred[:, 0] - y_true[:, 0]
    last_errors = y_pred[:, -1] - y_true[:, -1]
    pvr = compute_pvr(y_pred, I_unscaled)
    return {
        "n_windows": int(y_true.shape[0]),
        "rmse_full_pct": float(np.sqrt(np.mean(errors**2)) * 100.0),
        "mae_full_pct": float(np.mean(abs_errors) * 100.0),
        "maxe_full_pct": float(np.max(abs_errors) * 100.0),
        "r2_full": r2_score_np(y_true, y_pred),
        "rmse_last_pct": float(np.sqrt(np.mean(last_errors**2)) * 100.0),
        "mae_last_pct": float(np.mean(np.abs(last_errors)) * 100.0),
        "maxe_last_pct": float(np.max(np.abs(last_errors)) * 100.0),
        "t0_rmse_pct": float(np.sqrt(np.mean(t0_errors**2)) * 100.0),
        "t0_mae_pct": float(np.mean(np.abs(t0_errors)) * 100.0),
        "t0_maxe_pct": float(np.max(np.abs(t0_errors)) * 100.0),
        "pvr_pct": pvr["pvr_pct"],
        "pvr_violations": pvr["violations"],
        "pvr_discharge_steps": pvr["discharge_steps"],
    }


def predict_clean(model: nn.Module, data: CleanSplits, batch_size: int, device: torch.device) -> np.ndarray:
    loader = make_clean_loader(data.X_test, data.y_test, data.I_test, batch_size, False, 0, device)
    model.eval()
    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for X_batch, _y_batch, I_batch in tqdm(loader, desc="Clean TCN eval", leave=False, dynamic_ncols=True):
            X_batch = X_batch.to(device, non_blocking=True)
            I_batch = I_batch.to(device, non_blocking=True)
            y_pred = model(X_batch, I_batch)
            predictions.append(y_pred.detach().cpu().numpy())
    return np.concatenate(predictions, axis=0).squeeze(-1)


def predict_contextual(
    model: ContextualHardCoulombTCN,
    X_seq: np.ndarray,
    anchor_ctx: np.ndarray,
    current_seq: np.ndarray,
    y_seq: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = make_contextual_loader(X_seq, anchor_ctx, current_seq, y_seq, batch_size, False, 0, device)
    model.eval()
    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for X_batch, A_batch, I_batch, _y_batch in tqdm(loader, desc="Contextual TCN eval", leave=False, dynamic_ncols=True):
            X_batch = X_batch.to(device, non_blocking=True)
            A_batch = A_batch.to(device, non_blocking=True)
            I_batch = I_batch.to(device, non_blocking=True)
            y_pred = model(X_batch, I_batch, A_batch)
            predictions.append(y_pred.detach().cpu().numpy())
    return np.concatenate(predictions, axis=0).squeeze(-1)


def per_temperature_last_step(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    temp_labels: np.ndarray | None,
) -> Dict[str, Dict[str, float | int]]:
    if temp_labels is None or len(temp_labels) != y_true.shape[0]:
        return {}
    results: Dict[str, Dict[str, float | int]] = {}
    for temp in sorted(np.unique(temp_labels)):
        mask = temp_labels == temp
        errors = y_pred[mask, -1] - y_true[mask, -1]
        results[str(temp)] = {
            "rmse_pct": float(np.sqrt(np.mean(errors**2)) * 100.0),
            "mae_pct": float(np.mean(np.abs(errors)) * 100.0),
            "n_windows": int(mask.sum()),
        }
    return results


def evaluate_clean_result(
    scenario_key: str,
    model: nn.Module,
    data: CleanSplits,
    batch_size: int,
    device: torch.device,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    y_pred = predict_clean(model, data, batch_size, device)
    metrics = compute_metrics(data.y_test, y_pred, data.I_test)
    return {
        "milestone": "clean_v4_baseline",
        "scenario": scenario_key,
        "model": "HardCoulombTCN",
        "data_branch": str(V4_DATA_DIRS[scenario_key].relative_to(BASE_DIR)),
        "parameter_count": count_parameters(model),
        "receptive_field": int(getattr(model, "receptive_field", -1)),
        "training": summary,
        "metrics": metrics,
        "per_temperature_last_step": per_temperature_last_step(data.y_test, y_pred, data.temp_test),
    }


def evaluate_contextual_n20(
    model: ContextualHardCoulombTCN,
    data: ContextualSplits,
    batch_size: int,
    device: torch.device,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    target_mask = data.temp_test == TARGET_TEMP
    if not target_mask.any():
        raise ValueError(f"No {TARGET_TEMP} windows found in contextual Scenario A test split.")
    A_test_history = history_only_context(data.A_test[target_mask])
    y_pred = predict_contextual(
        model,
        data.X_test[target_mask],
        A_test_history,
        data.I_test[target_mask],
        data.y_test[target_mask],
        batch_size,
        device,
    )
    metrics = compute_metrics(data.y_test[target_mask], y_pred, data.I_test[target_mask])
    return {
        "milestone": "contextual_history_only_v5",
        "scenario": "scenario_A",
        "target_temp": TARGET_TEMP,
        "model": "ContextualHardCoulombTCN",
        "data_branch": str(V5_CONTEXTUAL_DIR.relative_to(BASE_DIR)),
        "parameter_count": count_parameters(model),
        "receptive_field": int(model.receptive_field),
        "anchor_context_protocol": {
            "mode": "history_only",
            "anchor_ctx_cols": ANCHOR_CTX_COLS,
            "ocv_ctx_indices_zeroed": OCV_CTX_INDICES,
            "history_ctx_indices_retained": HISTORY_CTX_INDICES,
        },
        "training": summary,
        "metrics": metrics,
    }


def print_clean_result(result: Dict[str, Any]) -> None:
    metrics = result["metrics"]
    print(
        f"  {result['scenario']} | HardCoulombTCN | "
        f"params={result['parameter_count']:,} | RF={result['receptive_field']}"
    )
    print(
        f"    Full: RMSE={metrics['rmse_full_pct']:.4f}% | "
        f"MAE={metrics['mae_full_pct']:.4f}% | MaxE={metrics['maxe_full_pct']:.4f}% | "
        f"R2={metrics['r2_full']:.6f}"
    )
    print(
        f"    t=0 : RMSE={metrics['t0_rmse_pct']:.4f}% | "
        f"MAE={metrics['t0_mae_pct']:.4f}% | MaxE={metrics['t0_maxe_pct']:.4f}%"
    )
    print(
        f"    Last: RMSE={metrics['rmse_last_pct']:.4f}% | "
        f"MAE={metrics['mae_last_pct']:.4f}% | MaxE={metrics['maxe_last_pct']:.4f}%"
    )
    print(
        f"    PVR : {metrics['pvr_pct']:.6f}% "
        f"({metrics['pvr_violations']:,} / {metrics['pvr_discharge_steps']:,}; I_unscaled < -0.05 A)"
    )
    if result["per_temperature_last_step"]:
        print("    Per-temperature last-step RMSE:")
        for temp, values in result["per_temperature_last_step"].items():
            print(
                f"      {temp:>8s}: RMSE={values['rmse_pct']:.4f}% | "
                f"MAE={values['mae_pct']:.4f}% | N={values['n_windows']:,}"
            )


def print_contextual_result(result: Dict[str, Any]) -> None:
    metrics = result["metrics"]
    print("\nSprint 52 Scientific Test: Contextual History-Only TCN at -20°C OOD")
    print("  " + "-" * 106)
    print("  Model                     | N      | t=0 RMSE | t=0 MaxE | Full RMSE | Full MaxE | PVR (%)")
    print("  " + "-" * 106)
    print(
        f"  ContextualHardCoulombTCN  | "
        f"{metrics['n_windows']:>6,} | "
        f"{metrics['t0_rmse_pct']:>8.4f} | "
        f"{metrics['t0_maxe_pct']:>8.4f} | "
        f"{metrics['rmse_full_pct']:>9.4f} | "
        f"{metrics['maxe_full_pct']:>9.4f} | "
        f"{metrics['pvr_pct']:>7.6f}"
    )
    print("  " + "-" * 106)
    print(
        f"  PVR violations: {metrics['pvr_violations']:,} / "
        f"{metrics['pvr_discharge_steps']:,} discharge steps (I_unscaled < -0.05 A)"
    )
    print("  Sprint 50 LSTM history-only reference wall: Full MaxE = 45.4195%")
    delta = metrics["maxe_full_pct"] - 45.4195
    verdict = "BROKE WALL" if metrics["maxe_full_pct"] < 45.4195 else "DID NOT BREAK WALL"
    print(f"  ΔMaxE vs Sprint50 history-only LSTM: {delta:+.4f}% -> {verdict}")


def run_clean_milestone(
    scenario_key: str,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    data = load_clean_splits(scenario_key)
    print(f"\nMilestone 1 data: {scenario_key} -> {V4_DATA_DIRS[scenario_key].relative_to(BASE_DIR)}")
    print(f"  Train: X={data.X_train.shape}, y={data.y_train.shape}, I={data.I_train.shape}")
    print(f"  Val  : X={data.X_val.shape}, y={data.y_val.shape}, I={data.I_val.shape}")
    print(f"  Test : X={data.X_test.shape}, y={data.y_test.shape}, I={data.I_test.shape}")

    model = build_clean_tcn(device)
    if args.dry_run:
        return {
            "milestone": "clean_v4_baseline",
            "scenario": scenario_key,
            "dry_run": True,
            "parameter_count": count_parameters(model),
            "receptive_field": int(model.receptive_field),
        }

    train_loader = make_clean_loader(data.X_train, data.y_train, data.I_train, args.batch_size, True, args.seed, device)
    val_loader = make_clean_loader(data.X_val, data.y_val, data.I_val, args.batch_size, False, args.seed, device)
    model, summary = train_model(
        model=model,
        mode="clean",
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_name=f"hard_coulomb_tcn_{scenario_key}.pt",
        label=f"HardCoulombTCN {scenario_key}",
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        device=device,
        config={
            "model": "HardCoulombTCN",
            "scenario": scenario_key,
            "data_branch": str(V4_DATA_DIRS[scenario_key].relative_to(BASE_DIR)),
            "num_inputs": NUM_INPUTS,
            "num_filters": NUM_FILTERS,
            "kernel_size": KERNEL_SIZE,
            "dropout": DROPOUT,
            "dilation_rates": list(DILATION_RATES),
            "safety_factor": SAFETY_FACTOR,
            "constraint_source": "model_v5_coulomb.SmoothHardCoulombConstraint",
        },
    )
    result = evaluate_clean_result(scenario_key, model, data, args.batch_size, device, summary)
    print_clean_result(result)
    return result


def run_contextual_milestone(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    data = load_contextual_splits()
    print(f"\nMilestone 2 data: Scenario A history-only -> {V5_CONTEXTUAL_DIR.relative_to(BASE_DIR)}")
    print(f"  Train: X={data.X_train.shape}, A={data.A_train.shape}, y={data.y_train.shape}")
    print(f"  Val  : X={data.X_val.shape}, A={data.A_val.shape}, y={data.y_val.shape}")
    print(f"  Test : X={data.X_test.shape}, A={data.A_test.shape}, y={data.y_test.shape}")
    print(f"  History retained indices: {HISTORY_CTX_INDICES}")
    print(f"  OCV/rest zeroed indices : {OCV_CTX_INDICES}")

    A_train = history_only_context(data.A_train)
    A_val = history_only_context(data.A_val)
    model = build_contextual_tcn(device)
    if args.dry_run:
        return {
            "milestone": "contextual_history_only_v5",
            "scenario": "scenario_A",
            "target_temp": TARGET_TEMP,
            "dry_run": True,
            "parameter_count": count_parameters(model),
            "receptive_field": int(model.receptive_field),
        }

    train_loader = make_contextual_loader(data.X_train, A_train, data.I_train, data.y_train, args.batch_size, True, args.seed, device)
    val_loader = make_contextual_loader(data.X_val, A_val, data.I_val, data.y_val, args.batch_size, False, args.seed, device)
    model, summary = train_model(
        model=model,
        mode="contextual",
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_name="contextual_history_hard_coulomb_tcn_scenario_A.pt",
        label="ContextualHardCoulombTCN scenario_A history_only",
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        device=device,
        config={
            "model": "ContextualHardCoulombTCN",
            "scenario": "scenario_A",
            "target_temp": TARGET_TEMP,
            "data_branch": str(V5_CONTEXTUAL_DIR.relative_to(BASE_DIR)),
            "num_inputs": NUM_INPUTS,
            "anchor_ctx_dim": len(ANCHOR_CTX_COLS),
            "num_filters": NUM_FILTERS,
            "kernel_size": KERNEL_SIZE,
            "dropout": DROPOUT,
            "dilation_rates": list(DILATION_RATES),
            "safety_factor": SAFETY_FACTOR,
            "constraint_source": "model_v5_coulomb.SmoothHardCoulombConstraint",
            "anchor_context_protocol": "history_only",
            "ocv_ctx_indices_zeroed": OCV_CTX_INDICES,
            "history_ctx_indices_retained": HISTORY_CTX_INDICES,
        },
    )
    result = evaluate_contextual_n20(model, data, args.batch_size, device, summary)
    print_contextual_result(result)
    return result


def parse_csv_arg(raw_value: str, allowed: Tuple[str, ...], arg_name: str) -> List[str]:
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    unknown = [item for item in values if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown {arg_name}: {unknown}; allowed={allowed}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 52 leak-free Hard-Coulomb TCN redemption study.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--milestones",
        type=str,
        default="clean,contextual",
        help="Comma-separated subset: clean,contextual",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="scenario_A,scenario_B",
        help="Comma-separated clean-baseline scenarios: scenario_A,scenario_B",
    )
    parser.add_argument("--dry-run", action="store_true", help="Load data and instantiate models without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    milestones = parse_csv_arg(args.milestones, ("clean", "contextual"), "milestones")
    scenarios = parse_csv_arg(args.scenarios, ("scenario_A", "scenario_B"), "scenarios")
    set_reproducibility(args.seed)
    device = resolve_device(args.device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 108)
    print("  Sprint 52 -- TCN Redemption under Leak-Free Hard-Coulomb Protocol")
    print("=" * 108)
    print(f"  Device       : {device}")
    print(f"  Output dir   : {OUTPUT_DIR.relative_to(BASE_DIR)}")
    print(f"  Milestones   : {', '.join(milestones)}")
    print(f"  Clean data   : v4 Scenario A/B only")
    print(f"  Context data : v5_contextual Scenario A only")
    print(f"  PVR rule     : I_unscaled < -0.05 A")
    print(f"  Dry run      : {args.dry_run}")

    results: List[Dict[str, Any]] = []
    if "clean" in milestones:
        print("\n" + "=" * 88)
        print("  MILESTONE 1: Clean v4 HardCoulombTCN Baseline")
        print("=" * 88)
        for scenario_key in scenarios:
            results.append(run_clean_milestone(scenario_key, args, device))

    if "contextual" in milestones:
        print("\n" + "=" * 88)
        print("  MILESTONE 2: v5 Contextual History-Only HardCoulombTCN")
        print("=" * 88)
        results.append(run_contextual_milestone(args, device))

    payload = {
        "script": "src/sprint52_tcn_redemption.py",
        "output_dir": str(OUTPUT_DIR.relative_to(BASE_DIR)),
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "patience": args.patience,
        "seed": args.seed,
        "dry_run": args.dry_run,
        "results": results,
    }
    output_path = OUTPUT_DIR / "sprint52_tcn_redemption_results.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\n  Results saved: {output_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
