"""
sprint48_common.py -- Shared utilities for final v7 isolated pipelines.

This module contains only scenario-agnostic mechanics: model definitions,
shape checks, dataloaders, training loops, checkpointing, and UTF-8 terminal
configuration. Scenario selection stays in the two dedicated train scripts.
"""

from __future__ import annotations

import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def configure_utf8_stdio() -> None:
    """Force UTF-8 terminal streams on Windows and POSIX terminals."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


configure_utf8_stdio()

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_INPUTS, RANDOM_SEED  # noqa: E402
from model_v5_coulomb import HardCoulombLSTM  # noqa: E402


OUTPUT_DIR = BASE_DIR / "outputs" / "v7_final"
LOG_DIR = OUTPUT_DIR / "logs"
WINDOW = 100
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
SAFETY_FACTOR = 1.5
PATIENCE = 10
MODEL_KINDS = ("vanilla_lstm", "hard_coulomb_lstm")


class VanillaLSTM(nn.Module):
    """
    Leakage-safe seq2seq vanilla LSTM baseline.

    Input:  (B, 100, 5)
    Output: (B, 100, 1)
    """

    def __init__(
        self,
        num_inputs: int = NUM_INPUTS,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.lstm(x)
        return self.fc(hidden)


@dataclass(frozen=True)
class LoadedSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    I_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    I_val: np.ndarray


@dataclass(frozen=True)
class LoadedTestSplit:
    X_test: np.ndarray
    y_test: np.ndarray
    I_test: np.ndarray
    temp_labels: np.ndarray | None


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_reproducibility(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def resolve_device(device_arg: str | None = None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_v4_data_dir(data_dir: Path, scenario_key: str) -> None:
    expected = f"v4_{scenario_key}"
    if data_dir.name != expected:
        raise ValueError(f"{scenario_key} must use {expected}, got {data_dir.name}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing processed v4 data directory: {data_dir}")


def load_metadata(data_dir: Path) -> Dict[str, Any]:
    metadata_path = data_dir / "metadata_v4.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_array(data_dir: Path, name: str) -> np.ndarray:
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required array: {path}")
    return np.load(path).astype(np.float32, copy=False)


def assert_seq2seq_contract(X: np.ndarray, y: np.ndarray, I: np.ndarray, split: str) -> None:
    if X.ndim != 3 or X.shape[1:] != (WINDOW, NUM_INPUTS):
        raise ValueError(f"{split}: expected X shape (N, {WINDOW}, {NUM_INPUTS}), got {X.shape}")
    if y.ndim != 2 or y.shape[1] != WINDOW:
        raise ValueError(f"{split}: expected y shape (N, {WINDOW}), got {y.shape}")
    if I.ndim != 2 or I.shape[1] != WINDOW:
        raise ValueError(f"{split}: expected I shape (N, {WINDOW}), got {I.shape}")
    if X.shape[0] != y.shape[0] or y.shape != I.shape:
        raise ValueError(f"{split}: inconsistent X/y/I sample counts: {X.shape}, {y.shape}, {I.shape}")


def load_training_splits(data_dir: Path, scenario_key: str) -> LoadedSplits:
    ensure_v4_data_dir(data_dir, scenario_key)
    X_train = load_array(data_dir, "X_train.npy")
    y_train = load_array(data_dir, "y_train.npy")
    I_train = load_array(data_dir, "I_unscaled_train.npy")
    X_val = load_array(data_dir, "X_val.npy")
    y_val = load_array(data_dir, "y_val.npy")
    I_val = load_array(data_dir, "I_unscaled_val.npy")
    assert_seq2seq_contract(X_train, y_train, I_train, "train")
    assert_seq2seq_contract(X_val, y_val, I_val, "val")
    return LoadedSplits(X_train, y_train, I_train, X_val, y_val, I_val)


def load_test_split(data_dir: Path, scenario_key: str) -> LoadedTestSplit:
    ensure_v4_data_dir(data_dir, scenario_key)
    X_test = load_array(data_dir, "X_test.npy")
    y_test = load_array(data_dir, "y_test.npy")
    I_test = load_array(data_dir, "I_unscaled_test.npy")
    assert_seq2seq_contract(X_test, y_test, I_test, "test")
    temp_path = data_dir / "temp_labels_test.npy"
    temp_labels = np.load(temp_path, allow_pickle=True) if temp_path.exists() else None
    return LoadedTestSplit(X_test, y_test, I_test, temp_labels)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    I: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(I))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        generator=generator if shuffle else None,
    )


def build_model(model_kind: str, config: Dict[str, Any] | None = None) -> nn.Module:
    config = config or {}
    if model_kind == "vanilla_lstm":
        return VanillaLSTM(
            num_inputs=int(config.get("num_inputs", NUM_INPUTS)),
            hidden_size=int(config.get("hidden_size", HIDDEN_SIZE)),
            num_layers=int(config.get("num_layers", NUM_LAYERS)),
            dropout=float(config.get("dropout", DROPOUT)),
        )
    if model_kind == "hard_coulomb_lstm":
        return HardCoulombLSTM(
            num_inputs=int(config.get("num_inputs", NUM_INPUTS)),
            hidden_size=int(config.get("hidden_size", HIDDEN_SIZE)),
            num_layers=int(config.get("num_layers", NUM_LAYERS)),
            dropout=float(config.get("dropout", DROPOUT)),
            safety_factor=float(config.get("safety_factor", SAFETY_FACTOR)),
        )
    raise ValueError(f"Unknown model kind: {model_kind}")


def model_display_name(model_kind: str) -> str:
    if model_kind == "vanilla_lstm":
        return "VanillaLSTM"
    if model_kind == "hard_coulomb_lstm":
        return "HardCoulombLSTM"
    raise ValueError(f"Unknown model kind: {model_kind}")


def model_config(model_kind: str) -> Dict[str, Any]:
    base = {
        "num_inputs": NUM_INPUTS,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
    }
    if model_kind == "hard_coulomb_lstm":
        base["safety_factor"] = SAFETY_FACTOR
    return base


def forward_model(model: nn.Module, model_kind: str, X_batch: torch.Tensor, I_batch: torch.Tensor) -> torch.Tensor:
    if model_kind == "hard_coulomb_lstm":
        return model(X_batch, I_batch)
    return model(X_batch)


def train_one_epoch(
    model: nn.Module,
    model_kind: str,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    description: str,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    progress = tqdm(loader, desc=description, leave=False, dynamic_ncols=True)
    for X_batch, y_batch, I_batch in progress:
        X_batch = X_batch.to(device, non_blocking=True)
        y_target = y_batch.unsqueeze(-1).to(device, non_blocking=True)
        I_batch = I_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        y_pred = forward_model(model, model_kind, X_batch, I_batch)
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
    model_kind: str,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    description: str,
) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    progress = tqdm(loader, desc=description, leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for X_batch, y_batch, I_batch in progress:
            X_batch = X_batch.to(device, non_blocking=True)
            y_target = y_batch.unsqueeze(-1).to(device, non_blocking=True)
            I_batch = I_batch.to(device, non_blocking=True)
            y_pred = forward_model(model, model_kind, X_batch, I_batch)
            loss = criterion(y_pred, y_target)
            total_loss += float(loss.item())
            batches += 1
            progress.set_postfix(val_loss=f"{loss.item():.6f}")
    return total_loss / max(batches, 1)


def checkpoint_path(model_kind: str, scenario_key: str, latest: bool = False) -> Path:
    suffix = "_latest" if latest else ""
    return OUTPUT_DIR / f"{model_kind}_{scenario_key}{suffix}.pt"


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_kind: str,
    scenario_key: str,
    data_dir: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    history: List[Dict[str, Any]] | None = None,
    best_val_loss: float | None = None,
) -> None:
    payload: Dict[str, Any] = {
        "format_version": "v7_final",
        "scenario": scenario_key,
        "data_source": str(data_dir.relative_to(BASE_DIR)),
        "model_kind": model_kind,
        "model_name": model_display_name(model_kind),
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": val_loss if best_val_loss is None else best_val_loss,
        "config": model_config(model_kind),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if history is not None:
        payload["history"] = history
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_kind = checkpoint["model_kind"]
    model = build_model(model_kind, checkpoint.get("config", {})).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def train_model(
    model: nn.Module,
    model_kind: str,
    scenario_key: str,
    data_dir: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    patience: int,
    resume: bool,
) -> Dict[str, Any]:
    display_name = model_display_name(model_kind)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)

    best_path = checkpoint_path(model_kind, scenario_key, latest=False)
    latest_path = checkpoint_path(model_kind, scenario_key, latest=True)
    log_path = LOG_DIR / f"training_log_{model_kind}_{scenario_key}.csv"

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 1

    if resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = list(checkpoint.get("history", []))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"  [{display_name}] Resuming from epoch {start_epoch}")

    print(f"\n  [{display_name}] Training {scenario_key} for {epochs} epochs")
    start_time = time.time()
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model,
            model_kind,
            train_loader,
            criterion,
            optimizer,
            device,
            f"{display_name} {scenario_key} epoch {epoch}/{epochs} train",
        )
        val_loss = validate_loss(
            model,
            model_kind,
            val_loader,
            criterion,
            device,
            f"{display_name} {scenario_key} epoch {epoch}/{epochs} val",
        )
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": round(time.time() - epoch_start, 2),
        }
        history.append(record)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                best_path,
                model,
                model_kind,
                scenario_key,
                data_dir,
                epoch,
                train_loss,
                val_loss,
                best_val_loss=best_val_loss,
            )
        else:
            patience_counter += 1

        save_checkpoint(
            latest_path,
            model,
            model_kind,
            scenario_key,
            data_dir,
            epoch,
            train_loss,
            val_loss,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            best_val_loss=best_val_loss,
        )

        marker = " BEST" if improved else ""
        print(
            f"    epoch {epoch:03d}/{epochs} | train={train_loss:.6f} | "
            f"val={val_loss:.6f} | lr={record['lr']:.2e} | {record['time_sec']:.1f}s{marker}"
        )
        if patience_counter >= patience:
            print(f"  [{display_name}] Early stopping after {epoch} epochs")
            break

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if history:
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    total_minutes = (time.time() - start_time) / 60.0
    return {
        "model_kind": model_kind,
        "model_name": display_name,
        "best_checkpoint": str(best_path.relative_to(BASE_DIR)),
        "latest_checkpoint": str(latest_path.relative_to(BASE_DIR)),
        "log": str(log_path.relative_to(BASE_DIR)),
        "best_val_loss": best_val_loss,
        "epochs_ran": len(history),
        "minutes": round(total_minutes, 2),
    }


def run_scenario_training(
    scenario_key: str,
    scenario_label: str,
    data_dir: Path,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    patience: int = PATIENCE,
    seed: int = RANDOM_SEED,
    device_arg: str | None = None,
    resume: bool = False,
) -> Dict[str, Any]:
    configure_utf8_stdio()
    set_reproducibility(seed)
    ensure_v4_data_dir(data_dir, scenario_key)
    metadata = load_metadata(data_dir)
    device = resolve_device(device_arg)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print(f"  Sprint 48 Final Training -- {scenario_label}")
    print("=" * 90)
    print(f"  Scenario key : {scenario_key}")
    print(f"  Data source  : {data_dir.relative_to(BASE_DIR)}")
    print(f"  Split method : {metadata.get('method', 'unknown')}")
    print(f"  Device       : {device}")
    print(f"  Batch size   : {batch_size}")
    print(f"  Epochs       : {epochs}")

    splits = load_training_splits(data_dir, scenario_key)
    print(f"  Train split  : X={splits.X_train.shape}, y={splits.y_train.shape}, I={splits.I_train.shape}")
    print(f"  Val split    : X={splits.X_val.shape}, y={splits.y_val.shape}, I={splits.I_val.shape}")

    vanilla = VanillaLSTM(num_inputs=NUM_INPUTS, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    hard_coulomb = HardCoulombLSTM(
        num_inputs=NUM_INPUTS,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        safety_factor=SAFETY_FACTOR,
    ).to(device)

    vanilla_count = sum(p.numel() for p in vanilla.parameters() if p.requires_grad)
    hard_coulomb_count = sum(p.numel() for p in hard_coulomb.parameters() if p.requires_grad)
    print(f"  VanillaLSTM trainable parameters      : {vanilla_count:,}")
    print(f"  HardCoulombLSTM trainable parameters  : {hard_coulomb_count:,}")

    summaries = []
    for model_kind, model in (("vanilla_lstm", vanilla), ("hard_coulomb_lstm", hard_coulomb)):
        train_loader = make_loader(splits.X_train, splits.y_train, splits.I_train, batch_size, True, seed, device)
        val_loader = make_loader(splits.X_val, splits.y_val, splits.I_val, batch_size, False, seed, device)
        summaries.append(
            train_model(
                model=model,
                model_kind=model_kind,
                scenario_key=scenario_key,
                data_dir=data_dir,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=epochs,
                learning_rate=learning_rate,
                patience=patience,
                resume=resume,
            )
        )

    summary = {
        "scenario": scenario_key,
        "data_source": str(data_dir.relative_to(BASE_DIR)),
        "output_dir": str(OUTPUT_DIR.relative_to(BASE_DIR)),
        "parameter_counts": {
            "vanilla_lstm": vanilla_count,
            "hard_coulomb_lstm": hard_coulomb_count,
        },
        "runs": summaries,
    }
    summary_path = OUTPUT_DIR / f"training_summary_{scenario_key}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\n  Summary saved: {summary_path.relative_to(BASE_DIR)}")
    print("  Training complete.")
    return summary
