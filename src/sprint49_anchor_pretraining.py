"""
sprint49_anchor_pretraining.py -- Strict two-stage anchor pretraining experiment.

Purpose
-------
Test whether the Hard-Coulomb LSTM "Anchor Trap" is caused by gradient
interference during joint sequence training or by a true observability limit
from the first-timestep battery features.

No preprocessing is performed here. The script consumes only leakage-safe v4
Train/Val/Test tensors created by preprocessing_v4.py.

Stage 1
-------
Train an isolated one-step anchor path on X[:, 0, :] -> y[:, 0].
The standalone model contains the exact full-model anchor path:

    one-step LSTM backbone -> anchor_head

The LSTM sees a sequence length of exactly 1, so no future timesteps are used.

Stage 2
-------
Initialize HardCoulombLSTM from the pretrained one-step anchor path, freeze
anchor_head only, and train the remaining LSTM backbone + delta_head on the
full sequence objective.

Stage 3
-------
Evaluate Scenario A and Scenario B test sets with RMSE, MaxE, PVR, and t=0
output error.
"""

from __future__ import annotations

import argparse
import copy
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

from config import BATCH_SIZE, LEARNING_RATE, NUM_INPUTS, RANDOM_SEED  # noqa: E402
from model_v5_coulomb import HardCoulombLSTM  # noqa: E402
from sprint48_common import (  # noqa: E402
    DROPOUT,
    HIDDEN_SIZE,
    NUM_LAYERS,
    SAFETY_FACTOR,
    configure_utf8_stdio,
    load_test_split,
    load_training_splits,
    make_loader,
    resolve_device,
)

configure_utf8_stdio()

SCENARIO_DATA_DIRS = {
    "scenario_A": BASE_DIR / "data" / "processed" / "v4_scenario_A",
    "scenario_B": BASE_DIR / "data" / "processed" / "v4_scenario_B",
}
OUTPUT_DIR = BASE_DIR / "outputs" / "v7_final" / "sprint49_anchor_pretraining"
DISCHARGE_THRESHOLD_A = -0.05
BASELINE_S48_MAXE = {
    "scenario_A": 54.8409,
    "scenario_B": 35.4357,
}


@dataclass(frozen=True)
class SplitBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    I_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    I_val: np.ndarray


class OneStepAnchorPath(nn.Module):
    """
    Isolated anchor pretrainer using the exact full-model t=0 anchor path.

    It is intentionally not a sequence predictor. The LSTM receives only
    X[:, 0, :] as a length-1 sequence, then anchor_head predicts SOC_t0.
    """

    def __init__(
        self,
        num_inputs: int = NUM_INPUTS,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        safety_factor: float = SAFETY_FACTOR,
    ) -> None:
        super().__init__()
        template = HardCoulombLSTM(
            num_inputs=num_inputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            safety_factor=safety_factor,
        )
        self.lstm = template.lstm
        self.anchor_head = template.anchor_head

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        if x0.ndim == 3:
            x0 = x0[:, 0, :]
        if x0.ndim != 2:
            raise ValueError(f"Expected x0 shape (B, C) or (B, T, C), got {tuple(x0.shape)}")
        one_step = x0.unsqueeze(1)
        hidden, _ = self.lstm(one_step)
        return self.anchor_head(hidden[:, 0, :]).squeeze(-1)


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


def count_trainable(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def concat_arrays(items: Iterable[np.ndarray]) -> np.ndarray:
    arrays = list(items)
    if not arrays:
        raise ValueError("No arrays to concatenate.")
    return np.concatenate(arrays, axis=0)


def load_train_val_source(source: str) -> SplitBundle:
    """
    Load train/val arrays without touching any test data.

    source="scenario_A" keeps the OOD Scenario A test split clean.
    source="combined" concatenates train splits and val splits from A+B only;
    test splits remain untouched for Stage 3.
    """
    if source not in {"scenario_A", "scenario_B", "combined"}:
        raise ValueError(f"Unknown train source: {source}")

    scenario_keys = ("scenario_A", "scenario_B") if source == "combined" else (source,)
    loaded = [load_training_splits(SCENARIO_DATA_DIRS[key], key) for key in scenario_keys]

    return SplitBundle(
        X_train=concat_arrays(split.X_train for split in loaded),
        y_train=concat_arrays(split.y_train for split in loaded),
        I_train=concat_arrays(split.I_train for split in loaded),
        X_val=concat_arrays(split.X_val for split in loaded),
        y_val=concat_arrays(split.y_val for split in loaded),
        I_val=concat_arrays(split.I_val for split in loaded),
    )


def make_anchor_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(X[:, 0, :].astype(np.float32, copy=False)),
        torch.from_numpy(y[:, 0].astype(np.float32, copy=False)),
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


def train_anchor_epoch(
    model: OneStepAnchorPath,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    progress = tqdm(loader, desc=f"Stage 1 anchor {epoch}/{total_epochs}", leave=False, dynamic_ncols=True)
    for x0_batch, y0_batch in progress:
        x0_batch = x0_batch.to(device, non_blocking=True)
        y0_batch = y0_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x0_batch)
        loss = criterion(pred, y0_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
        progress.set_postfix(loss=f"{loss.item():.6f}")
    return total_loss / max(batches, 1)


def validate_anchor(
    model: OneStepAnchorPath,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for x0_batch, y0_batch in loader:
            x0_batch = x0_batch.to(device, non_blocking=True)
            y0_batch = y0_batch.to(device, non_blocking=True)
            pred = model(x0_batch)
            loss = criterion(pred, y0_batch)
            total_loss += float(loss.item())
            batches += 1
    return total_loss / max(batches, 1)


def train_anchor_stage(
    splits: SplitBundle,
    batch_size: int,
    epochs: int,
    lr: float,
    patience: int,
    seed: int,
    device: torch.device,
    train_source: str,
) -> Tuple[OneStepAnchorPath, Path, List[Dict[str, Any]]]:
    anchor_model = OneStepAnchorPath().to(device)
    train_loader = make_anchor_loader(splits.X_train, splits.y_train, batch_size, True, seed, device)
    val_loader = make_anchor_loader(splits.X_val, splits.y_val, batch_size, False, seed, device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(anchor_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)

    best_state = copy.deepcopy(anchor_model.state_dict())
    best_val = float("inf")
    wait = 0
    history: List[Dict[str, Any]] = []

    print("\nSTAGE 1: Isolated anchor pretraining")
    print(f"  Train source       : {train_source}")
    print(f"  Trainable params   : {count_trainable(anchor_model):,}")
    print("  Objective          : MSE(anchor(X[:,0,:]), y[:,0])")
    print("  Future timesteps   : not visible to Stage 1")

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = train_anchor_epoch(anchor_model, train_loader, criterion, optimizer, device, epoch, epochs)
        val_loss = validate_anchor(anchor_model, val_loader, criterion, device)
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
            best_state = copy.deepcopy(anchor_model.state_dict())
        else:
            wait += 1
        marker = " BEST" if improved else ""
        print(
            f"  Anchor epoch {epoch:03d}/{epochs} | train={train_loss:.6f} | "
            f"val={val_loss:.6f} | lr={record['lr']:.2e} | {record['time_sec']:.1f}s{marker}"
        )
        if wait >= patience:
            print(f"  Anchor early stopping at epoch {epoch}")
            break

    anchor_model.load_state_dict(best_state)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"stage1_anchor_path_{train_source}.pt"
    torch.save(
        {
            "stage": "stage1_anchor_path",
            "train_source": train_source,
            "model_state_dict": anchor_model.state_dict(),
            "best_val_loss": best_val,
            "history": history,
            "config": {
                "num_inputs": NUM_INPUTS,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "safety_factor": SAFETY_FACTOR,
            },
        },
        path,
    )
    write_history(OUTPUT_DIR / f"stage1_anchor_history_{train_source}.csv", history)
    print(f"  Stage 1 best val loss: {best_val:.6f}")
    print(f"  Stage 1 checkpoint   : {path.relative_to(BASE_DIR)}")
    return anchor_model, path, history


def freeze_anchor_head(model: HardCoulombLSTM) -> None:
    for param in model.anchor_head.parameters():
        param.requires_grad = False


def initialize_stage2_model(anchor_model: OneStepAnchorPath, warm_start_lstm: bool, device: torch.device) -> HardCoulombLSTM:
    model = HardCoulombLSTM(
        num_inputs=NUM_INPUTS,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        safety_factor=SAFETY_FACTOR,
    ).to(device)
    model.anchor_head.load_state_dict(anchor_model.anchor_head.state_dict())
    if warm_start_lstm:
        model.lstm.load_state_dict(anchor_model.lstm.state_dict())
    freeze_anchor_head(model)
    return model


def train_sequence_epoch(
    model: HardCoulombLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    progress = tqdm(loader, desc=f"Stage 2 sequence {epoch}/{total_epochs}", leave=False, dynamic_ncols=True)
    for X_batch, y_batch, I_batch in progress:
        X_batch = X_batch.to(device, non_blocking=True)
        y_target = y_batch.unsqueeze(-1).to(device, non_blocking=True)
        I_batch = I_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(X_batch, I_batch)
        loss = criterion(pred, y_target)
        loss.backward()
        nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
        progress.set_postfix(loss=f"{loss.item():.6f}")
    return total_loss / max(batches, 1)


def validate_sequence(
    model: HardCoulombLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for X_batch, y_batch, I_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_target = y_batch.unsqueeze(-1).to(device, non_blocking=True)
            I_batch = I_batch.to(device, non_blocking=True)
            pred = model(X_batch, I_batch)
            loss = criterion(pred, y_target)
            total_loss += float(loss.item())
            batches += 1
    return total_loss / max(batches, 1)


def train_delta_stage(
    anchor_model: OneStepAnchorPath,
    splits: SplitBundle,
    batch_size: int,
    epochs: int,
    lr: float,
    patience: int,
    seed: int,
    device: torch.device,
    train_source: str,
    warm_start_lstm: bool,
) -> Tuple[HardCoulombLSTM, Path, List[Dict[str, Any]]]:
    model = initialize_stage2_model(anchor_model, warm_start_lstm, device)
    train_loader = make_loader(splits.X_train, splits.y_train, splits.I_train, batch_size, True, seed, device)
    val_loader = make_loader(splits.X_val, splits.y_val, splits.I_val, batch_size, False, seed, device)
    criterion = nn.MSELoss()
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    wait = 0
    history: List[Dict[str, Any]] = []

    print("\nSTAGE 2: Delta/head fine-tuning with frozen anchor_head")
    print(f"  Anchor head frozen : {not any(p.requires_grad for p in model.anchor_head.parameters())}")
    print(f"  LSTM warm start    : {warm_start_lstm}")
    print(f"  Trainable params   : {count_trainable(model):,}")
    print("  Objective          : MSE(HardCoulombLSTM(X, I), y[:, :])")

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = train_sequence_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs)
        val_loss = validate_sequence(model, val_loader, criterion, device)
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
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        marker = " BEST" if improved else ""
        print(
            f"  Seq epoch {epoch:03d}/{epochs} | train={train_loss:.6f} | "
            f"val={val_loss:.6f} | lr={record['lr']:.2e} | {record['time_sec']:.1f}s{marker}"
        )
        if wait >= patience:
            print(f"  Sequence early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    freeze_anchor_head(model)
    path = OUTPUT_DIR / f"stage2_hard_coulomb_anchor_frozen_{train_source}.pt"
    torch.save(
        {
            "stage": "stage2_hard_coulomb_anchor_frozen",
            "train_source": train_source,
            "model_kind": "hard_coulomb_lstm",
            "model_state_dict": model.state_dict(),
            "best_val_loss": best_val,
            "history": history,
            "anchor_head_frozen": True,
            "lstm_warm_start": warm_start_lstm,
            "config": {
                "num_inputs": NUM_INPUTS,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "safety_factor": SAFETY_FACTOR,
            },
        },
        path,
    )
    write_history(OUTPUT_DIR / f"stage2_sequence_history_{train_source}.csv", history)
    print(f"  Stage 2 best val loss: {best_val:.6f}")
    print(f"  Stage 2 checkpoint   : {path.relative_to(BASE_DIR)}")
    return model, path, history


def compute_pvr(y_pred: np.ndarray, I_unscaled: np.ndarray) -> Dict[str, float | int]:
    delta_soc = y_pred[:, 1:] - y_pred[:, :-1]
    discharge_mask = I_unscaled[:, 1:] < DISCHARGE_THRESHOLD_A
    violations = (delta_soc > 0.0) & discharge_mask
    discharge_steps = int(discharge_mask.sum())
    violation_count = int(violations.sum())
    pvr_pct = 0.0 if discharge_steps == 0 else (violation_count / discharge_steps) * 100.0
    return {
        "pvr_pct": float(pvr_pct),
        "violations": violation_count,
        "discharge_steps": discharge_steps,
    }


def error_stats_pct(values_fraction: np.ndarray) -> Dict[str, float]:
    values_pct = values_fraction.astype(np.float64) * 100.0
    return {
        "mae_pct": float(np.mean(np.abs(values_pct))),
        "rmse_pct": float(np.sqrt(np.mean(values_pct**2))),
        "p95_pct": float(np.percentile(np.abs(values_pct), 95)),
        "maxe_pct": float(np.max(np.abs(values_pct))),
    }


def evaluate_test_set(
    model: HardCoulombLSTM,
    scenario_key: str,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    split = load_test_split(SCENARIO_DATA_DIRS[scenario_key], scenario_key)
    loader = make_loader(split.X_test, split.y_test, split.I_test, batch_size, False, 0, device)
    model.eval()
    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for X_batch, _y_batch, I_batch in tqdm(
            loader,
            desc=f"Evaluate {scenario_key}",
            leave=False,
            dynamic_ncols=True,
        ):
            X_batch = X_batch.to(device, non_blocking=True)
            I_batch = I_batch.to(device, non_blocking=True)
            pred = model(X_batch, I_batch)
            predictions.append(pred.detach().cpu().numpy())

    y_pred = np.concatenate(predictions, axis=0).squeeze(-1)
    y_true = split.y_test
    errors = y_pred - y_true
    t0_errors = y_pred[:, 0] - y_true[:, 0]
    pvr = compute_pvr(y_pred, split.I_test)
    maxe = float(np.max(np.abs(errors)) * 100.0)
    baseline = BASELINE_S48_MAXE.get(scenario_key)
    return {
        "scenario": scenario_key,
        "n_windows": int(y_true.shape[0]),
        "rmse_pct": float(np.sqrt(np.mean(errors**2)) * 100.0),
        "mae_pct": float(np.mean(np.abs(errors)) * 100.0),
        "maxe_pct": maxe,
        "t0": error_stats_pct(t0_errors),
        "pvr_pct": pvr["pvr_pct"],
        "pvr_violations": pvr["violations"],
        "pvr_discharge_steps": pvr["discharge_steps"],
        "sprint48_maxe_baseline_pct": baseline,
        "maxe_delta_vs_sprint48_pct": None if baseline is None else maxe - baseline,
        "reduced_below_sprint48_maxe": None if baseline is None else maxe < baseline,
    }


def print_eval_report(results: List[Dict[str, Any]]) -> None:
    print("\nSTAGE 3: Test-set evaluation")
    print("  " + "-" * 106)
    print(
        "  Scenario   | RMSE (%) | MAE (%) | MaxE (%) | t=0 RMSE | t=0 MaxE | "
        "PVR (%) | ΔMaxE vs S48"
    )
    print("  " + "-" * 106)
    for result in results:
        delta = result["maxe_delta_vs_sprint48_pct"]
        delta_text = "n/a" if delta is None else f"{delta:+.4f}"
        print(
            f"  {result['scenario']:<10} | "
            f"{result['rmse_pct']:>8.4f} | "
            f"{result['mae_pct']:>7.4f} | "
            f"{result['maxe_pct']:>8.4f} | "
            f"{result['t0']['rmse_pct']:>8.4f} | "
            f"{result['t0']['maxe_pct']:>8.4f} | "
            f"{result['pvr_pct']:>7.6f} | "
            f"{delta_text:>12}"
        )
        if result["sprint48_maxe_baseline_pct"] is not None:
            verdict = "REDUCED" if result["reduced_below_sprint48_maxe"] else "NOT REDUCED"
            print(
                f"    {result['scenario']}: Sprint 48 MaxE baseline="
                f"{result['sprint48_maxe_baseline_pct']:.4f}% -> {verdict}"
            )
        print(
            f"    {result['scenario']}: PVR violations="
            f"{int(result['pvr_violations']):,}/{int(result['pvr_discharge_steps']):,}; "
            f"t=0 MAE={result['t0']['mae_pct']:.4f}%, t=0 P95={result['t0']['p95_pct']:.4f}%"
        )
    print("  " + "-" * 106)


def write_history(path: Path, history: List[Dict[str, Any]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 49 strict two-stage anchor pretraining.")
    parser.add_argument(
        "--train-source",
        choices=["scenario_A", "scenario_B", "combined"],
        default="scenario_A",
        help="Training source. scenario_A preserves the Scenario A OOD test interpretation.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--anchor-epochs", type=int, default=100)
    parser.add_argument("--sequence-epochs", type=int, default=100)
    parser.add_argument("--anchor-lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--sequence-lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--no-lstm-warm-start",
        action="store_true",
        help="Only transfer anchor_head. Default also transfers the one-step pretrained LSTM.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(OUTPUT_DIR / "sprint49_anchor_pretraining_results.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_utf8_stdio()
    set_reproducibility(args.seed)
    device = resolve_device(args.device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("  Sprint 49 -- Strict Two-Stage Anchor Pretraining")
    print("=" * 96)
    print(f"  Device             : {device}")
    print(f"  Train source       : {args.train_source}")
    print(f"  Batch size         : {args.batch_size}")
    print(f"  Anchor epochs      : {args.anchor_epochs}")
    print(f"  Sequence epochs    : {args.sequence_epochs}")
    print(f"  LSTM warm start    : {not args.no_lstm_warm_start}")
    print(f"  Leakage guard      : train/val splits only until final Stage 3 test evaluation")

    splits = load_train_val_source(args.train_source)
    print(
        f"  Train arrays       : X={splits.X_train.shape}, y={splits.y_train.shape}, I={splits.I_train.shape}"
    )
    print(f"  Val arrays         : X={splits.X_val.shape}, y={splits.y_val.shape}, I={splits.I_val.shape}")

    anchor_model, stage1_path, stage1_history = train_anchor_stage(
        splits=splits,
        batch_size=args.batch_size,
        epochs=args.anchor_epochs,
        lr=args.anchor_lr,
        patience=args.patience,
        seed=args.seed,
        device=device,
        train_source=args.train_source,
    )

    sequence_model, stage2_path, stage2_history = train_delta_stage(
        anchor_model=anchor_model,
        splits=splits,
        batch_size=args.batch_size,
        epochs=args.sequence_epochs,
        lr=args.sequence_lr,
        patience=args.patience,
        seed=args.seed,
        device=device,
        train_source=args.train_source,
        warm_start_lstm=not args.no_lstm_warm_start,
    )

    eval_results = [
        evaluate_test_set(sequence_model, "scenario_A", args.batch_size, device),
        evaluate_test_set(sequence_model, "scenario_B", args.batch_size, device),
    ]
    print_eval_report(eval_results)

    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path
    payload = {
        "train_source": args.train_source,
        "stage1_checkpoint": str(stage1_path.relative_to(BASE_DIR)),
        "stage2_checkpoint": str(stage2_path.relative_to(BASE_DIR)),
        "lstm_warm_start": not args.no_lstm_warm_start,
        "anchor_head_frozen": True,
        "stage1_epochs_ran": len(stage1_history),
        "stage2_epochs_ran": len(stage2_history),
        "evaluation": eval_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\n  Results saved: {output_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
