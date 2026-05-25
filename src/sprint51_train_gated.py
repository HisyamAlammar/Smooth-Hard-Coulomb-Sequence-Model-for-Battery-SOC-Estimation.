"""
sprint51_train_gated.py -- Sprint 51 Gated Context Anchor
=========================================================

Trains GatedContextualHardCoulombLSTM on the same Scenario A v5_contextual
splits used by Sprint 50, preserving Sprint 50 files and writing only to:

    outputs/v7_gated_context/sprint51/

Evaluation reports:
  - Global Scenario A test metrics
  - -20°C OOD metrics
  - Valid-rest subgroup metrics
  - Invalid-rest subgroup metrics
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
from typing import Any, Dict, List

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
from model_v7_gated_context import (  # noqa: E402
    ANCHOR_CTX_DIM,
    GatedContextualHardCoulombLSTM,
    OCV_VALID_INDEX,
    count_parameters,
)
from preprocessing_v5_contextual import ANCHOR_CTX_COLS  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

DATA_DIR = BASE_DIR / "data" / "processed" / "v5_contextual" / "scenario_A"
OUTPUT_DIR = BASE_DIR / "outputs" / "v7_gated_context" / "sprint51"
TARGET_TEMP = "n20degC"
DISCHARGE_THRESHOLD_A = -0.05


@dataclass(frozen=True)
class ContextualData:
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


def resolve_device(device_arg: str | None = None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_array(name: str) -> np.ndarray:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing v5_contextual array: {path}")
    return np.load(path).astype(np.float32, copy=False)


def load_contextual_data() -> ContextualData:
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Missing {DATA_DIR}. Run src/preprocessing_v5_contextual.py before Sprint 51 training."
        )
    temp_path = DATA_DIR / "temp_labels_test.npy"
    if not temp_path.exists():
        raise FileNotFoundError(f"Missing test temperature labels: {temp_path}")

    data = ContextualData(
        X_train=load_array("X_train.npy"),
        A_train=load_array("A_anchor_train.npy"),
        I_train=load_array("I_unscaled_train.npy"),
        y_train=load_array("y_train.npy"),
        X_val=load_array("X_val.npy"),
        A_val=load_array("A_anchor_val.npy"),
        I_val=load_array("I_unscaled_val.npy"),
        y_val=load_array("y_val.npy"),
        X_test=load_array("X_test.npy"),
        A_test=load_array("A_anchor_test.npy"),
        I_test=load_array("I_unscaled_test.npy"),
        y_test=load_array("y_test.npy"),
        temp_test=np.load(temp_path, allow_pickle=True),
    )
    validate_shapes(data)
    return data


def validate_shapes(data: ContextualData) -> None:
    for split_name in ["train", "val", "test"]:
        X_seq = getattr(data, f"X_{split_name}")
        A_anchor = getattr(data, f"A_{split_name}")
        current = getattr(data, f"I_{split_name}")
        targets = getattr(data, f"y_{split_name}")
        if X_seq.ndim != 3 or X_seq.shape[1:] != (100, NUM_INPUTS):
            raise ValueError(f"{split_name}: bad X shape {X_seq.shape}")
        if A_anchor.ndim != 2 or A_anchor.shape[1] != ANCHOR_CTX_DIM:
            raise ValueError(f"{split_name}: bad A shape {A_anchor.shape}")
        if current.shape != targets.shape or targets.ndim != 2 or targets.shape[1] != 100:
            raise ValueError(f"{split_name}: bad I/y shapes {current.shape}, {targets.shape}")
        if X_seq.shape[0] != A_anchor.shape[0] or X_seq.shape[0] != targets.shape[0]:
            raise ValueError(f"{split_name}: mismatched sample counts")


def make_loader(
    X_seq: np.ndarray,
    A_anchor: np.ndarray,
    current: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(X_seq),
        torch.from_numpy(A_anchor),
        torch.from_numpy(current),
        torch.from_numpy(targets),
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


def train_epoch(
    model: GatedContextualHardCoulombLSTM,
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
    progress = tqdm(loader, desc=f"Sprint51 gated {epoch}/{total_epochs}", leave=False, dynamic_ncols=True)
    for X_batch, A_batch, I_batch, y_batch in progress:
        X_batch = X_batch.to(device, non_blocking=True)
        A_batch = A_batch.to(device, non_blocking=True)
        I_batch = I_batch.to(device, non_blocking=True)
        y_target = y_batch.unsqueeze(-1).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        y_pred = model(X_batch, I_batch, A_batch)
        loss = criterion(y_pred, y_target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        batches += 1
        progress.set_postfix(loss=f"{loss.item():.6f}")
    return total_loss / max(batches, 1)


def validate_loss(
    model: GatedContextualHardCoulombLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for X_batch, A_batch, I_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            A_batch = A_batch.to(device, non_blocking=True)
            I_batch = I_batch.to(device, non_blocking=True)
            y_target = y_batch.unsqueeze(-1).to(device, non_blocking=True)
            y_pred = model(X_batch, I_batch, A_batch)
            loss = criterion(y_pred, y_target)
            total_loss += float(loss.item())
            batches += 1
    return total_loss / max(batches, 1)


def train_model(
    data: ContextualData,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    patience: int,
    seed: int,
    device: torch.device,
) -> tuple[GatedContextualHardCoulombLSTM, Dict[str, Any]]:
    train_loader = make_loader(data.X_train, data.A_train, data.I_train, data.y_train, batch_size, True, seed, device)
    val_loader = make_loader(data.X_val, data.A_val, data.I_val, data.y_val, batch_size, False, seed, device)

    model = GatedContextualHardCoulombLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)

    best_state = None
    best_val = float("inf")
    wait = 0
    history: List[Dict[str, Any]] = []

    print("\nTraining Sprint 51 gated-context model")
    print(f"  Model params          : {count_parameters(model):,}")
    print("  OCV/history context   : full context")
    print("  Invalid OCV rest gate : hard-forced to zero")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, epochs)
        val_loss = validate_loss(model, val_loader, criterion, device)
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": round(time.time() - start_time, 2),
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
    checkpoint_path = OUTPUT_DIR / "gated_context_hard_coulomb_scenario_A.pt"
    torch.save(
        {
            "model_kind": "gated_contextual_hard_coulomb_lstm",
            "model_state_dict": model.state_dict(),
            "best_val_loss": best_val,
            "history": history,
            "anchor_ctx_cols": ANCHOR_CTX_COLS,
            "config": {
                "num_inputs": NUM_INPUTS,
                "anchor_ctx_dim": ANCHOR_CTX_DIM,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "safety_factor": 1.5,
            },
        },
        checkpoint_path,
    )
    write_history(OUTPUT_DIR / "gated_context_history.csv", history)
    return model, {
        "best_val_loss": best_val,
        "epochs_ran": len(history),
        "checkpoint": str(checkpoint_path.relative_to(BASE_DIR)),
    }


def compute_pvr(y_pred: np.ndarray, current: np.ndarray) -> Dict[str, float | int]:
    delta_soc = y_pred[:, 1:] - y_pred[:, :-1]
    discharge_mask = current[:, 1:] < DISCHARGE_THRESHOLD_A
    violations = (delta_soc > 0.0) & discharge_mask
    discharge_steps = int(discharge_mask.sum())
    violation_count = int(violations.sum())
    return {
        "pvr_pct": 0.0 if discharge_steps == 0 else float(violation_count / discharge_steps * 100.0),
        "violations": violation_count,
        "discharge_steps": discharge_steps,
    }


def predict(
    model: GatedContextualHardCoulombLSTM,
    X_seq: np.ndarray,
    A_anchor: np.ndarray,
    current: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    loader = make_loader(X_seq, A_anchor, current, targets, batch_size, False, 0, device)
    model.eval()
    predictions: List[np.ndarray] = []
    raw_gates: List[np.ndarray] = []
    effective_gates: List[np.ndarray] = []
    valid_flags: List[np.ndarray] = []

    with torch.no_grad():
        for X_batch, A_batch, I_batch, _y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            A_batch = A_batch.to(device, non_blocking=True)
            I_batch = I_batch.to(device, non_blocking=True)
            y_pred, aux = model(X_batch, I_batch, A_batch, return_aux=True)
            predictions.append(y_pred.detach().cpu().numpy())
            raw_gates.append(aux["raw_gate"].detach().cpu().numpy())
            effective_gates.append(aux["effective_gate"].detach().cpu().numpy())
            valid_flags.append(aux["ocv_valid"].detach().cpu().numpy())

    return np.concatenate(predictions, axis=0).squeeze(-1), {
        "raw_gate": np.concatenate(raw_gates, axis=0).squeeze(-1),
        "effective_gate": np.concatenate(effective_gates, axis=0).squeeze(-1),
        "ocv_valid": np.concatenate(valid_flags, axis=0).squeeze(-1),
    }


def metric_block(
    label: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    current: np.ndarray,
    raw_gate: np.ndarray,
    effective_gate: np.ndarray,
    valid_flags: np.ndarray,
) -> Dict[str, Any]:
    if len(y_true) == 0:
        return {
            "label": label,
            "n_windows": 0,
            "t0_rmse_pct": None,
            "t0_maxe_pct": None,
            "full_rmse_pct": None,
            "full_maxe_pct": None,
            "pvr_pct": None,
            "pvr_violations": 0,
            "pvr_discharge_steps": 0,
            "raw_gate_mean": None,
            "effective_gate_mean": None,
            "valid_rate_pct": None,
        }

    errors = y_pred - y_true
    t0_errors = y_pred[:, 0] - y_true[:, 0]
    pvr = compute_pvr(y_pred, current)
    return {
        "label": label,
        "n_windows": int(y_true.shape[0]),
        "t0_rmse_pct": float(np.sqrt(np.mean(t0_errors**2)) * 100.0),
        "t0_maxe_pct": float(np.max(np.abs(t0_errors)) * 100.0),
        "full_rmse_pct": float(np.sqrt(np.mean(errors**2)) * 100.0),
        "full_maxe_pct": float(np.max(np.abs(errors)) * 100.0),
        "pvr_pct": pvr["pvr_pct"],
        "pvr_violations": pvr["violations"],
        "pvr_discharge_steps": pvr["discharge_steps"],
        "raw_gate_mean": float(np.mean(raw_gate)),
        "effective_gate_mean": float(np.mean(effective_gate)),
        "valid_rate_pct": float(np.mean(valid_flags >= 0.5) * 100.0),
    }


def evaluate(
    model: GatedContextualHardCoulombLSTM,
    data: ContextualData,
    batch_size: int,
    device: torch.device,
    target_temp: str,
) -> Dict[str, Any]:
    y_pred_all, aux = predict(model, data.X_test, data.A_test, data.I_test, data.y_test, batch_size, device)
    valid_mask = data.A_test[:, OCV_VALID_INDEX] >= 0.5
    temp_mask = data.temp_test == target_temp

    def select(mask: np.ndarray, label: str) -> Dict[str, Any]:
        return metric_block(
            label=label,
            y_true=data.y_test[mask],
            y_pred=y_pred_all[mask],
            current=data.I_test[mask],
            raw_gate=aux["raw_gate"][mask],
            effective_gate=aux["effective_gate"][mask],
            valid_flags=aux["ocv_valid"][mask],
        )

    results = {
        "global": select(np.ones(len(data.y_test), dtype=bool), "Scenario A all-test"),
        "global_valid_rest": select(valid_mask, "All-test valid-rest"),
        "global_invalid_rest": select(~valid_mask, "All-test invalid-rest"),
        "target_temp": target_temp,
        "target": select(temp_mask, f"{target_temp} all"),
        "target_valid_rest": select(temp_mask & valid_mask, f"{target_temp} valid-rest"),
        "target_invalid_rest": select(temp_mask & ~valid_mask, f"{target_temp} invalid-rest"),
    }
    return results


def fmt_metric(value: Any, width: int = 8, precision: int = 4) -> str:
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value:>{width}.{precision}f}"


def print_report(results: Dict[str, Any]) -> None:
    print("\nSprint 51 Gated Context Evaluation")
    print("  " + "-" * 124)
    print(
        "  Group                  | N      | t=0 RMSE | t=0 MaxE | Full RMSE | Full MaxE | PVR (%) | Gate raw | Gate eff"
    )
    print("  " + "-" * 124)
    for key in [
        "global",
        "global_valid_rest",
        "global_invalid_rest",
        "target",
        "target_valid_rest",
        "target_invalid_rest",
    ]:
        row = results[key]
        print(
            f"  {row['label']:<22} | "
            f"{row['n_windows']:>6,} | "
            f"{fmt_metric(row['t0_rmse_pct'])} | "
            f"{fmt_metric(row['t0_maxe_pct'])} | "
            f"{fmt_metric(row['full_rmse_pct'], 9)} | "
            f"{fmt_metric(row['full_maxe_pct'], 9)} | "
            f"{fmt_metric(row['pvr_pct'], 7, 6)} | "
            f"{fmt_metric(row['raw_gate_mean'], 8, 4)} | "
            f"{fmt_metric(row['effective_gate_mean'], 8, 4)}"
        )
    print("  " + "-" * 124)
    global_row = results["global"]
    print(
        f"  Global MaxE: {global_row['full_maxe_pct']:.4f}% | "
        f"Global PVR: {global_row['pvr_pct']:.6f}% "
        f"({int(global_row['pvr_violations']):,}/{int(global_row['pvr_discharge_steps']):,})"
    )


def write_history(path: Path, history: List[Dict[str, Any]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sprint 51 gated contextual anchor on Scenario A.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--target-temp", type=str, default=TARGET_TEMP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed)
    device = resolve_device(args.device)
    data = load_contextual_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 104)
    print("  Sprint 51 -- Gated Context Anchor")
    print("=" * 104)
    print(f"  Data branch : {DATA_DIR.relative_to(BASE_DIR)}")
    print(f"  Output dir  : {OUTPUT_DIR.relative_to(BASE_DIR)}")
    print(f"  Device      : {device}")
    print(f"  Train split : X={data.X_train.shape}, A={data.A_train.shape}, y={data.y_train.shape}")
    print(f"  Target temp : {args.target_temp}")

    model, summary = train_model(
        data=data,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed,
        device=device,
    )

    results = evaluate(model, data, args.batch_size, device, args.target_temp)
    print_report(results)

    output_payload = {
        "data_branch": str(DATA_DIR.relative_to(BASE_DIR)),
        "target_temp": args.target_temp,
        "anchor_ctx_cols": ANCHOR_CTX_COLS,
        "summary": summary,
        "evaluation": results,
    }
    output_path = OUTPUT_DIR / "sprint51_gated_context_results.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2)
    print(f"\n  Results saved: {output_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
