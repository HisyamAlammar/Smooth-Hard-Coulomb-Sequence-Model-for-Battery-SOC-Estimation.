"""
sprint48_train_scenario_B.py -- Final isolated Scenario B training.

Scenario B uses only data/processed/v4_scenario_B:
  - Train/val/test: chronological splits created before windowing
  - Test evaluation is performed separately by sprint48_evaluate_all.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sprint48_common import (
    BASE_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    PATIENCE,
    RANDOM_SEED,
    configure_utf8_stdio,
    run_scenario_training,
)

SCENARIO_KEY = "scenario_B"
SCENARIO_LABEL = "Scenario B (chronological split-before-windowing)"
DATA_DIR = BASE_DIR / "data" / "processed" / "v4_scenario_B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final v7 models on Scenario B only.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_utf8_stdio()
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    run_scenario_training(
        scenario_key=SCENARIO_KEY,
        scenario_label=SCENARIO_LABEL,
        data_dir=Path(DATA_DIR),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed,
        device_arg=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
