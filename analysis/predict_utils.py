"""
predict_utils.py -- Shared checkpoint inference + provenance helpers.

Used by all audit-fix phases so every experiment runs the same forward pass
and records the same provenance block (split, scenario, checkpoint, seed,
timestamp, config values).
"""

from __future__ import annotations

import datetime
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
for p in (str(BASE_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import CURRENT_THRESHOLD_A, PVR_EPSILONS, Q_NOMINAL, RANDOM_SEED  # noqa: E402
from sprint48_common import (  # noqa: E402
    checkpoint_path,
    forward_model,
    load_checkpoint,
    load_test_split,
    resolve_device,
)

SCENARIO_DATA_DIRS = {
    "scenario_A": BASE_DIR / "data" / "processed" / "v4_scenario_A",
    "scenario_B": BASE_DIR / "data" / "processed" / "v4_scenario_B",
}

# v4 physics scaling bounds for X columns [V_proxy, I, T, dVp/dt, dI/dt];
# needed to recover unscaled physical signals from stored windows.
from config import PHYS_MAX_V3, PHYS_MIN_V3  # noqa: E402

X_MIN = np.asarray(PHYS_MIN_V3, dtype=np.float32)
X_MAX = np.asarray(PHYS_MAX_V3, dtype=np.float32)


def unscale_feature(X: np.ndarray, col: int) -> np.ndarray:
    """Invert the fixed physics min-max scaling for one feature column."""
    return X[:, :, col] * (X_MAX[col] - X_MIN[col]) + X_MIN[col]


def scale_feature(values: np.ndarray, col: int) -> np.ndarray:
    return (values - X_MIN[col]) / (X_MAX[col] - X_MIN[col])


@dataclass(frozen=True)
class TestBundle:
    scenario: str
    X: np.ndarray                    # scaled features (N, T, 5)
    y_true: np.ndarray               # SOC labels (N, T)
    I: np.ndarray                    # unscaled current (N, T)
    temp_labels: np.ndarray | None   # (N,) str or None
    timestamp_keys: np.ndarray | None  # (N, T) int64 or None


def load_test_bundle(scenario: str) -> TestBundle:
    data_dir = SCENARIO_DATA_DIRS[scenario]
    split = load_test_split(data_dir, scenario)
    key_path = data_dir / "timestamp_key_test.npy"
    keys = np.load(key_path) if key_path.exists() else None
    return TestBundle(
        scenario=scenario,
        X=split.X_test,
        y_true=split.y_test,
        I=split.I_test,
        temp_labels=split.temp_labels,
        timestamp_keys=keys,
    )


def predict_checkpoint(
    scenario: str,
    model_kind: str,
    device: torch.device | None = None,
    batch_size: int = 1024,
) -> Dict[str, Any]:
    """Run a finalized v7 checkpoint over its matching test split."""
    device = device or resolve_device(None)
    bundle = load_test_bundle(scenario)
    ckpt = checkpoint_path(model_kind, scenario, latest=False)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    model, payload = load_checkpoint(ckpt, device)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(bundle.X), batch_size):
            xb = torch.from_numpy(bundle.X[i : i + batch_size]).to(device)
            ib = torch.from_numpy(bundle.I[i : i + batch_size]).to(device)
            preds.append(forward_model(model, model_kind, xb, ib).cpu().numpy())
    y_pred = np.concatenate(preds, axis=0).squeeze(-1)
    return {
        "bundle": bundle,
        "model": model,
        "y_pred": y_pred,
        "checkpoint": str(ckpt.relative_to(BASE_DIR)),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "model_kind": model_kind,
    }


def provenance(scenario: str, checkpoint: str | None = None, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    prov = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "scenario": scenario,
        "split": "test",
        "data_source": str(SCENARIO_DATA_DIRS[scenario].relative_to(BASE_DIR)),
        "checkpoint": checkpoint,
        "seed": RANDOM_SEED,
        "config": {
            "current_threshold_A": CURRENT_THRESHOLD_A,
            "pvr_epsilons": list(PVR_EPSILONS),
            "q_nominal_Ah": Q_NOMINAL,
        },
    }
    if extra:
        prov.update(extra)
    return prov
