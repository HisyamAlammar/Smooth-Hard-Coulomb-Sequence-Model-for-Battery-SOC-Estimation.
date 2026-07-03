# Shared utilities for Sprint 54-57 reviewer baselines.

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import BATCH_SIZE, CURRENT_THRESHOLD, Q_NOMINAL  # noqa: E402
from preprocessing_v4 import build_ocv_soc_lookup  # noqa: E402
from sprint48_common import configure_utf8_stdio  # noqa: E402
from sprint48_evaluate_all import compute_metrics, compute_per_temp_last_step  # noqa: E402

SCENARIO_DATA_DIRS = {
    'scenario_A': BASE_DIR / 'data' / 'processed' / 'v4_scenario_A',
    'scenario_B': BASE_DIR / 'data' / 'processed' / 'v4_scenario_B',
}
TEMPERATURE_ORDER = ['40degC', '25degC', '10degC', '0degC', 'n10degC', 'n20degC']


def setup_script() -> None:
    configure_utf8_stdio()
    np.seterr(all='raise')


def load_metadata(data_dir: Path) -> Dict[str, Any]:
    with (data_dir / 'metadata_v4.json').open('r', encoding='utf-8') as handle:
        return json.load(handle)


def unscale_features(X_scaled: np.ndarray, data_dir: Path) -> np.ndarray:
    x_min = np.load(data_dir / 'X_min.npy').astype(np.float32, copy=False)
    x_max = np.load(data_dir / 'X_max.npy').astype(np.float32, copy=False)
    return X_scaled * (x_max - x_min) + x_min


def q_actual_for_labels(temp_labels: np.ndarray, q_actual: Dict[str, float]) -> np.ndarray:
    return np.array([float(q_actual[str(temp)]) for temp in temp_labels], dtype=np.float32)


def load_ocv_lookups(temps: Iterable[str]) -> Dict[str, Callable[[np.ndarray], np.ndarray] | None]:
    lookups: Dict[str, Callable[[np.ndarray], np.ndarray] | None] = {}
    for temp in temps:
        lookup, _q_actual = build_ocv_soc_lookup(str(temp))
        lookups[str(temp)] = lookup
    return lookups


def dense_voltage_to_soc_fallback(v_proxy: np.ndarray) -> np.ndarray:
    return np.clip((v_proxy - 2.5) / (4.2 - 2.5), 0.0, 1.0).astype(np.float32)


def ocv_anchor_from_vproxy(
    v_proxy_first: np.ndarray,
    temp_labels: np.ndarray,
    lookups: Dict[str, Callable[[np.ndarray], np.ndarray] | None],
) -> np.ndarray:
    anchors = np.empty(v_proxy_first.shape[0], dtype=np.float32)
    for temp in np.unique(temp_labels):
        temp_key = str(temp)
        mask = temp_labels == temp
        lookup = lookups.get(temp_key)
        if lookup is None:
            anchors[mask] = dense_voltage_to_soc_fallback(v_proxy_first[mask])
        else:
            anchors[mask] = np.clip(lookup(v_proxy_first[mask]), 0.0, 1.0).astype(np.float32)
    return anchors


def per_temp_full_metrics(y_true: np.ndarray, y_pred: np.ndarray, I: np.ndarray, labels: np.ndarray | None):
    if labels is None or len(labels) != y_true.shape[0]:
        return {}
    out = {}
    for temp in sorted(np.unique(labels)):
        mask = labels == temp
        m = compute_metrics(y_true[mask], y_pred[mask], I[mask])
        out[str(temp)] = {
            'rmse_full_pct': m['rmse_full_pct'],
            'mae_full_pct': m['mae_full_pct'],
            'maxe_full_pct': m['maxe_full_pct'],
            'rmse_last_pct': m['rmse_last_pct'],
            'mae_last_pct': m['mae_last_pct'],
            'maxe_last_pct': m['maxe_last_pct'],
            'pvr_pct': m['pvr_pct'],
            'pvr_violations': m['pvr_violations'],
            'pvr_discharge_steps': m['pvr_discharge_steps'],
            'n_windows': int(mask.sum()),
        }
    return out


def build_result_record(
    scenario_key: str,
    model_kind: str,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    I_unscaled: np.ndarray,
    temp_labels: np.ndarray | None,
    data_dir: Path,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        'scenario': scenario_key,
        'model_kind': model_kind,
        'model_name': model_name,
        'data_source': str(data_dir.relative_to(BASE_DIR)),
        'metrics': compute_metrics(y_true, y_pred, I_unscaled),
        'per_temp_last_step': compute_per_temp_last_step(y_true, y_pred, temp_labels),
        'per_temp_full_sequence': per_temp_full_metrics(y_true, y_pred, I_unscaled, temp_labels),
    }
    if extra:
        record.update(extra)
    return record


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
