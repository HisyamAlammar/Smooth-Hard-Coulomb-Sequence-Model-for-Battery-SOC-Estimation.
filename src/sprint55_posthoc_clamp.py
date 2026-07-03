# Sprint 55: inference-only clamp projected onto Vanilla LSTM outputs.

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from baseline_eval_common import (
    BASE_DIR,
    CURRENT_THRESHOLD,
    SCENARIO_DATA_DIRS,
    build_result_record,
    load_metadata,
    q_actual_for_labels,
    save_json,
    setup_script,
)
from sprint48_common import (
    BATCH_SIZE,
    checkpoint_path,
    forward_model,
    load_checkpoint,
    load_test_split,
    resolve_device,
)


def vanilla_predict(scenario_key: str, batch_size: int = BATCH_SIZE, device_arg: str | None = None):
    device = resolve_device(device_arg)
    data_dir = SCENARIO_DATA_DIRS[scenario_key]
    test = load_test_split(data_dir, scenario_key)
    model, payload = load_checkpoint(checkpoint_path('vanilla_lstm', scenario_key), device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(test.X_test), torch.from_numpy(test.y_test), torch.from_numpy(test.I_test)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    chunks = []
    model.eval()
    with torch.no_grad():
        for X_batch, _y_batch, I_batch in tqdm(loader, desc=f'Vanilla {scenario_key}', leave=False, dynamic_ncols=True):
            X_batch = X_batch.to(device, non_blocking=True)
            I_batch = I_batch.to(device, non_blocking=True)
            chunks.append(forward_model(model, 'vanilla_lstm', X_batch, I_batch).cpu().numpy().squeeze(-1))
    return np.concatenate(chunks, axis=0).astype(np.float32), test, payload


def posthoc_project(raw_pred, I_unscaled, temp_labels, q_actual):
    if temp_labels is None:
        raise RuntimeError('temp_labels_test.npy is required for q_actual routing')
    q = q_actual_for_labels(temp_labels, q_actual)
    raw_delta = raw_pred[:, 1:] - raw_pred[:, :-1]
    limit = np.abs(I_unscaled[:, 1:]) / (q[:, None] * 3600.0)
    discharge = I_unscaled[:, 1:] < -CURRENT_THRESHOLD
    charge = I_unscaled[:, 1:] > CURRENT_THRESHOLD
    delta = np.zeros_like(raw_delta, dtype=np.float32)
    delta = np.where(discharge, np.minimum(raw_delta, 0.0), delta)
    delta = np.where(charge, np.maximum(raw_delta, 0.0), delta)
    delta = np.where(discharge, np.maximum(delta, -limit), delta)
    delta = np.where(charge, np.minimum(delta, limit), delta)
    projected = np.empty_like(raw_pred, dtype=np.float32)
    projected[:, 0] = np.clip(raw_pred[:, 0], 0.0, 1.0)
    projected[:, 1:] = projected[:, [0]] + np.cumsum(delta, axis=1)
    return np.clip(projected, 0.0, 1.0).astype(np.float32)


def evaluate_scenario(scenario_key: str):
    data_dir = SCENARIO_DATA_DIRS[scenario_key]
    metadata = load_metadata(data_dir)
    raw_pred, test, payload = vanilla_predict(scenario_key)
    y_pred = posthoc_project(raw_pred, test.I_test, test.temp_labels, metadata['q_actual'])
    return build_result_record(
        scenario_key,
        'posthoc_clamped_vanilla_lstm',
        'Post-hoc Clamp Vanilla LSTM',
        test.y_test,
        y_pred,
        test.I_test,
        test.temp_labels,
        data_dir,
        {
            'source_checkpoint': str(checkpoint_path('vanilla_lstm', scenario_key).relative_to(BASE_DIR)),
            'source_checkpoint_epoch': int(payload.get('epoch', -1)),
            'parameter_count': 53569,
            'projection': 'inference-only directional and Q_actual(T) magnitude clamp',
        },
    )


def main() -> None:
    setup_script()
    results = [evaluate_scenario('scenario_A'), evaluate_scenario('scenario_B')]
    out_path = BASE_DIR / 'outputs' / 'sprint55_posthoc_clamp_results.json'
    save_json(out_path, results)
    for row in results:
        m = row['metrics']
        print(row['scenario'], 'PostHoc', 'RMSE', round(m['rmse_full_pct'], 4), 'MaxE', round(m['maxe_full_pct'], 4), 'PVR', round(m['pvr_pct'], 6))
    print('Saved:', out_path.relative_to(BASE_DIR))


if __name__ == '__main__':
    main()
