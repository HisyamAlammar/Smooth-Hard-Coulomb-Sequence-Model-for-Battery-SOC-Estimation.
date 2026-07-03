# Sprint 56: replace Hard-Coulomb anchor with true SOC(t=0) on -20 C windows.

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from baseline_eval_common import BASE_DIR, SCENARIO_DATA_DIRS, build_result_record, save_json, setup_script
from sprint48_common import BATCH_SIZE, checkpoint_path, load_checkpoint, load_test_split, resolve_device


def hard_coulomb_pred_and_delta(scenario_key: str, batch_size: int = BATCH_SIZE, device_arg: str | None = None):
    device = resolve_device(device_arg)
    data_dir = SCENARIO_DATA_DIRS[scenario_key]
    test = load_test_split(data_dir, scenario_key)
    model, payload = load_checkpoint(checkpoint_path('hard_coulomb_lstm', scenario_key), device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(test.X_test), torch.from_numpy(test.y_test), torch.from_numpy(test.I_test)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    preds, deltas = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, _y_batch, I_batch in tqdm(loader, desc=f'Oracle delta {scenario_key}', leave=False, dynamic_ncols=True):
            X_batch = X_batch.to(device, non_blocking=True)
            I_batch = I_batch.to(device, non_blocking=True)
            y_pred, delta = model(X_batch, I_batch, return_delta=True)
            preds.append(y_pred.cpu().numpy().squeeze(-1))
            deltas.append(delta.cpu().numpy().squeeze(-1))
    return np.concatenate(preds).astype(np.float32), np.concatenate(deltas).astype(np.float32), test, payload


def evaluate_scenario(scenario_key: str):
    data_dir = SCENARIO_DATA_DIRS[scenario_key]
    y_model, delta, test, payload = hard_coulomb_pred_and_delta(scenario_key)
    if test.temp_labels is None:
        raise RuntimeError('temp_labels_test.npy is required for -20 C filtering')
    mask = test.temp_labels == 'n20degC'
    if not mask.any():
        raise RuntimeError('No n20degC windows found in ' + scenario_key)
    y_true = test.y_test[mask]
    I = test.I_test[mask]
    labels = test.temp_labels[mask]
    y_model_20 = y_model[mask]
    delta_20 = delta[mask]
    y_oracle = y_true[:, [0]] + np.cumsum(delta_20, axis=1)
    y_oracle = np.clip(y_oracle, 0.0, 1.0).astype(np.float32)
    base = build_result_record(scenario_key, 'hard_coulomb_lstm_original_anchor_n20', 'Hard-Coulomb LSTM Original Anchor -20C', y_true, y_model_20, I, labels, data_dir)
    oracle = build_result_record(scenario_key, 'hard_coulomb_lstm_oracle_anchor_n20', 'Hard-Coulomb LSTM Oracle Anchor -20C', y_true, y_oracle, I, labels, data_dir)
    for rec in (base, oracle):
        rec['checkpoint'] = str(checkpoint_path('hard_coulomb_lstm', scenario_key).relative_to(BASE_DIR))
        rec['checkpoint_epoch'] = int(payload.get('epoch', -1))
        rec['n20_windows'] = int(mask.sum())
    return {'scenario': scenario_key, 'original_anchor': base, 'oracle_anchor': oracle}


def main() -> None:
    setup_script()
    results = [evaluate_scenario('scenario_A'), evaluate_scenario('scenario_B')]
    out_path = BASE_DIR / 'outputs' / 'sprint56_oracle_anchor_results.json'
    save_json(out_path, results)
    for row in results:
        m0 = row['original_anchor']['metrics']
        m1 = row['oracle_anchor']['metrics']
        print(row['scenario'], 'n20 original MaxE', round(m0['maxe_full_pct'], 4), 'oracle MaxE', round(m1['maxe_full_pct'], 4), 'oracle PVR', round(m1['pvr_pct'], 6))
    print('Saved:', out_path.relative_to(BASE_DIR))


if __name__ == '__main__':
    main()
