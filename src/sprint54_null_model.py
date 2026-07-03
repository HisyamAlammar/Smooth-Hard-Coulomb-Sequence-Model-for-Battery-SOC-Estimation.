# Sprint 54: no-neural OCV anchor + pure Coulomb counting baseline.

from __future__ import annotations

import numpy as np

from baseline_eval_common import (
    BASE_DIR,
    SCENARIO_DATA_DIRS,
    TEMPERATURE_ORDER,
    build_result_record,
    load_metadata,
    load_ocv_lookups,
    ocv_anchor_from_vproxy,
    q_actual_for_labels,
    save_json,
    setup_script,
    unscale_features,
)
from sprint48_common import load_test_split


def predict_null_soc(X_scaled, I_unscaled, temp_labels, metadata, data_dir):
    X_raw = unscale_features(X_scaled, data_dir)
    v_proxy_first = X_raw[:, 0, 0].astype(np.float32)
    q = q_actual_for_labels(temp_labels, metadata['q_actual'])
    lookups = load_ocv_lookups(np.unique(temp_labels))
    anchor = ocv_anchor_from_vproxy(v_proxy_first, temp_labels, lookups)
    pred = np.empty_like(I_unscaled, dtype=np.float32)
    pred[:, 0] = anchor
    delta = I_unscaled[:, 1:] / (q[:, None] * 3600.0)
    pred[:, 1:] = anchor[:, None] + np.cumsum(delta, axis=1)
    return np.clip(pred, 0.0, 1.0).astype(np.float32)


def evaluate_scenario(scenario_key: str):
    data_dir = SCENARIO_DATA_DIRS[scenario_key]
    metadata = load_metadata(data_dir)
    test = load_test_split(data_dir, scenario_key)
    if test.temp_labels is None:
        raise RuntimeError('temp_labels_test.npy is required for q_actual routing')
    y_pred = predict_null_soc(test.X_test, test.I_test, test.temp_labels, metadata, data_dir)
    return build_result_record(
        scenario_key,
        'null_ocv_coulomb_counting',
        'Null OCV + Coulomb Counting',
        test.y_test,
        y_pred,
        test.I_test,
        test.temp_labels,
        data_dir,
        {
            'parameter_count': 0,
            'anchor': 'OCV_lookup(V_proxy_first_sample)',
            'integration': 'pure current integration with Q_actual(T), dt=1s',
            'temperature_order': TEMPERATURE_ORDER,
        },
    )


def main() -> None:
    setup_script()
    results = [evaluate_scenario('scenario_A'), evaluate_scenario('scenario_B')]
    out_path = BASE_DIR / 'outputs' / 'sprint54_null_model_results.json'
    save_json(out_path, results)
    for row in results:
        m = row['metrics']
        print(row['scenario'], 'Null', 'RMSE', round(m['rmse_full_pct'], 4), 'MaxE', round(m['maxe_full_pct'], 4), 'PVR', round(m['pvr_pct'], 6))
    print('Saved:', out_path.relative_to(BASE_DIR))


if __name__ == '__main__':
    main()
