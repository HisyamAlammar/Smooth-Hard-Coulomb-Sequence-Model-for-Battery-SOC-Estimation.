# Sprint 57: classical 1-RC ECM EKF baseline for SOC estimation.

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from baseline_eval_common import (
    BASE_DIR,
    SCENARIO_DATA_DIRS,
    build_result_record,
    load_metadata,
    save_json,
    setup_script,
    unscale_features,
)
from config import R_INT_PER_TEMP
from preprocessing_v4 import DATA_RAW, find_header_row
from sprint48_common import load_test_split


def _read_hppc(temp: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA_RAW, temp, '*HPPC*.csv')))
    if not files:
        raise FileNotFoundError('No HPPC CSV found for ' + temp)
    hrow = find_header_row(files[0])
    df = pd.read_csv(files[0], skiprows=hrow, encoding='utf-8', encoding_errors='replace', low_memory=False)
    df.columns = df.columns.str.strip()
    for col in ['Voltage', 'Current', 'Capacity']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    ts = pd.to_datetime(df['Time Stamp'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    if ts.isna().all():
        ts = pd.to_datetime(df['Time Stamp'], errors='coerce', format='mixed')
    if ts.notna().any():
        t0 = ts.dropna().iloc[0]
        df['time_sec'] = (ts - t0).dt.total_seconds()
    else:
        df['time_sec'] = np.arange(len(df), dtype=np.float64)
    return df.dropna(subset=['Voltage', 'Current', 'Capacity', 'time_sec']).reset_index(drop=True)


def build_ocv_soc_model(temp: str):
    df = _read_hppc(temp)
    q_actual = float(df['Capacity'].abs().max())
    is_rest = df['Current'].abs() < 0.01
    rest_groups = (is_rest != is_rest.shift()).cumsum()
    ocv_list, soc_list = [], []
    for _idx, seg in df[is_rest].groupby(rest_groups):
        duration = float(seg['time_sec'].max() - seg['time_sec'].min())
        if duration >= 600.0:
            ocv = float(seg['Voltage'].iloc[-1])
            cap_used = abs(float(seg['Capacity'].iloc[-1]))
            soc = float(np.clip(1.0 - cap_used / q_actual, 0.0, 1.0))
            ocv_list.append(ocv)
            soc_list.append(soc)
    if len(ocv_list) < 4:
        raise RuntimeError('Insufficient rest OCV points for EKF: ' + temp)
    pairs = sorted(zip(soc_list, ocv_list), key=lambda item: item[0])
    soc, ocv = [], []
    for s, v in pairs:
        if not soc or s > soc[-1] + 1e-5:
            soc.append(s)
            ocv.append(v)
    soc_arr = np.array(soc, dtype=np.float64)
    ocv_arr = np.maximum.accumulate(np.array(ocv, dtype=np.float64))
    ocv_arr = ocv_arr + np.arange(len(ocv_arr), dtype=np.float64) * 1e-7
    ocv_from_soc = PchipInterpolator(soc_arr, ocv_arr, extrapolate=True)
    docv_from_soc = ocv_from_soc.derivative()
    inv_soc = PchipInterpolator(ocv_arr, soc_arr, extrapolate=True)
    return ocv_from_soc, docv_from_soc, inv_soc


def ekf_group(v_proxy, current, q_actual, r0, ocv_from_soc, docv_from_soc, inv_soc):
    n, t_len = v_proxy.shape
    pred = np.empty((n, t_len), dtype=np.float32)
    soc = np.clip(inv_soc(v_proxy[:, 0]), 0.0, 1.0).astype(np.float64)
    vrc = np.zeros(n, dtype=np.float64)
    pred[:, 0] = soc.astype(np.float32)

    r1 = max(0.005, 0.60 * float(r0))
    tau = 45.0 + 900.0 * float(r0)
    a = float(np.exp(-1.0 / tau))
    q_soc = 2.5e-6
    q_vrc = 2.5e-6
    r_meas = (0.018 + 0.35 * float(r0)) ** 2
    p00 = np.full(n, 0.0400, dtype=np.float64)
    p01 = np.zeros(n, dtype=np.float64)
    p11 = np.full(n, 0.0100, dtype=np.float64)

    for t in range(1, t_len):
        soc_pred = soc + current[:, t].astype(np.float64) / (float(q_actual) * 3600.0)
        soc_pred = np.clip(soc_pred, -0.20, 1.20)
        vrc_pred = a * vrc + (1.0 - a) * r1 * (-current[:, t].astype(np.float64))

        p00p = p00 + q_soc
        p01p = a * p01
        p11p = a * a * p11 + q_vrc
        soc_eval = np.clip(soc_pred, 0.0, 1.0)
        h0 = np.clip(docv_from_soc(soc_eval), 0.01, 10.0)
        h1 = -1.0
        z_pred = ocv_from_soc(soc_eval) - vrc_pred
        s_cov = h0 * h0 * p00p + 2.0 * h0 * h1 * p01p + h1 * h1 * p11p + r_meas
        s_cov = np.maximum(s_cov, 1e-9)
        k0 = (p00p * h0 + p01p * h1) / s_cov
        k1 = (p01p * h0 + p11p * h1) / s_cov
        resid = np.clip(v_proxy[:, t].astype(np.float64) - z_pred, -0.40, 0.40)

        soc = np.clip(soc_pred + k0 * resid, 0.0, 1.0)
        vrc = np.clip(vrc_pred + k1 * resid, -1.0, 1.0)
        hp0 = h0 * p00p + h1 * p01p
        hp1 = h0 * p01p + h1 * p11p
        p00 = np.maximum(p00p - k0 * hp0, 1e-8)
        p01 = p01p - k0 * hp1
        p11 = np.maximum(p11p - k1 * hp1, 1e-8)
        pred[:, t] = soc.astype(np.float32)
    return pred


def evaluate_scenario(scenario_key: str):
    data_dir = SCENARIO_DATA_DIRS[scenario_key]
    metadata = load_metadata(data_dir)
    test = load_test_split(data_dir, scenario_key)
    if test.temp_labels is None:
        raise RuntimeError('temp_labels_test.npy is required for EKF temperature routing')
    x_raw = unscale_features(test.X_test, data_dir)
    v_proxy = x_raw[:, :, 0].astype(np.float32)
    pred = np.empty_like(test.y_test, dtype=np.float32)
    temp_params = {}
    for temp in sorted(np.unique(test.temp_labels)):
        mask = test.temp_labels == temp
        ocv, docv, inv = build_ocv_soc_model(str(temp))
        r0 = float(R_INT_PER_TEMP[str(temp)])
        q_actual = float(metadata['q_actual'][str(temp)])
        pred[mask] = ekf_group(v_proxy[mask], test.I_test[mask], q_actual, r0, ocv, docv, inv)
        temp_params[str(temp)] = {
            'q_actual_ah': q_actual,
            'r0_ohm': r0,
            'r1_ohm_assumed': max(0.005, 0.60 * r0),
            'tau_sec_assumed': 45.0 + 900.0 * r0,
        }
    return build_result_record(
        scenario_key,
        'ekf_1rc_ecm',
        'EKF 1-RC ECM Baseline',
        test.y_test,
        pred,
        test.I_test,
        test.temp_labels,
        data_dir,
        {
            'parameter_count': 0,
            'ocv_model': 'HPPC rest-derived PCHIP OCV(SOC)',
            'r0_source': 'HPPC R_INT_PER_TEMP from config.py',
            'r1_tau_source': 'assumed temperature-scaled 1-RC ECM parameters',
            'state': ['SOC', 'V_rc'],
            'per_temperature_ecm_parameters': temp_params,
        },
    )


def main() -> None:
    setup_script()
    np.seterr(all='warn')
    results = [evaluate_scenario('scenario_A'), evaluate_scenario('scenario_B')]
    out_path = BASE_DIR / 'outputs' / 'sprint57_ekf_results.json'
    save_json(out_path, results)
    for row in results:
        m = row['metrics']
        print(row['scenario'], 'EKF', 'RMSE', round(m['rmse_full_pct'], 4), 'MaxE', round(m['maxe_full_pct'], 4), 'PVR', round(m['pvr_pct'], 6))
    print('Saved:', out_path.relative_to(BASE_DIR))


if __name__ == '__main__':
    main()
