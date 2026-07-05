# Empirical Figure Export Report

- SVG removed: 0
- Remaining SVG under figure/output artifact dirs: 0

## GENERATED: Raw UDDS signal and V_proxy preprocessing are plotted from actual LG HG2 CSV files.
- outputs: figures\fig_empirical_01_raw_signal_preprocessing.png, figures\fig_empirical_01_raw_signal_preprocessing.pdf
- sources: data/raw/LG Dataset/LG_HG2_Original_Dataset/25degC/551_UDDS.csv, data/raw/LG Dataset/LG_HG2_Original_Dataset/n20degC/610_UDDS.csv, src/config.py, src/preprocessing_v4.py
- caveat: SOC_actual is derived from recorded Capacity and q_actual table used by preprocessing_v4.

## GENERATED: Scenario A RMSE comparison uses exact means/std where seed data exist.
- outputs: figures\fig_empirical_02_multiseed_baselines.png, figures\fig_empirical_02_multiseed_baselines.pdf
- sources: results/v5/multiseed/multiseed_summary.csv, results/v5/final_v5_model_comparison.csv, results/v5/ekf_ecm/recursive_vs_ekf_comparison.csv
- caveat: Null and EKF rows are deterministic/single-checkpoint artifacts, so no 5-seed std is plotted.

## GENERATED: η* sweep shows recursive RMSE minimum near η*=2.0 and delta-ratio recovery near 1.
- outputs: figures\fig_empirical_03_eta_calibration_curve.png, figures\fig_empirical_03_eta_calibration_curve.pdf
- sources: results/v5/delta_calibration/eta_gamma_sweep.csv
- caveat: Inference sweep uses fixed Scenario A seed 42 weights trained at η=1.5.

## GENERATED: −20 °C SOC trajectory uses fixed checkpoint inference and continuous 1RC EKF mapping.
- outputs: figures\fig_empirical_04_subzero_trajectory_ekf_hc.png, figures\fig_empirical_04_subzero_trajectory_ekf_hc.pdf
- sources: data/processed/v5c_scenario_A/X_test.npy, data/processed/v5c_scenario_A/y_test.npy, data/processed/v5c_scenario_A/I_unscaled_test.npy, data/processed/v5c_scenario_A/temp_labels_test.npy, data/processed/v5c_scenario_A/timestamp_key_test.npy, results/v5/headline_models/checkpoints/hard_coulomb_lstm_v5c_scenario_A_seed42.pt, results/v5/delta_calibration/eta_gamma_sweep.csv, results/v5/ekf_ecm/recursive_vs_ekf_comparison.csv
- caveat: Window index 16832; selected by largest EKF-minus-HC MAE gap among n20degC test windows. No training rerun.

## GENERATED: Training and validation loss curves are plotted from saved HC-LSTM and HC-TCN logs.
- outputs: figures\fig_empirical_05_training_curves_anchor_last.png, figures\fig_empirical_05_training_curves_anchor_last.pdf
- sources: outputs\v7_final\logs\training_log_hard_coulomb_lstm_scenario_A.csv, outputs\v7_final\logs\training_log_hard_coulomb_lstm_scenario_B.csv, logs\training_log_v5_coulomb_tcn_scenario_A.csv, logs\training_log_v5_coulomb_tcn_scenario_B.csv
- caveat: Logs are saved training artifacts; no training rerun. HC-LSTM rows use v7_final logs, HC-TCN rows use v5 Coulomb TCN logs.

## Validation
- {'file': 'figures\\fig_empirical_01_raw_signal_preprocessing.png', 'exists': True, 'bytes': 375832, 'width': 2130, 'height': 1230, 'pixel_std': 53.90050581402285, 'blank': False}
- {'file': 'figures\\fig_empirical_01_raw_signal_preprocessing.pdf', 'exists': True, 'bytes': 66077}
- {'file': 'figures\\fig_empirical_02_multiseed_baselines.png', 'exists': True, 'bytes': 105755, 'width': 2010, 'height': 930, 'pixel_std': 59.73552346714131, 'blank': False}
- {'file': 'figures\\fig_empirical_02_multiseed_baselines.pdf', 'exists': True, 'bytes': 26785}
- {'file': 'figures\\fig_empirical_03_eta_calibration_curve.png', 'exists': True, 'bytes': 133369, 'width': 1949, 'height': 990, 'pixel_std': 24.165394587624686, 'blank': False}
- {'file': 'figures\\fig_empirical_03_eta_calibration_curve.pdf', 'exists': True, 'bytes': 20570}
- {'file': 'figures\\fig_empirical_04_subzero_trajectory_ekf_hc.png', 'exists': True, 'bytes': 74736, 'width': 1980, 'height': 990, 'pixel_std': 27.428593461967345, 'blank': False}
- {'file': 'figures\\fig_empirical_04_subzero_trajectory_ekf_hc.pdf', 'exists': True, 'bytes': 23203}
- {'file': 'figures\\fig_empirical_05_training_curves_anchor_last.png', 'exists': True, 'bytes': 196656, 'width': 2069, 'height': 1380, 'pixel_std': 34.33749076627472, 'blank': False}
- {'file': 'figures\\fig_empirical_05_training_curves_anchor_last.pdf', 'exists': True, 'bytes': 25571}
