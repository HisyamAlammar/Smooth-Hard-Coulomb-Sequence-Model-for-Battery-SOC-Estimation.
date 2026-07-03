"""
test_soc_metrics.py -- Sanity checks for the shared metrics module.

Run directly: python analysis/test_soc_metrics.py
Plain asserts (no pytest dependency). Each case encodes one audit finding.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from soc_metrics import (  # noqa: E402
    delta_magnitude_metrics,
    evaluate_soc_predictions,
    legacy_pvr,
    pvr_metrics,
    region_masks,
)

THR = 0.05


def make_current(T=6):
    # per-step current: [discharge, discharge, rest, charge, charge] classified at I[1:]
    return np.array([[-1.0, -1.0, -1.0, 0.0, 1.0, 1.0]], dtype=np.float32)


def test_frozen_output_passes_pvr_but_fails_delta_tracking():
    I = make_current()
    y_true = np.array([[0.9, 0.899, 0.898, 0.898, 0.899, 0.9]], dtype=np.float32)
    y_frozen = np.full_like(y_true, 0.5)
    pvr = pvr_metrics(y_frozen, I, THR)
    for region in ("discharge", "charge", "rest"):
        assert pvr[region]["by_epsilon"]["0"]["rate_pct"] == 0.0, region
    dm = delta_magnitude_metrics(y_frozen, y_true, I, THR)
    assert dm["all"]["pred_true_delta_ratio"] == 0.0
    assert dm["all"]["delta_soc_mae"] > 0.0
    print("PASS frozen output: PVR 0 in all regions, delta ratio 0 (audit finding 2)")


def test_sign_flip_triggers_pvr():
    I = make_current()
    # SOC rises during discharge steps and falls during charge steps
    y_bad = np.array([[0.5, 0.51, 0.52, 0.52, 0.51, 0.50]], dtype=np.float32)
    pvr = pvr_metrics(y_bad, I, THR)
    assert pvr["discharge"]["by_epsilon"]["0"]["rate_pct"] == 100.0
    assert pvr["charge"]["by_epsilon"]["0"]["rate_pct"] == 100.0
    print("PASS sign flip: discharge and charge violations both detected")


def test_epsilon_deadband_filters_small_wiggles():
    I = np.full((1, 5), -1.0, dtype=np.float32)
    y = np.array([[0.5, 0.5001, 0.5, 0.51, 0.5]], dtype=np.float32)  # +0.0001 wiggle, +0.01 real
    pvr = pvr_metrics(y, I, THR, epsilons=(0.0, 0.0005, 0.02))
    d = pvr["discharge"]["by_epsilon"]
    assert d["0"]["violations"] == 2
    assert d["0.0005"]["violations"] == 1     # wiggle filtered, real one kept
    assert d["0.02"]["violations"] == 0
    print("PASS epsilon dead-band separates float wiggles from real violations")


def test_rest_drift_detected():
    I = np.zeros((1, 5), dtype=np.float32)
    y = np.array([[0.5, 0.49, 0.48, 0.47, 0.46]], dtype=np.float32)  # drift at rest
    pvr = pvr_metrics(y, I, THR)
    assert pvr["rest"]["by_epsilon"]["0"]["rate_pct"] == 100.0
    assert pvr["discharge"]["n_steps"] == 0 and "note" in pvr["discharge"]
    print("PASS rest drift detected; empty regions logged, not crashed")


def test_legacy_matches_old_definition():
    rng = np.random.default_rng(0)
    y = rng.random((8, 50)).astype(np.float32)
    I = (rng.random((8, 50)).astype(np.float32) - 0.5) * 4
    old_delta = y[:, 1:] - y[:, :-1]
    old_mask = I[:, 1:] < -0.05
    old_viol = int(((old_delta > 0) & old_mask).sum())
    new = legacy_pvr(y, I, 0.05)
    assert new["violations"] == old_viol
    assert new["discharge_steps"] == int(old_mask.sum())
    print("PASS legacy_pvr reproduces the pre-audit discharge-only definition")


def test_masks_partition_all_steps():
    rng = np.random.default_rng(1)
    I = (rng.random((4, 30)).astype(np.float32) - 0.5) * 2
    masks = region_masks(I, THR)
    total = sum(int(m.sum()) for m in masks.values())
    assert total == I[:, 1:].size
    print("PASS region masks partition every delta step exactly once")


def test_bundle_runs_end_to_end():
    rng = np.random.default_rng(2)
    y_true = np.clip(rng.random((6, 20)), 0, 1).astype(np.float32)
    y_pred = np.clip(y_true + rng.normal(0, 0.01, y_true.shape), 0, 1).astype(np.float32)
    I = (rng.random((6, 20)).astype(np.float32) - 0.5) * 4
    temps = np.array(["25degC", "25degC", "25degC", "n20degC", "n20degC", "n20degC"])
    b = evaluate_soc_predictions(y_true, y_pred, I, temps)
    assert "per_temperature" in b and set(b["per_temperature"]) == {"25degC", "n20degC"}
    assert b["regression"]["rmse_full_pct"] > 0
    print("PASS full bundle incl. per-temperature split")


if __name__ == "__main__":
    test_frozen_output_passes_pvr_but_fails_delta_tracking()
    test_sign_flip_triggers_pvr()
    test_epsilon_deadband_filters_small_wiggles()
    test_rest_drift_detected()
    test_legacy_matches_old_definition()
    test_masks_partition_all_steps()
    test_bundle_runs_end_to_end()
    print("\nALL METRIC SANITY CHECKS PASSED")
