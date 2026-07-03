"""
quantized_pvr_check.py -- Phase 9: does the PVR invariant survive quantization?

Three measured checks on the production HardCoulombLSTM (Scenario A test):

  1. PyTorch dynamic quantization (int8 weights for LSTM/Linear, float
     activations, CPU): the constraint layer stays float, so the invariant
     should survive; accuracy drift is measured. NOTE: this is NOT a
     full-integer MCU deployment -- it validates one quantization path only.

  2. Monotone output quantization: SOC trajectory quantized to a uint8 grid.
     Claim: any monotone elementwise quantizer preserves sign-consistency
     (x<=y => q(x)<=q(y); violations require strict increase). Measured.

  3. float16 accumulation: cumsum of routed deltas computed in float16.
     Nearest rounding of adding a non-positive increment to a representable
     partial sum cannot overshoot above it, so PVR should stay 0. Measured.

  4. Counterexample (constructed): per-timestep asymmetric requantization
     (different scale/zero-point at t-1 and t) CAN invert order -- shows which
     integer pipelines would break the guarantee on target hardware.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "src"), str(BASE_DIR / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

from predict_utils import load_test_bundle, provenance  # noqa: E402
from soc_metrics import pvr_metrics, regression_metrics  # noqa: E402
from sprint48_common import checkpoint_path, load_checkpoint  # noqa: E402

OUT_DIR = BASE_DIR / "results" / "edge"
N_SUBSET = 4096  # CPU inference bound; documented in output


def predict_cpu(model, X, I, batch=256) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i : i + batch])
            ib = torch.from_numpy(I[i : i + batch])
            preds.append(model(xb, ib).squeeze(-1).numpy())
    return np.concatenate(preds, axis=0)


def main() -> None:
    device = torch.device("cpu")
    bundle = load_test_bundle("scenario_A")
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(bundle.X), size=min(N_SUBSET, len(bundle.X)), replace=False))
    X, I, y_true = bundle.X[idx], bundle.I[idx], bundle.y_true[idx]

    model, _ = load_checkpoint(checkpoint_path("hard_coulomb_lstm", "scenario_A", latest=False), device)
    model.eval()

    y_fp32 = predict_cpu(model, X, I)
    m_fp32 = regression_metrics(y_true, y_fp32)
    pvr_fp32 = pvr_metrics(y_fp32, I, epsilons=(0.0,))

    # 1. dynamic int8 quantization of LSTM + Linear weights
    qmodel = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
    )
    y_q = predict_cpu(qmodel, X, I)
    m_q = regression_metrics(y_true, y_q)
    pvr_q = pvr_metrics(y_q, I, epsilons=(0.0,))

    # 2. monotone uint8 quantization of the SOC trajectory
    y_u8 = np.round(y_fp32 * 255.0) / 255.0
    pvr_u8 = pvr_metrics(y_u8.astype(np.float32), I, epsilons=(0.0,))

    # 3. float16 accumulation of routed deltas
    delta = (y_fp32[:, 1:] - y_fp32[:, :-1]).astype(np.float16)
    acc = np.concatenate(
        [y_fp32[:, :1].astype(np.float16),
         y_fp32[:, :1].astype(np.float16) + np.cumsum(delta, axis=1, dtype=np.float16)],
        axis=1,
    ).astype(np.float32)
    pvr_f16 = pvr_metrics(acc, I, epsilons=(0.0,))

    # 4. constructed counterexample: per-timestep requantization with
    # DIFFERENT scales. a rounds down by ~scale_a/2, b rounds up by ~scale_b/2,
    # so the dequantized pair inverts even though b < a.
    a, b = 0.50199, 0.50189  # b < a (discharging step)
    scale_t0, zp_t0 = 0.0040, 0   # timestep t-1 quantizer
    scale_t1, zp_t1 = 0.0038, 0   # timestep t quantizer
    qa = np.round(a / scale_t0) + zp_t0
    qb = np.round(b / scale_t1) + zp_t1
    deq_a, deq_b = (qa - zp_t0) * scale_t0, (qb - zp_t1) * scale_t1
    counterexample_inverts = bool(deq_b > deq_a)

    result = {
        "provenance": provenance("scenario_A",
                                 "outputs/v7_final/hard_coulomb_lstm_scenario_A.pt",
                                 {"experiment": "quantized_pvr_check",
                                  "n_windows_subset": int(len(idx)), "subset_seed": 42,
                                  "scope": "dynamic weight quantization only; NOT a full-integer deployment"}),
        "fp32": {"rmse_pct": m_fp32["rmse_full_pct"],
                 "pvr_disch_eps0": pvr_fp32["discharge"]["by_epsilon"]["0"]["rate_pct"]},
        "dynamic_int8": {"rmse_pct": m_q["rmse_full_pct"],
                         "rmse_drift_pp": m_q["rmse_full_pct"] - m_fp32["rmse_full_pct"],
                         "max_abs_output_diff": float(np.max(np.abs(y_q - y_fp32))),
                         "pvr_disch_eps0": pvr_q["discharge"]["by_epsilon"]["0"]["rate_pct"]},
        "uint8_trajectory_quant": {"pvr_disch_eps0": pvr_u8["discharge"]["by_epsilon"]["0"]["rate_pct"],
                                   "claim": "monotone elementwise quantization preserves sign-consistency"},
        "float16_accumulation": {"pvr_disch_eps0": pvr_f16["discharge"]["by_epsilon"]["0"]["rate_pct"]},
        "asymmetric_requant_counterexample": {
            "inverts_order": counterexample_inverts,
            "detail": f"a={a}, b={b}<a; per-timestep scales {scale_t0}/{scale_t1} with zero-points "
                      f"{zp_t0}/{zp_t1} dequantize to {deq_a:.6f}/{deq_b:.6f}",
            "implication": "integer pipelines that requantize each timestep with independent "
                           "scale/zero-point CAN create sign violations; deployment must use a "
                           "single-scale accumulator (e.g. int32 Coulomb counter) for the SOC state",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "quantized_pvr_results.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
