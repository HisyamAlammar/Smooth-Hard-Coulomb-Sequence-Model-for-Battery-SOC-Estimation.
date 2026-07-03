"""
posthoc_clamp.py -- Baseline B: trained Vanilla LSTM + inference-time
Hard-Coulomb envelope (no retraining).

Tests whether the proposed contribution is a *trainable layer* or reducible to
a three-line output filter: take the already-trained vanilla predictions,
convert them to per-step deltas, and route/clamp each delta with the exact
same rule the Hard-Coulomb layer uses:

    I[t] < -thr : delta in [-eta*gamma*|I[t]|, 0]
    I[t] > +thr : delta in [0, +eta*gamma*|I[t]|]
    |I[t]|<=thr : delta = 0

Anchor = vanilla prediction at t=0 (clipped to [0,1]). Reconstruction is a
cumulative sum, then a monotone [0,1] clip (order-preserving, so it cannot
create sign violations).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "src"), str(BASE_DIR / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import CURRENT_THRESHOLD_A, Q_NOMINAL  # noqa: E402
from predict_utils import predict_checkpoint, provenance  # noqa: E402
from soc_metrics import evaluate_soc_predictions  # noqa: E402

RESULTS_DIR = BASE_DIR / "results" / "baselines"
GAMMA = 1.0 / (Q_NOMINAL * 3600.0)  # SOC fraction per Amp-second, same as model
DEFAULT_ETA = 1.5                   # same safety factor as HardCoulombLSTM


def apply_hard_coulomb_clamp(
    y_pred: np.ndarray,
    I: np.ndarray,
    eta: float = DEFAULT_ETA,
    threshold: float = CURRENT_THRESHOLD_A,
) -> np.ndarray:
    """Route/clamp vanilla per-step deltas through the Hard-Coulomb envelope."""
    anchor = np.clip(y_pred[:, 0], 0.0, 1.0)
    delta = np.diff(y_pred, axis=1)              # delta[t-1] = y[t] - y[t-1]
    I_step = I[:, 1:]                            # classified by I at the arrival step
    limit = eta * GAMMA * np.abs(I_step)

    clamped = np.zeros_like(delta)
    disch = I_step < -threshold
    charge = I_step > threshold
    clamped[disch] = np.clip(delta[disch], -limit[disch], 0.0)
    clamped[charge] = np.clip(delta[charge], 0.0, limit[charge])
    # rest stays exactly 0, matching the model's rest routing

    traj = np.concatenate([anchor[:, None], anchor[:, None] + np.cumsum(clamped, axis=1)], axis=1)
    return np.clip(traj, 0.0, 1.0)


def run_scenario(scenario: str, eta: float) -> Dict[str, object]:
    run = predict_checkpoint(scenario, "vanilla_lstm")
    b = run["bundle"]
    y_clamped = apply_hard_coulomb_clamp(run["y_pred"], b.I, eta=eta)

    metrics = evaluate_soc_predictions(b.y_true, y_clamped, b.I, b.temp_labels)
    raw_metrics = evaluate_soc_predictions(b.y_true, run["y_pred"], b.I, b.temp_labels)

    reg, raw = metrics["regression"], raw_metrics["regression"]
    print(f"  {scenario} vanilla raw    RMSE {raw['rmse_full_pct']:7.4f}% | "
          f"PVR(disch,0) {raw_metrics['pvr']['discharge']['by_epsilon']['0']['rate_pct']:.4f}%")
    print(f"  {scenario} vanilla+clamp  RMSE {reg['rmse_full_pct']:7.4f}% | "
          f"MAE {reg['mae_full_pct']:7.4f}% | MaxE {reg['maxe_full_pct']:7.4f}% | "
          f"PVR(disch,0) {metrics['pvr']['discharge']['by_epsilon']['0']['rate_pct']:.4f}%")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RESULTS_DIR / f"posthoc_clamp_predictions_{scenario}.npz",
        clamped=y_clamped.astype(np.float32),
    )
    return {
        "provenance": provenance(scenario, run["checkpoint"], {
            "model": "vanilla_lstm + posthoc hard-coulomb clamp",
            "eta": eta, "gamma_per_A_s": GAMMA,
        }),
        "clamped": metrics,
        "vanilla_raw_reference": raw_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Vanilla LSTM + post-hoc Hard-Coulomb clamp baseline.")
    parser.add_argument("--scenarios", nargs="+", default=["scenario_A", "scenario_B"])
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA)
    parser.add_argument("--output", default=str(RESULTS_DIR / "posthoc_clamp_results.json"))
    args = parser.parse_args()

    results = {s: run_scenario(s, args.eta) for s in args.scenarios}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
