"""
edge_feasibility.py -- Phase 9: parameter-level edge feasibility accounting.

Produces analytic (NOT hardware-measured) compute/memory numbers for the
production HardCoulombLSTM, labeled as parameter-level feasibility only.
WCET/latency on real silicon remains unvalidated and is flagged as such.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "src"), str(BASE_DIR / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

from model_v5_coulomb import HardCoulombLSTM, count_parameters  # noqa: E402
from predict_utils import provenance  # noqa: E402

OUT_DIR = BASE_DIR / "results" / "edge"
H, IN, WINDOW = 64, 5, 100


def main() -> None:
    model = HardCoulombLSTM(num_inputs=IN, hidden_size=H, num_layers=2)
    params = count_parameters(model)

    # per-timestep multiply-accumulates
    lstm_l1 = 4 * H * (IN + H)      # 17,664
    lstm_l2 = 4 * H * (H + H)       # 32,768
    delta_head = H * 32 + 32 * 1    # 2,080
    per_step = lstm_l1 + lstm_l2 + delta_head
    anchor_head = H * 16 + 16 * 1   # once per window
    per_window = per_step * WINDOW + anchor_head + 3 * WINDOW  # + constraint ops

    report = {
        "provenance": provenance("scenario_A", "outputs/v7_final/hard_coulomb_lstm_scenario_A.pt", {
            "experiment": "edge_feasibility",
            "status": "PARAMETER-LEVEL ESTIMATE ONLY -- no WCET/hardware validation performed",
        }),
        "parameters": params,
        "flash_bytes": {"fp32": params * 4, "int8_weights_plus_fp_scales_approx": params + 2048},
        "macs": {
            "per_timestep_stateful": per_step,
            "anchor_head_once_per_window": anchor_head,
            "per_window_full_recompute": per_window,
            "stream_1hz_stateful_macs_per_s": per_step,
            "stream_1hz_sliding_window_macs_per_s": per_window,
        },
        "ram_bytes_estimate": {
            "input_window_buffer_fp32": WINDOW * IN * 4,
            "lstm_states_fp32": 2 * 2 * H * 4,
            "activation_scratch_fp32_approx": 4 * H * 4 * 2,
        },
        "architectural_deployment_constraints": [
            "Constraint layer is non-causal within the window: [lo,hi] anchor "
            "remapping needs min/max of the full window cumsum -> streaming "
            "deployment must either buffer 100 s (anchor latency) or recompute "
            "the full window each second (per_window MACs/s).",
            "Per-window re-anchoring produces output discontinuities between "
            "consecutive windows (no slew-rate analysis exists).",
            "Phase 6 recursive inference removes re-anchoring but accumulates "
            "delta-path rate error at warm temperatures.",
            "Sigmoid/tanh: 8 activations per timestep; CMSIS-NN LUT cost not measured.",
        ],
        "nonclaims": [
            "No WCET measurement", "No hardware latency profile",
            "No int8 full-integer deployment artifact", "No RAM measurement on target",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "edge_feasibility_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("parameters", "flash_bytes", "macs")}, indent=1))
    print(f"Saved to {OUT_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
