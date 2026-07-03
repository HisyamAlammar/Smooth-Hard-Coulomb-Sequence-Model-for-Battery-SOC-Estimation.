# Phase 9 — Edge Feasibility and Quantization Report

Date: 2026-07-03. Artifacts: `analysis/edge_feasibility.py`, `experiments/quantized_pvr_check.py`, `results/edge/edge_feasibility_report.json`, `results/edge/quantized_pvr_results.json`.

## Parameter-level feasibility (analytic — explicitly NOT hardware-validated)

| Quantity | Value |
|---|---|
| Parameters | 54,626 |
| Flash FP32 / est. int8 | 218.5 KB / ~56.7 KB |
| MACs per timestep (stateful) | 52,512 |
| MACs per window (full recompute) | 5.25 M |
| Stream @1 Hz — stateful / sliding-window | 0.053 / 5.25 MMAC/s |
| RAM: window buffer + LSTM states | 2.0 KB + 1.0 KB (FP32) |

Compute is trivial for a Cortex-M4-class MCU *if stateful streaming were possible* — but the constraint layer is **non-causal within the window** (the [lo,hi] anchor remap needs the full window's cumsum min/max), so deployment must either accept 100 s anchor latency or recompute the full window each second (5.25 MMAC/s — still feasible on M4/M7, but WCET unmeasured). Per-window re-anchoring also produces output discontinuities with no slew-rate analysis. **No WCET, no target-RAM measurement, no full-integer artifact exist. All edge claims must be labeled "parameter-level feasibility only"** (consistent with draft line 319; the stronger verbal claim "suitable for Edge MCUs" is not supported).

## Does the PVR invariant survive quantization? (measured, 4,096-window subset, seed 42)

| Path | PVR disch ε=0 | Note |
|---|---:|---|
| FP32 reference | 0.00 % | RMSE 12.70 % |
| Dynamic int8 (LSTM+Linear weights) | **0.00 %** | RMSE drift −0.018 pp; but max per-window output shift **4.7 %SOC** (anchor sensitivity) |
| uint8 trajectory quantization | **0.00 %** | any monotone elementwise quantizer preserves sign-consistency (x≤y ⇒ q(x)≤q(y); violations need strict increase) |
| float16 delta accumulation | **0.00 %** | adding a non-positive f16 increment to a representable partial sum cannot round above it |
| Per-timestep asymmetric requantization | **breaks** | constructed counterexample inverts order (0.50199→0.5000 vs 0.50189→0.5016); pipelines that requantize each timestep with independent scale/zero-point CAN emit sign violations |

Conclusions:
1. The invariant is robust to the *benign* quantization paths (weight-only, monotone output grids, f16 accumulation) — stronger than the audit's worst-case worry, and now proven-plus-measured rather than assumed.
2. It is NOT robust to arbitrary integer pipelines: per-timestep requantization with varying scales (as some accelerator toolchains emit) breaks it. Deployment requirement: keep the SOC state in a **single-scale integer accumulator** (int32 Coulomb-counter style), which also matches BMS engineering practice.
3. Dynamic quantization preserved average accuracy but moved individual anchors by up to 4.7 %SOC — quantization must be re-validated per checkpoint, not assumed benign.
