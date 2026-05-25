"""
web_app.py -- Interactive BMS Dashboard
=======================================

Streamlit dashboard for running the contextual Hard-Coulomb LSTM on raw LG HG2
battery CSV profiles. The app reuses the project preprocessing logic:

  - strict 1 Hz profile conversion
  - V_proxy = Voltage - Current * R_int(T)
  - causal 60 s contextual history features
  - physics scaling from the leak-free preprocessing pipeline

Run:
    streamlit run src/web_app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import streamlit as st
except ModuleNotFoundError as exc:  # pragma: no cover - Streamlit runtime guard
    raise SystemExit(
        "Streamlit is required to run this dashboard. Install it with: pip install streamlit"
    ) from exc

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:
    go = None
    make_subplots = None

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import PHYS_MAX_V3, PHYS_MIN_V3, R_INT_PER_TEMP  # noqa: E402
from model_v6_contextual import ContextualHardCoulombLSTM  # noqa: E402
from preprocessing_v4 import (  # noqa: E402
    DATA_RAW,
    FEATURE_COLS_V4,
    Q_ACTUAL_PER_TEMP,
    WINDOW,
    build_ocv_soc_lookup,
    engineer_features_v4,
    read_csv,
    to_strict_1hz_segments,
)
from preprocessing_v5_contextual import (  # noqa: E402
    ANCHOR_CTX_COLS,
    OCV_CTX_INDICES,
    add_history_features,
    add_ocv_rest_features,
    scale_anchor_context,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = BASE_DIR / "outputs" / "v5_contextual" / "sprint50_contextual" / "history_only.pt"
DISCHARGE_THRESHOLD_A = -0.05
LOOKBACK_SEC = 60
MAX_INFERENCE_POINTS = 1200

TEMP_VALUES: Dict[str, float] = {
    "n20degC": -20.0,
    "n10degC": -10.0,
    "0degC": 0.0,
    "10degC": 10.0,
    "25degC": 25.0,
    "40degC": 40.0,
}

SAMPLE_PROFILES = {
    "-20°C OOD Cold Start": Path(DATA_RAW) / "n20degC" / "610_Mixed1.csv",
    "25°C Normal": Path(DATA_RAW) / "25degC" / "551_Mixed1.csv",
    "0°C Cold Normal": Path(DATA_RAW) / "0degC" / "589_Mixed1.csv",
    "40°C Hot OOD": Path(DATA_RAW) / "40degC" / "556_Mixed1.csv",
}


st.set_page_config(
    page_title="Safety-Constrained BMS",
    page_icon="🔋",
    layout="wide",
)


def pct(value: float) -> float:
    return float(value) * 100.0


def find_nearest_temp_name(temperature_c: float) -> str:
    return min(TEMP_VALUES, key=lambda name: abs(TEMP_VALUES[name] - float(temperature_c)))


def infer_temperature_name(df: pd.DataFrame, source_path: Optional[Path] = None) -> str:
    if source_path is not None:
        for part in source_path.parts:
            if part in R_INT_PER_TEMP:
                return part
    if "Temperature" in df.columns and df["Temperature"].notna().any():
        return find_nearest_temp_name(float(df["Temperature"].median()))
    return "25degC"


def validate_raw_columns(df: pd.DataFrame) -> None:
    missing = [column for column in ["Voltage", "Current"] if column not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {missing}. "
            "Expected at least Voltage and Current; Temperature/Capacity are optional."
        )
    if "Temperature" not in df.columns:
        df["Temperature"] = 25.0
    if "timestamp_ns" not in df.columns:
        df["timestamp_ns"] = (df["time_sec"].to_numpy(dtype=np.float64) * 1_000_000_000).astype(np.int64)


@st.cache_resource(show_spinner=False)
def load_model() -> Tuple[ContextualHardCoulombLSTM, Dict]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint.get("config", {})
    model = ContextualHardCoulombLSTM(
        num_inputs=int(config.get("num_inputs", 5)),
        anchor_ctx_dim=int(config.get("anchor_ctx_dim", len(ANCHOR_CTX_COLS))),
        hidden_size=int(config.get("hidden_size", 64)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.2)),
        safety_factor=float(config.get("safety_factor", 1.5)),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def read_uploaded_profile(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)
    try:
        return read_csv(str(tmp_path))
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def read_sample_profile(sample_name: str) -> Tuple[pd.DataFrame, Path]:
    sample_path = SAMPLE_PROFILES[sample_name]
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample profile not found: {sample_path}")
    return read_csv(str(sample_path)), sample_path


def choose_longest_segment(segments) -> pd.DataFrame:
    if not segments:
        raise ValueError("No strict 1 Hz segment with enough samples was found.")
    return max(segments, key=len).reset_index(drop=True)


def scale_sequence_features(features: np.ndarray) -> np.ndarray:
    phys_min = np.asarray(PHYS_MIN_V3, dtype=np.float32).reshape(1, -1)
    phys_max = np.asarray(PHYS_MAX_V3, dtype=np.float32).reshape(1, -1)
    scaled = (features.astype(np.float32) - phys_min) / (phys_max - phys_min)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def prepare_profile(
    raw_df: pd.DataFrame,
    source_path: Optional[Path],
    max_points: int,
) -> Dict:
    validate_raw_columns(raw_df)
    temp_name = infer_temperature_name(raw_df, source_path)
    r_int = float(R_INT_PER_TEMP.get(temp_name, R_INT_PER_TEMP["25degC"]))

    segments, _, _ = to_strict_1hz_segments(
        raw_df,
        source_id=source_path.stem if source_path is not None else "uploaded_profile",
        profile_code_start=1,
        min_len=max(WINDOW, LOOKBACK_SEC + 2),
    )
    segment = choose_longest_segment(segments)

    ocv_lookup, q_actual = build_ocv_soc_lookup(temp_name)
    q_actual = float(Q_ACTUAL_PER_TEMP.get(temp_name, q_actual))
    engineered, _ = engineer_features_v4(segment, q_actual=q_actual, r_int=r_int, ocv_lookup=ocv_lookup)

    contextual = add_ocv_rest_features(engineered)
    contextual = add_history_features(contextual)

    valid_history = contextual["ctx_hist_valid_60s"].to_numpy(dtype=np.float32)
    valid_indices = np.flatnonzero(valid_history >= 1.0)
    if valid_indices.size == 0:
        start_idx = min(LOOKBACK_SEC, len(contextual) - 1)
        st.warning(
            "No fully valid 60 s history context was found. "
            "Using the earliest possible timestep with zero-filled history context."
        )
    else:
        start_idx = int(valid_indices[0])

    end_idx = min(len(contextual), start_idx + max_points)
    if end_idx - start_idx < 2:
        raise ValueError("The selected profile is too short for sequence inference after contextual history extraction.")

    features_raw = contextual.loc[start_idx:end_idx - 1, FEATURE_COLS_V4].to_numpy(dtype=np.float32)
    X_scaled = scale_sequence_features(features_raw)
    current = contextual.loc[start_idx:end_idx - 1, "Current"].to_numpy(dtype=np.float32)
    y_true = contextual.loc[start_idx:end_idx - 1, "SOC_cc"].to_numpy(dtype=np.float32)

    anchor_raw = contextual.loc[[start_idx], ANCHOR_CTX_COLS].to_numpy(dtype=np.float32)
    anchor_scaled = scale_anchor_context(anchor_raw)
    anchor_scaled[:, OCV_CTX_INDICES] = 0.0  # finalized history-only contextual anchor

    time_sec = contextual.loc[start_idx:end_idx - 1, "time_sec"].to_numpy(dtype=np.float32)
    if len(time_sec):
        time_sec = time_sec - time_sec[0]

    return {
        "df": contextual.iloc[start_idx:end_idx].copy().reset_index(drop=True),
        "temp_name": temp_name,
        "r_int": r_int,
        "q_actual": q_actual,
        "X_scaled": X_scaled,
        "current": current,
        "y_true": y_true,
        "anchor_ctx": anchor_scaled.astype(np.float32),
        "time_sec": time_sec,
        "source_rows": len(raw_df),
        "strict_rows": len(segment),
        "start_idx": start_idx,
    }


def model_forward_with_audit(
    model: ContextualHardCoulombLSTM,
    X_scaled: np.ndarray,
    current: np.ndarray,
    anchor_ctx: np.ndarray,
) -> Dict:
    X_tensor = torch.from_numpy(X_scaled[None, :, :]).to(DEVICE)
    I_tensor = torch.from_numpy(current[None, :]).to(DEVICE)
    A_tensor = torch.from_numpy(anchor_ctx).to(DEVICE)

    with torch.no_grad():
        hidden, _ = model.lstm(X_tensor)
        delta_raw = model.delta_head(hidden)
        context_embedding = model.anchor_ctx_encoder(A_tensor)
        anchor_input = torch.cat([hidden[:, 0, :], context_embedding], dim=-1)
        soc_anchor = model.anchor_head(anchor_input)
        y_pred = model.hard_constraint(delta_raw, I_tensor, soc_anchor)

    pred = y_pred.detach().cpu().numpy().squeeze(0).squeeze(-1)
    raw_delta = delta_raw.detach().cpu().numpy().squeeze(0).squeeze(-1)
    anchor = float(soc_anchor.detach().cpu().numpy().squeeze())

    delta_pred = pred[1:] - pred[:-1]
    discharge_mask = current[1:] < DISCHARGE_THRESHOLD_A
    violations = (delta_pred > 1e-8) & discharge_mask

    raw_discharge_mask = current < DISCHARGE_THRESHOLD_A
    raw_illegal = (raw_delta > 0.0) & raw_discharge_mask

    discharge_steps = int(discharge_mask.sum())
    violation_count = int(violations.sum())
    raw_illegal_count = int(raw_illegal.sum())
    pvr_pct = 0.0 if discharge_steps == 0 else violation_count / discharge_steps * 100.0

    return {
        "pred": pred,
        "raw_delta": raw_delta,
        "anchor": anchor,
        "pvr_pct": pvr_pct,
        "violations": violation_count,
        "discharge_steps": discharge_steps,
        "prevented": raw_illegal_count,
    }


def compute_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.shape != y_pred.shape or y_true.size == 0:
        return {"rmse_pct": float("nan"), "maxe_pct": float("nan"), "mae_pct": float("nan")}
    errors = y_pred - y_true
    return {
        "rmse_pct": pct(np.sqrt(np.mean(errors**2))),
        "maxe_pct": pct(np.max(np.abs(errors))),
        "mae_pct": pct(np.mean(np.abs(errors))),
    }


def render_voltage_plot(df: pd.DataFrame, time_sec: np.ndarray):
    if go is None:
        st.line_chart(pd.DataFrame({"Voltage": df["Voltage"], "V_proxy": df["V_proxy"]}, index=time_sec))
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_sec, y=df["Voltage"], mode="lines", name="Terminal Voltage"))
    fig.add_trace(go.Scatter(x=time_sec, y=df["V_proxy"], mode="lines", name="Proxy Voltage"))
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=35, b=10),
        title="Voltage and Ohmic-Drop-Corrected Proxy",
        xaxis_title="Time (s)",
        yaxis_title="Voltage (V)",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_current_plot(df: pd.DataFrame, time_sec: np.ndarray):
    if go is None:
        st.line_chart(pd.DataFrame({"Current": df["Current"]}, index=time_sec))
        return
    current = df["Current"].to_numpy(dtype=np.float32)
    colors = np.where(current < DISCHARGE_THRESHOLD_A, "#d62728", np.where(current > 0.05, "#2ca02c", "#7f7f7f"))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_sec,
            y=current,
            mode="lines",
            name="Current",
            line=dict(color="#1f77b4"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_sec,
            y=current,
            mode="markers",
            marker=dict(color=colors, size=4),
            name="discharge/charge/rest",
            showlegend=False,
        )
    )
    fig.add_hline(y=DISCHARGE_THRESHOLD_A, line_dash="dash", line_color="#d62728", annotation_text="Discharge threshold")
    fig.add_hline(y=0.05, line_dash="dash", line_color="#2ca02c", annotation_text="Charge threshold")
    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=35, b=10),
        title="Current Profile",
        xaxis_title="Time (s)",
        yaxis_title="Current (A)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_soc_plot(time_sec: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, anchor: float):
    if go is None:
        st.line_chart(pd.DataFrame({"True SOC": y_true * 100.0, "Predicted SOC": y_pred * 100.0}, index=time_sec))
        st.caption(f"Anchor prediction: {anchor * 100.0:.2f}%")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_sec, y=y_true * 100.0, mode="lines", name="True SOC", line=dict(width=3)))
    fig.add_trace(
        go.Scatter(
            x=time_sec,
            y=y_pred * 100.0,
            mode="lines",
            name="Predicted SOC",
            line=dict(width=3, color="#ff7f0e"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[time_sec[0]],
            y=[anchor * 100.0],
            mode="markers+text",
            text=["Anchor"],
            textposition="top center",
            marker=dict(size=13, color="#9467bd", symbol="diamond"),
            name="Anchor start",
        )
    )
    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=40, b=10),
        title="True SOC vs Predicted SOC — Hard-Coulomb Bounded Trajectory",
        xaxis_title="Time (s)",
        yaxis_title="SOC (%)",
        yaxis=dict(range=[-2, 102]),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("Safety-Constrained BMS: Contextual Hard-Coulomb LSTM")
    st.caption(
        "Interactive inference dashboard for the history-only contextual Hard-Coulomb LSTM. "
        "The model uses V_proxy, causal 60 s history context, and a structural Coulomb-bound layer."
    )

    with st.sidebar:
        st.header("Input Profile")
        uploaded_file = st.file_uploader("Upload raw battery CSV profile", type=["csv"])
        available_samples = {name: path for name, path in SAMPLE_PROFILES.items() if path.exists()}
        if available_samples:
            sample_name = st.selectbox(
                "Or select a pre-loaded sample profile",
                options=list(available_samples.keys()),
                index=0,
            )
        else:
            sample_name = None
            st.warning("No bundled sample profiles were found. Please upload a raw CSV.")
        max_points = st.slider("Max inference timesteps", min_value=100, max_value=3000, value=MAX_INFERENCE_POINTS, step=100)
        run_clicked = st.button("Run Inference", type="primary", use_container_width=True)

        st.divider()
        st.subheader("Model")
        st.write(f"Checkpoint: `{CHECKPOINT_PATH.relative_to(BASE_DIR)}`")
        st.write(f"Device: `{DEVICE}`")

    if not run_clicked:
        st.info("Upload a CSV or choose a sample profile, then click **Run Inference**.")
        return

    try:
        model, checkpoint = load_model()
        if uploaded_file is not None:
            raw_df = read_uploaded_profile(uploaded_file)
            source_path = None
            source_label = uploaded_file.name
        else:
            if not available_samples:
                st.error("No sample profiles were found. Upload a raw CSV profile to continue.")
                return
            raw_df, source_path = read_sample_profile(sample_name)
            source_label = f"{sample_name} ({source_path.name})"

        with st.spinner("Preprocessing raw profile and running Hard-Coulomb inference..."):
            prepared = prepare_profile(raw_df, source_path, max_points=max_points)
            audit = model_forward_with_audit(
                model,
                prepared["X_scaled"],
                prepared["current"],
                prepared["anchor_ctx"],
            )
            metrics = compute_error_metrics(prepared["y_true"], audit["pred"])

    except Exception as exc:  # Streamlit should show readable failures, not tracebacks by default.
        st.error(f"Inference failed: {exc}")
        st.stop()

    st.success(f"Inference complete for **{source_label}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperature Map", prepared["temp_name"], f"R_int={prepared['r_int'] * 1000:.2f} mΩ")
    c2.metric("Processed Rows", f"{prepared['strict_rows']:,}", f"raw={prepared['source_rows']:,}")
    c3.metric("Checkpoint", checkpoint.get("label", "Contextual HC-LSTM"))
    c4.metric("Anchor SOC", f"{audit['anchor'] * 100.0:.2f}%")

    left, right = st.columns(2)
    with left:
        render_voltage_plot(prepared["df"], prepared["time_sec"])
    with right:
        render_current_plot(prepared["df"], prepared["time_sec"])

    render_soc_plot(prepared["time_sec"], prepared["y_true"], audit["pred"], audit["anchor"])

    st.subheader("Safety Audit Panel")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Timesteps", f"{len(audit['pred']):,}")
    m2.metric("RMSE", "N/A" if np.isnan(metrics["rmse_pct"]) else f"{metrics['rmse_pct']:.3f}%")
    m3.metric("MaxE", "N/A" if np.isnan(metrics["maxe_pct"]) else f"{metrics['maxe_pct']:.3f}%")
    m4.metric(
        "PVR",
        f"{audit['pvr_pct']:.2f}% (Structurally Guaranteed)",
        f"{audit['violations']:,}/{audit['discharge_steps']:,} violations",
    )

    st.info(f"Hard-Coulomb bounds prevented {audit['prevented']:,} illegal state transitions.")

    with st.expander("Technical details"):
        st.write(
            {
                "feature_columns": FEATURE_COLS_V4,
                "anchor_context_columns": ANCHOR_CTX_COLS,
                "anchor_context_mode": "history_only (OCV/rest indices zeroed)",
                "start_index_after_history": int(prepared["start_idx"]),
                "q_actual_Ah": prepared["q_actual"],
                "pvr_rule": "predicted SOC increase while I_unscaled < -0.05 A",
            }
        )


if __name__ == "__main__":
    main()
