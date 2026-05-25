"""
sprint45_investigate_maxe.py — Root Cause Analysis for v4-LSTM MaxE Anomaly
=============================================================================
Forensic script: finds the exact sequence responsible for the ~99.90% MaxE
in Scenario A (Full OOD) and extracts a full diagnostic report.

Hypotheses to test:
  H1. Anchor Failure  — anchor_head predicts wildly wrong SOC_0
  H2. Cumulative Drift — delta_head accumulates signed error over 100 steps
  H3. Clamp Saturation — final clamp(0,1) masks a runaway trajectory

Usage: python src/sprint45_investigate_maxe.py
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_PROC, OUTPUT_MOD, OUTPUT_FIG,
    NUM_INPUTS, BATCH_SIZE, PHYS_MIN_V3, PHYS_MAX_V3,
    CURRENT_THRESHOLD,
)
from model_v4_lstm import HybridPhysicsLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PHYS_MIN = np.array(PHYS_MIN_V3, dtype=np.float32)
PHYS_MAX = np.array(PHYS_MAX_V3, dtype=np.float32)
PHYS_RNG = PHYS_MAX - PHYS_MIN


def log(msg):
    print(msg, flush=True)


def unscale_features(X_scaled_seq):
    """Unscale a single (100, 5) scaled sequence back to physical units."""
    return X_scaled_seq * PHYS_RNG.reshape(1, 5) + PHYS_MIN.reshape(1, 5)


def main():
    log("=" * 70)
    log("  Sprint 45 — MaxE Root Cause Analysis (v4-LSTM, Scenario A)")
    log("=" * 70)

    # ── 1. Load checkpoint ───────────────────────────────────────────
    ckpt_path = os.path.join(OUTPUT_MOD, "hybrid_v4_lstm_scenario_A.pt")
    if not os.path.exists(ckpt_path):
        log(f"  FATAL: Checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})
    model = HybridPhysicsLSTM(
        num_inputs=cfg.get("num_inputs", NUM_INPUTS),
        hidden_size=cfg.get("hidden_size", 64),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.2),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # ── 2. Load test data ────────────────────────────────────────────
    d = os.path.join(DATA_PROC, "v3_scenario_A")
    X_test = np.load(os.path.join(d, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(d, "y_test.npy")).astype(np.float32)
    I_test = np.load(os.path.join(d, "I_unscaled_test.npy")).astype(np.float32)
    temp_labels = np.load(os.path.join(d, "temp_labels_test.npy"), allow_pickle=True)

    N, T = y_test.shape
    log(f"  Test set: N={N:,}, T={T}, temps={np.unique(temp_labels)}")

    # ── 3. Full inference ────────────────────────────────────────────
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_test),
            torch.from_numpy(y_test),
            torch.from_numpy(I_test),
        ),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    all_preds = []
    with torch.no_grad():
        for X_b, _, I_b in loader:
            X_b = X_b.to(DEVICE)
            I_b = I_b.to(DEVICE)
            yp = model(X_b, I_b)  # (B, 100, 1)
            all_preds.append(yp.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0).squeeze(-1)  # (N, 100)
    abs_err = np.abs(y_pred - y_test)  # (N, 100)

    # ── 4. Find the worst sequence ───────────────────────────────────
    flat_idx = np.argmax(abs_err)
    worst_seq, worst_step = np.unravel_index(flat_idx, abs_err.shape)
    max_err = abs_err[worst_seq, worst_step]

    log(f"\n{'=' * 70}")
    log(f"  WORST SEQUENCE IDENTIFIED")
    log(f"{'=' * 70}")
    log(f"  Sequence Index : {worst_seq:,}")
    log(f"  Worst Timestep : {worst_step}")
    log(f"  Max Abs Error  : {max_err * 100:.4f}%")
    log(f"  Temperature    : {temp_labels[worst_seq]}")

    # ── 5. Forensic extraction ───────────────────────────────────────
    yt = y_test[worst_seq]      # (100,)
    yp = y_pred[worst_seq]      # (100,)
    X_raw = X_test[worst_seq]   # (100, 5) — scaled
    I_raw = I_test[worst_seq]   # (100,)   — unscaled Amps

    X_unscaled = unscale_features(X_raw)
    V_proxy = X_unscaled[:, 0]
    I_phys  = X_unscaled[:, 1]
    T_phys  = X_unscaled[:, 2]

    log(f"\n{'=' * 70}")
    log(f"  FORENSIC REPORT — Sequence #{worst_seq:,}")
    log(f"{'=' * 70}")
    log(f"  Temperature label  : {temp_labels[worst_seq]}")
    log(f"")
    log(f"  --- SOC Trajectory ---")
    log(f"  True SOC  [t=0]    : {yt[0]*100:.4f}%")
    log(f"  True SOC  [t=99]   : {yt[-1]*100:.4f}%")
    log(f"  Pred SOC  [t=0]    : {yp[0]*100:.4f}%")
    log(f"  Pred SOC  [t=99]   : {yp[-1]*100:.4f}%")
    log(f"  True delta (0->99) : {(yt[-1]-yt[0])*100:+.4f}%")
    log(f"  Pred delta (0->99) : {(yp[-1]-yp[0])*100:+.4f}%")
    log(f"")
    log(f"  --- Physical Features (unscaled) ---")
    log(f"  V_proxy [t=0]      : {V_proxy[0]:.4f} V")
    log(f"  V_proxy [t=99]     : {V_proxy[-1]:.4f} V")
    log(f"  I (unscaled) [t=0] : {I_raw[0]:.4f} A")
    log(f"  I (unscaled) [t=99]: {I_raw[-1]:.4f} A")
    log(f"  T (unscaled) [t=0] : {T_phys[0]:.1f} degC")
    log(f"  I_mean             : {I_raw.mean():.4f} A")
    log(f"  I_min / I_max      : [{I_raw.min():.4f}, {I_raw.max():.4f}] A")

    # ── 6. Anchor head dissection ────────────────────────────────────
    log(f"\n  --- Anchor Head Dissection ---")
    X_single = torch.from_numpy(X_raw).unsqueeze(0).to(DEVICE)  # (1, 100, 5)
    I_single = torch.from_numpy(I_raw).unsqueeze(0).to(DEVICE)  # (1, 100)

    with torch.no_grad():
        h, _ = model.lstm(X_single)                    # (1, 100, 64)
        soc_anchor = model.anchor_head(h[:, 0, :])     # (1, 1)
        delta_raw  = model.delta_head(h)               # (1, 100, 1)

    anchor_val = soc_anchor.item()
    log(f"  Anchor SOC (raw)   : {anchor_val*100:.4f}%")
    log(f"  True SOC [t=0]     : {yt[0]*100:.4f}%")
    log(f"  Anchor Error       : {abs(anchor_val - yt[0])*100:.4f}%")

    # ── 7. Delta head analysis ───────────────────────────────────────
    delta_raw_np = delta_raw.cpu().numpy().squeeze()  # (100,)
    I_np = I_raw  # (100,)

    # Apply hard constraint manually for analysis
    delta_constrained = np.zeros_like(delta_raw_np)
    for t in range(T):
        if I_np[t] < -CURRENT_THRESHOLD:      # discharge
            delta_constrained[t] = -max(0.0, -delta_raw_np[t])
        elif I_np[t] > CURRENT_THRESHOLD:      # charge
            delta_constrained[t] = max(0.0, delta_raw_np[t])
        # else: rest → 0.0

    cumulative = np.cumsum(delta_constrained)
    soc_reconstructed = anchor_val + cumulative
    soc_clamped = np.clip(soc_reconstructed, 0.0, 1.0)

    log(f"\n  --- Delta Head Analysis ---")
    log(f"  delta_raw  : mean={delta_raw_np.mean():.6f}, "
        f"std={delta_raw_np.std():.6f}")
    log(f"  delta_raw  : min={delta_raw_np.min():.6f}, "
        f"max={delta_raw_np.max():.6f}")
    log(f"  delta_constrained : sum={delta_constrained.sum():.6f}")
    log(f"  cumsum [t=99]     : {cumulative[-1]:.6f}")
    log(f"  SOC pre-clamp [t=0]  : {(anchor_val + cumulative[0])*100:.4f}%")
    log(f"  SOC pre-clamp [t=99] : {(anchor_val + cumulative[-1])*100:.4f}%")
    log(f"  SOC clamped   [t=99] : {soc_clamped[-1]*100:.4f}%")

    # Count how many steps hit the [0,1] clamp
    pre_clamp = anchor_val + cumulative
    n_clamp_lo = (pre_clamp < 0.0).sum()
    n_clamp_hi = (pre_clamp > 1.0).sum()
    log(f"  Steps clamped to 0.0 : {n_clamp_lo} / {T}")
    log(f"  Steps clamped to 1.0 : {n_clamp_hi} / {T}")

    # ── 8. Diagnosis ─────────────────────────────────────────────────
    anchor_err = abs(anchor_val - yt[0])
    drift_err = abs(delta_constrained.sum() - (yt[-1] - yt[0]))
    clamp_affected = n_clamp_lo + n_clamp_hi

    log(f"\n{'=' * 70}")
    log(f"  ROOT CAUSE DIAGNOSIS")
    log(f"{'=' * 70}")

    causes = []
    if anchor_err > 0.3:
        causes.append(f"H1. ANCHOR FAILURE: anchor predicts {anchor_val*100:.1f}% "
                       f"vs true {yt[0]*100:.1f}% (error={anchor_err*100:.1f}%)")
    if drift_err > 0.3:
        causes.append(f"H2. CUMULATIVE DRIFT: total delta={delta_constrained.sum()*100:.2f}% "
                       f"vs needed={(yt[-1]-yt[0])*100:.2f}%")
    if clamp_affected > 10:
        causes.append(f"H3. CLAMP SATURATION: {clamp_affected} of {T} steps "
                       f"hit the [0,1] boundary")

    if not causes:
        causes.append("No single dominant cause > 30% — error is distributed")

    for c in causes:
        log(f"  >> {c}")

    # ── 9. Top-10 worst sequences (distribution analysis) ────────────
    log(f"\n{'=' * 70}")
    log(f"  TOP-10 WORST SEQUENCES")
    log(f"{'=' * 70}")

    per_seq_maxe = np.max(abs_err, axis=1)  # (N,)
    top10_idx = np.argsort(per_seq_maxe)[-10:][::-1]
    log(f"  {'Rank':<5} {'Index':<10} {'MaxE(%)':<10} {'Step':<6} {'Temp':<10} "
        f"{'True[0]':<10} {'Pred[0]':<10} {'True[99]':<10} {'Pred[99]':<10}")
    log(f"  {'-'*80}")

    for rank, idx in enumerate(top10_idx, 1):
        step = np.argmax(abs_err[idx])
        log(f"  {rank:<5} {idx:<10,} {per_seq_maxe[idx]*100:<10.2f} "
            f"{step:<6} {temp_labels[idx]:<10} "
            f"{y_test[idx, 0]*100:<10.2f} {y_pred[idx, 0]*100:<10.2f} "
            f"{y_test[idx, -1]*100:<10.2f} {y_pred[idx, -1]*100:<10.2f}")

    # Temperature distribution of top-50 worst
    top50_idx = np.argsort(per_seq_maxe)[-50:][::-1]
    top50_temps = temp_labels[top50_idx]
    log(f"\n  Temperature distribution of top-50 worst sequences:")
    for t in np.unique(top50_temps):
        cnt = (top50_temps == t).sum()
        log(f"    {t:<10}: {cnt}/50 ({cnt/50*100:.0f}%)")

    # ── 10. Plot forensic figure ─────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.size": 10, "axes.grid": True,
        "grid.alpha": 0.25, "figure.dpi": 300, "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"MaxE Forensic — Sequence #{worst_seq:,}  |  "
                 f"{temp_labels[worst_seq]}  |  "
                 f"MaxE = {max_err*100:.2f}%",
                 fontweight="bold", fontsize=12)

    steps = np.arange(T)

    # Panel 1: True vs Predicted SOC + Anchor point
    axes[0].plot(steps, yt * 100, "b-", lw=1.5, label="True SOC")
    axes[0].plot(steps, yp * 100, "r--", lw=1.5, label="Predicted SOC")
    axes[0].plot(steps, pre_clamp * 100, "g:", lw=1.0, alpha=0.7,
                 label="Pre-clamp SOC")
    axes[0].axhline(y=anchor_val * 100, color="purple", ls=":", lw=0.8,
                    label=f"Anchor = {anchor_val*100:.1f}%")
    axes[0].scatter([0], [anchor_val * 100], c="purple", s=40, zorder=5,
                    marker="D")
    axes[0].set_ylabel("SOC (%)")
    axes[0].legend(loc="best", fontsize=8, framealpha=0.9)
    axes[0].set_ylim(-5, 105)

    # Panel 2: Absolute Error
    abs_err_seq = np.abs(yt - yp) * 100
    axes[1].fill_between(steps, abs_err_seq, color="#FF9800", alpha=0.5)
    axes[1].axhline(y=max_err * 100, color="red", ls=":", lw=1.0,
                    label=f"MaxE = {max_err*100:.2f}%")
    axes[1].scatter([worst_step], [max_err * 100], c="red", s=30, zorder=5)
    axes[1].set_ylabel("Abs. Error (%)")
    axes[1].legend(loc="best", fontsize=8)

    # Panel 3: Current + Constrained Deltas
    ax3a = axes[2]
    ax3b = ax3a.twinx()
    ax3a.plot(steps, I_raw, "b-", lw=0.8, alpha=0.7, label="Current (A)")
    ax3a.set_ylabel("Current (A)", color="blue")
    ax3b.bar(steps, delta_constrained * 100, color="orange", alpha=0.5,
             width=1.0, label="Constrained delta (%)")
    ax3b.set_ylabel("Delta SOC (%)", color="orange")
    ax3a.set_xlabel("Timestep (within 100-step window)")

    lines1, labels1 = ax3a.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3a.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(OUTPUT_FIG, exist_ok=True)
    save_path = os.path.join(OUTPUT_FIG, "v4_lstm_maxe_forensic.png")
    fig.savefig(save_path)
    fig.savefig(save_path.replace(".png", ".pdf"))
    plt.close(fig)
    log(f"\n  Forensic plot saved: {save_path}")

    log(f"\n{'=' * 70}")
    log(f"  RCA COMPLETE")
    log(f"{'=' * 70}")


if __name__ == "__main__":
    main()
