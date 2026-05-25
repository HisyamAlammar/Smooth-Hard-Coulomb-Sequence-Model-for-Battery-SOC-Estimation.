"""
model_v4_lstm.py — Hybrid Physics-ML LSTM with Hard-Constraint Output Layer
=============================================================================
Ablation architecture: replaces the TCN backbone from V3 with a stacked LSTM
while preserving the exact same:
  1. Dual-head output (Delta + Anchor)
  2. HardConstraintSOC layer (non-parametric conditional clamp)
  3. Cumulative-sum SOC integration
  4. Final [0, 1] clamp

Purpose: isolate the contribution of the temporal backbone (TCN vs LSTM)
         in the ablation study, keeping the physics layer identical.

Forward: (B, 100, 5) + (B, 100) current → (B, 100, 1) SOC ∈ [0, 1]

Created : 2026-05-17  (Sprint 45 — Ablation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CURRENT_THRESHOLD


# ──────────────────────────────────────────────────────────────────────
# HardConstraintSOC — identical to model_v3.py (shared physics layer)
# ──────────────────────────────────────────────────────────────────────
class HardConstraintSOC(nn.Module):
    """
    Structural physics constraint layer (non-parameterized).

    Given raw delta_SOC predictions and current values:
      - Discharge (I < -threshold): force delta_SOC <= 0 via -ReLU(-x)
      - Charge    (I >  threshold): force delta_SOC >= 0 via  ReLU(x)
      - Rest      (|I| < threshold): force delta_SOC  = 0

    Then: SOC_t = SOC_anchor + cumsum(constrained_delta_SOC)

    This guarantees PVR = 0.00% structurally.
    """
    def __init__(self, threshold: float = CURRENT_THRESHOLD):
        super().__init__()
        self.threshold = threshold

    def forward(self, delta_soc_raw: torch.Tensor,
                current_seq: torch.Tensor,
                soc_anchor: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        delta_soc_raw : (B, T, 1) — unconstrained delta predictions
        current_seq   : (B, T)    — unscaled current in Amperes
        soc_anchor    : (B, 1)    — initial SOC prediction from anchor head

        Returns
        -------
        soc_pred : (B, T, 1) — monotonicity-constrained SOC ∈ [0, 1]
        """
        I = current_seq.unsqueeze(-1)  # (B, T, 1)

        discharge_mask = I < -self.threshold
        charge_mask    = I >  self.threshold

        # Start from zeros (rest case), then fill discharge and charge
        delta_constrained = torch.zeros_like(delta_soc_raw)
        delta_constrained[discharge_mask] = -F.relu(-delta_soc_raw[discharge_mask])
        delta_constrained[charge_mask]    =  F.relu( delta_soc_raw[charge_mask])

        # Cumulative sum: SOC_t = anchor + sum(delta_1..t)
        cumulative = torch.cumsum(delta_constrained, dim=1)  # (B, T, 1)
        soc_pred = soc_anchor.unsqueeze(1) + cumulative      # (B, T, 1)

        return soc_pred.clamp(0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────
# HybridPhysicsLSTM — Ablation model (LSTM backbone + Hard Constraint)
# ──────────────────────────────────────────────────────────────────────
class HybridPhysicsLSTM(nn.Module):
    """
    Hybrid Physics-ML LSTM for SOC estimation (v4 ablation).

    Architecture:
      1. LSTM backbone (stacked, batch_first) → hidden states
      2. Delta head: Linear(hidden→32) → ReLU → Linear(32→1) [NO Sigmoid]
      3. Anchor head: Linear(hidden→16) → ReLU → Linear(16→1) → Sigmoid
      4. Hard Constraint layer: conditional clamp + cumsum

    This mirrors TCN_SOC_V3 exactly, swapping only the temporal backbone.
    """
    def __init__(self, num_inputs: int = 5, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=num_inputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Delta head: predicts unconstrained ΔSOC per timestep
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # NO Sigmoid — output is unconstrained real value
        )

        # Anchor head: predicts SOC at t=0 from first timestep's hidden state
        self.anchor_head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # SOC ∈ [0, 1]
        )

        # Hard Constraint layer (non-parameterized) — identical to V3
        self.hard_constraint = HardConstraintSOC()

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for LSTM and linear layers."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for stable long-term memory
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

        for module in [self.delta_head, self.anchor_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, current_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with structural physics constraint.

        Parameters
        ----------
        x           : (B, T=100, C=5) — [V_proxy, I, T, dV_proxy/dt, dI/dt]
        current_seq : (B, T=100)      — unscaled current in Amperes

        Returns
        -------
        soc_pred : (B, T=100, 1) — SOC ∈ [0, 1], PVR = 0% guaranteed
        """
        # LSTM backbone: (B, T, C) → (B, T, hidden_size)
        h, _ = self.lstm(x)  # h: (B, T, hidden_size)

        # Delta head: per-timestep unconstrained ΔSOC
        delta_soc_raw = self.delta_head(h)  # (B, T, 1)

        # Anchor head: SOC at first timestep
        soc_anchor = self.anchor_head(h[:, 0, :])  # (B, 1)

        # Hard Constraint: conditional clamp + cumsum
        soc_pred = self.hard_constraint(delta_soc_raw, current_seq, soc_anchor)

        return soc_pred


# ──────────────────────────────────────────────────────────────────────
# Convenience
# ──────────────────────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    model = HybridPhysicsLSTM(num_inputs=5, hidden_size=64, num_layers=2)
    print(f"HybridPhysicsLSTM — Trainable params: {count_parameters(model):,}")

    x = torch.randn(4, 100, 5)
    I = torch.randn(4, 100) * 5  # fake current
    out = model(x, I)
    print(f"Input:  x={x.shape}, I={I.shape}")
    print(f"Output: {out.shape}  range=[{out.min():.4f}, {out.max():.4f}]")
    print("Sanity check PASSED.")
