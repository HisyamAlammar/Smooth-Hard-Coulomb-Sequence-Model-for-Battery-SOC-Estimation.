"""
model_v3.py — Hybrid Physics-ML TCN with Hard-Constraint Output Layer
=======================================================================
Architecture changes from v2:
  1. TCN backbone predicts raw delta_SOC (no Sigmoid)
  2. Conditional Hard Constraint: clamps delta_SOC sign by current direction
  3. Cumulative sum from auxiliary anchor head → SOC trajectory
  4. PVR = 0.00% is structurally guaranteed (not learned)

Forward: (B,100,5) + (B,100) current → (B,100,1) SOC ∈ [0,1]

Created : 2026-05-16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from config import CURRENT_THRESHOLD


# ──────────────────────────────────────────────────────────────────────
# Chomp1d — causal padding removal (unchanged from v2)
# ──────────────────────────────────────────────────────────────────────
class Chomp1d(nn.Module):
    """Remove the extra right-side padding added by Conv1d to enforce causality."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


# ──────────────────────────────────────────────────────────────────────
# TemporalBlock — single TCN residual block (unchanged from v2)
# ──────────────────────────────────────────────────────────────────────
class TemporalBlock(nn.Module):
    """
    One residual block of the TCN:
      2× (Conv1d → WeightNorm → LayerNorm → ReLU → Dropout)
      + Residual connection (1×1 conv if channel mismatch)
    """
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                 stride: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.norm1 = nn.LayerNorm(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.norm2 = nn.LayerNorm(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None)
        self.relu_out = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout1(self.relu1(
            self.norm1(self.chomp1(self.conv1(x)).transpose(1, 2)).transpose(1, 2)))
        out = self.dropout2(self.relu2(
            self.norm2(self.chomp2(self.conv2(out)).transpose(1, 2)).transpose(1, 2)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


# ──────────────────────────────────────────────────────────────────────
# HardConstraintSOC — Conditional clamp + cumulative sum
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
        # rest_mask: everything else → delta = 0 (initialized below)

        # Start from zeros (rest case), then fill discharge and charge
        delta_constrained = torch.zeros_like(delta_soc_raw)
        delta_constrained[discharge_mask] = -F.relu(-delta_soc_raw[discharge_mask])
        delta_constrained[charge_mask]    =  F.relu( delta_soc_raw[charge_mask])

        # Cumulative sum: SOC_t = anchor + sum(delta_1..t)
        cumulative = torch.cumsum(delta_constrained, dim=1)  # (B, T, 1)
        soc_pred = soc_anchor.unsqueeze(1) + cumulative      # (B, T, 1)

        return soc_pred.clamp(0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────
# TCN_SOC_V3 — Full Hybrid Physics-ML Model
# ──────────────────────────────────────────────────────────────────────
class TCN_SOC_V3(nn.Module):
    """
    Hybrid Physics-ML TCN for SOC estimation (v3).

    Architecture:
      1. TCN backbone (4× TemporalBlock) → hidden states
      2. Delta head: Linear(64→32) → ReLU → Linear(32→1) [NO Sigmoid]
      3. Anchor head: Linear(64→16) → ReLU → Linear(16→1) → Sigmoid
      4. Hard Constraint layer: conditional clamp + cumsum
    """
    def __init__(self, num_inputs: int = 5, num_filters: int = 64,
                 kernel_size: int = 7, dropout: float = 0.2,
                 dilation_rates: list = None):
        super().__init__()

        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 8]

        # TCN backbone
        layers = []
        for i, dilation in enumerate(dilation_rates):
            in_ch = num_inputs if i == 0 else num_filters
            layers.append(TemporalBlock(
                n_inputs=in_ch, n_outputs=num_filters,
                kernel_size=kernel_size, stride=1,
                dilation=dilation, dropout=dropout))
        self.tcn = nn.Sequential(*layers)

        # Delta head: predicts unconstrained ΔSOC per timestep
        self.delta_head = nn.Sequential(
            nn.Linear(num_filters, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # NO Sigmoid — output is unconstrained real value
        )

        # Anchor head: predicts SOC at t=0 from first timestep's hidden state
        self.anchor_head = nn.Sequential(
            nn.Linear(num_filters, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # SOC ∈ [0, 1]
        )

        # Hard Constraint layer (non-parameterized)
        self.hard_constraint = HardConstraintSOC()

        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(dilation_rates)

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
        # TCN backbone: (B, T, C) → (B, C, T) → TCN → (B, T, C)
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)  # (B, T, num_filters)

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
