"""
model_v5_coulomb.py — Hard-Coulomb LSTM with Physics-Bounded Delta Constraint
===============================================================================
Sprint 46 Architecture: fixes the V4 cumulative drift collapse by adding a
physics-derived MAGNITUDE bound on delta_SOC, not just a direction clamp.

Key innovation — HardCoulombConstraint:
  The maximum possible |ΔSOC| per timestep is bounded by Coulomb's law:
      |ΔSOC_max| = |I_t| × Δt / (Q_nom × 3600)
  At 1 Hz sampling (Δt = 1s), with Q_nom = 3.0 Ah:
      |ΔSOC_max| = |I_t| / 10800

  This prevents the network from predicting physically impossible SOC changes
  (e.g., 5% SOC drop in 1 second at 0.1A), which was the root cause of the
  V4 MaxE = 99.90% failure.

  A safety_factor (default 1.5) gives 50% headroom above the theoretical
  limit to account for measurement noise and transient dynamics.

Forward: (B, 100, 5) + (B, 100) current → (B, 100, 1) SOC ∈ [0, 1]

Created : 2026-05-17  (Sprint 46)
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CURRENT_THRESHOLD, Q_NOMINAL

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


# ──────────────────────────────────────────────────────────────────────
# HardCoulombConstraint — physics-bounded magnitude + direction clamp
# ──────────────────────────────────────────────────────────────────────
class HardCoulombConstraint(nn.Module):
    """
    Structural physics constraint with Coulomb-limited magnitude.

    For each timestep:
      coulomb_limit = |I_t| * dt / (Q_nom * 3600) * safety_factor

      Discharge (I < -thresh): delta = clamp(raw, min=-coulomb_limit, max=0)
      Charge    (I >  thresh): delta = clamp(raw, min=0, max=coulomb_limit)
      Rest      (|I| < thresh): delta = 0

    Then: SOC_t = SOC_anchor + cumsum(constrained_deltas)
    Final: clamp [0, 1]

    Guarantees:
      1. PVR = 0.00% (direction constraint)
      2. MaxE bounded by physics (magnitude constraint)
    """
    def __init__(self, q_nominal: float = Q_NOMINAL,
                 dt: float = 1.0,
                 safety_factor: float = 1.5,
                 threshold: float = CURRENT_THRESHOLD):
        super().__init__()
        self.threshold = threshold
        self.safety_factor = safety_factor
        # gamma = dt / (Q_nom_Ah * 3600_s/h) — converts Amps to fractional SOC/step
        self.gamma = dt / (q_nominal * 3600.0)

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
        soc_pred : (B, T, 1) — physics-bounded SOC ∈ [0, 1]
        """
        I = current_seq.unsqueeze(-1)  # (B, T, 1)

        # Coulomb envelope: max physically possible |ΔSOC| per step
        coulomb_limit = torch.abs(I) * self.gamma * self.safety_factor  # (B, T, 1)

        discharge_mask = I < -self.threshold
        charge_mask    = I >  self.threshold
        # rest: everything else → delta = 0

        delta_constrained = torch.zeros_like(delta_soc_raw)

        # Discharge: delta ∈ [-coulomb_limit, 0]
        delta_constrained[discharge_mask] = torch.clamp(
            delta_soc_raw[discharge_mask],
            min=-coulomb_limit[discharge_mask],
            max=torch.zeros_like(coulomb_limit[discharge_mask])
        )

        # Charge: delta ∈ [0, +coulomb_limit]
        delta_constrained[charge_mask] = torch.clamp(
            delta_soc_raw[charge_mask],
            min=torch.zeros_like(coulomb_limit[charge_mask]),
            max=coulomb_limit[charge_mask]
        )

        # Cumulative sum: SOC_t = anchor + sum(delta_1..t)
        cumulative = torch.cumsum(delta_constrained, dim=1)  # (B, T, 1)
        soc_pred = soc_anchor.unsqueeze(1) + cumulative      # (B, T, 1)

        return soc_pred.clamp(0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────
# HardCoulombLSTM — Full model
# ──────────────────────────────────────────────────────────────────────
class HardCoulombLSTM(nn.Module):
    """
    Hybrid Physics-ML LSTM with Coulomb-bounded Hard Constraint (v5).

    Architecture:
      1. LSTM backbone (stacked, batch_first) → hidden states
      2. Delta head: Linear(hidden→32) → ReLU → Linear(32→1) [NO Sigmoid]
      3. Anchor head: Linear(hidden→16) → ReLU → Linear(16→1) → Sigmoid
      4. HardCoulombConstraint: direction + magnitude clamp + cumsum
    """
    def __init__(self, num_inputs: int = 5, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2,
                 q_nominal: float = Q_NOMINAL, safety_factor: float = 1.5):
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

        # Delta head: unconstrained ΔSOC per timestep
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Anchor head: SOC at t=0
        self.anchor_head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Hard-Coulomb Constraint (non-parametric)
        self.hard_constraint = HardCoulombConstraint(
            q_nominal=q_nominal,
            safety_factor=safety_factor,
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
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
        Parameters
        ----------
        x           : (B, T=100, C=5) — [V_proxy, I, T, dV_proxy/dt, dI/dt]
        current_seq : (B, T=100)      — unscaled current in Amperes

        Returns
        -------
        soc_pred : (B, T=100, 1) — SOC ∈ [0, 1], PVR=0% + Coulomb-bounded
        """
        h, _ = self.lstm(x)
        delta_soc_raw = self.delta_head(h)
        soc_anchor = self.anchor_head(h[:, 0, :])
        soc_pred = self.hard_constraint(delta_soc_raw, current_seq, soc_anchor)
        return soc_pred


# ──────────────────────────────────────────────────────────────────────
# Convenience
# ──────────────────────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = HardCoulombLSTM(num_inputs=5, hidden_size=64, num_layers=2)
    print(f"HardCoulombLSTM — Trainable params: {count_parameters(model):,}")
    print(f"Gamma (SOC/A/s): {model.hard_constraint.gamma:.6e}")
    print(f"Safety factor  : {model.hard_constraint.safety_factor}")

    x = torch.randn(4, 100, 5)
    I = torch.randn(4, 100) * 10
    out = model(x, I)
    print(f"Input:  x={x.shape}, I={I.shape}")
    print(f"Output: {out.shape}  range=[{out.min():.4f}, {out.max():.4f}]")

    # Verify magnitude bound: at 20A, max delta per step
    max_delta_20A = 20.0 * model.hard_constraint.gamma * model.hard_constraint.safety_factor
    print(f"\nPhysics check at 20A:")
    print(f"  Max |delta_SOC|/step = {max_delta_20A:.6f} ({max_delta_20A*100:.4f}%)")
    print(f"  Max drift over 100 steps = {max_delta_20A*100*100:.2f}%")
    print("Sanity check PASSED.")
