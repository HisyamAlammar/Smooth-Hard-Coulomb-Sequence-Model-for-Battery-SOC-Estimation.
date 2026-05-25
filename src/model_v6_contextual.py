"""
model_v6_contextual.py -- Contextual Hard-Coulomb LSTM
======================================================

Sprint 50 architecture: the delta path remains the Hard-Coulomb LSTM sequence
path, while the anchor path receives causal static observability features.
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn

from config import NUM_INPUTS, Q_NOMINAL
from model_v5_coulomb import HardCoulombConstraint

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


class ContextualHardCoulombLSTM(nn.Module):
    """
    Hard-Coulomb LSTM with a contextual anchor head.

    Inputs:
      x_seq       : (B, T=100, 5)
      current_seq : (B, T=100), unscaled Amperes
      anchor_ctx  : (B, 14), scaled contextual anchor features

    Output:
      soc_pred    : (B, T=100, 1)
    """

    def __init__(
        self,
        num_inputs: int = NUM_INPUTS,
        anchor_ctx_dim: int = 14,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        q_nominal: float = Q_NOMINAL,
        safety_factor: float = 1.5,
    ) -> None:
        super().__init__()
        self.num_inputs = num_inputs
        self.anchor_ctx_dim = anchor_ctx_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_inputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.delta_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.anchor_ctx_encoder = nn.Sequential(
            nn.Linear(anchor_ctx_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
        )

        self.anchor_head = nn.Sequential(
            nn.Linear(hidden_size + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.hard_constraint = HardCoulombConstraint(
            q_nominal=q_nominal,
            safety_factor=safety_factor,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                gate_count = param.size(0)
                param.data[gate_count // 4:gate_count // 2].fill_(1.0)

        for module in [self.delta_head, self.anchor_ctx_encoder, self.anchor_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(
        self,
        x_seq: torch.Tensor,
        current_seq: torch.Tensor,
        anchor_ctx: torch.Tensor,
    ) -> torch.Tensor:
        hidden, _ = self.lstm(x_seq)
        delta_soc_raw = self.delta_head(hidden)

        context_embedding = self.anchor_ctx_encoder(anchor_ctx)
        anchor_input = torch.cat([hidden[:, 0, :], context_embedding], dim=-1)
        soc_anchor = self.anchor_head(anchor_input)

        return self.hard_constraint(delta_soc_raw, current_seq, soc_anchor)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


if __name__ == "__main__":
    model = ContextualHardCoulombLSTM()
    x_seq = torch.randn(4, 100, 5)
    current_seq = torch.randn(4, 100)
    anchor_ctx = torch.rand(4, 14)
    output = model(x_seq, current_seq, anchor_ctx)
    print(f"ContextualHardCoulombLSTM parameters: {count_parameters(model):,}")
    print(f"Output shape: {tuple(output.shape)}")
