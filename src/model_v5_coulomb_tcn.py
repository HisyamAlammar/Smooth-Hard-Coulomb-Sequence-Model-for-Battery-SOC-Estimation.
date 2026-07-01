"""
model_v5_coulomb_tcn.py -- Hard-Coulomb TCN for SOC Estimation
================================================================

Sprint 46 backbone-agnostic test: duplicate the V3 Hybrid Physics-ML TCN
backbone and replace the original sign-only Hard Constraint with the same
Hard-Coulomb envelope used by the V5 LSTM.

Forward: (B, 100, 5) + (B, 100) current -> (B, 100, 1) SOC in [0, 1]
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from config import Q_NOMINAL
from model_v5_coulomb import SmoothHardCoulombConstraint


class Chomp1d(nn.Module):
    """Remove extra right-side padding added by Conv1d for causal TCN blocks."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """One residual V3 TCN block: Conv1d -> LayerNorm -> ReLU -> Dropout x2."""
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
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout1(self.relu1(
            self.norm1(self.chomp1(self.conv1(x)).transpose(1, 2)).transpose(1, 2)))
        out = self.dropout2(self.relu2(
            self.norm2(self.chomp2(self.conv2(out)).transpose(1, 2)).transpose(1, 2)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class HardCoulombTCN(nn.Module):
    """V3 TCN backbone with V5 Hard-Coulomb output constraint."""
    def __init__(self, num_inputs: int = 5, num_filters: int = 64,
                 kernel_size: int = 7, dropout: float = 0.2,
                 dilation_rates: list = None,
                 q_nominal: float = Q_NOMINAL,
                 safety_factor: float = 1.5):
        super().__init__()

        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 8]

        layers = []
        for i, dilation in enumerate(dilation_rates):
            in_ch = num_inputs if i == 0 else num_filters
            layers.append(TemporalBlock(
                n_inputs=in_ch, n_outputs=num_filters,
                kernel_size=kernel_size, stride=1,
                dilation=dilation, dropout=dropout))
        self.tcn = nn.Sequential(*layers)

        self.delta_head = nn.Sequential(
            nn.Linear(num_filters, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.anchor_head = nn.Sequential(
            nn.Linear(num_filters, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.hard_constraint = SmoothHardCoulombConstraint(
            q_nominal=q_nominal,
            safety_factor=safety_factor,
        )
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(dilation_rates)
        self._init_heads()

    def _init_heads(self) -> None:
        for module in [self.delta_head, self.anchor_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.delta_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.xavier_uniform_(self.anchor_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.anchor_head[-1].bias)

    def forward(self, x: torch.Tensor, current_seq: torch.Tensor) -> torch.Tensor:
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        delta_logits = self.delta_head(h)
        anchor_logit = self.anchor_head(h[:, 0, :])
        soc_pred, _delta = self.hard_constraint(delta_logits, current_seq, anchor_logit)
        return soc_pred


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = HardCoulombTCN(num_inputs=5, num_filters=64)
    print(f"HardCoulombTCN trainable params: {count_parameters(model):,}")
    print(f"Receptive field: {model.receptive_field} steps")
    print(f"Gamma: {model.hard_constraint.gamma:.6e} SOC/A/s")
    print(f"Gamma factor: {model.hard_constraint.gamma_factor:.6e}")
    x = torch.randn(4, 100, 5)
    current = torch.randn(4, 100) * 10.0
    y = model(x, current)
    print(f"Output: {y.shape}, range=[{y.min():.4f}, {y.max():.4f}]")
