# Smooth Hard-Coulomb SOC constraint and LSTM model.
from __future__ import annotations
import sys
import torch
import torch.nn as nn
from config import CURRENT_THRESHOLD, Q_NOMINAL

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass


class SmoothHardCoulombConstraint(nn.Module):
    def __init__(
        self,
        q_nominal: float = Q_NOMINAL,
        dt: float = 1.0,
        safety_factor: float = 1.5,
        threshold: float = CURRENT_THRESHOLD,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.safety_factor = float(safety_factor)
        self.eps = float(eps)
        self.gamma = float(dt) / (float(q_nominal) * 3600.0)

    @property
    def gamma_factor(self) -> float:
        return self.gamma * self.safety_factor

    @gamma_factor.setter
    def gamma_factor(self, value: float) -> None:
        self.safety_factor = float(value) / self.gamma

    def forward(
        self,
        delta_logits: torch.Tensor,
        current_seq: torch.Tensor,
        anchor_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        I = current_seq.unsqueeze(-1)
        limit = I.abs() * self.gamma_factor
        mag_frac = torch.sigmoid(delta_logits)
        zero = torch.zeros_like(delta_logits)
        delta = torch.where(I < -self.threshold, -limit * mag_frac, zero)
        delta = torch.where(I > self.threshold, limit * mag_frac, delta)
        cumulative = torch.cumsum(delta, dim=1)
        lo = (-cumulative.min(dim=1).values).clamp(0.0, 1.0)
        hi = (1.0 - cumulative.max(dim=1).values).clamp(0.0, 1.0)
        width = (hi - lo).clamp_min(self.eps)
        soc_anchor = lo + width * torch.sigmoid(anchor_logit)
        soc_pred = soc_anchor.unsqueeze(1) + cumulative
        return soc_pred, delta


HardCoulombConstraint = SmoothHardCoulombConstraint


class HardCoulombLSTM(nn.Module):
    def __init__(
        self,
        num_inputs: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        q_nominal: float = Q_NOMINAL,
        safety_factor: float = 1.5,
    ) -> None:
        super().__init__()
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
        self.anchor_head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.hard_constraint = SmoothHardCoulombConstraint(
            q_nominal=q_nominal,
            safety_factor=safety_factor,
        )
        self._init_weights()

    def _init_weights(self) -> None:
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
        nn.init.xavier_uniform_(self.delta_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.xavier_uniform_(self.anchor_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.anchor_head[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        current_seq: torch.Tensor,
        return_delta: bool = False,
    ):
        h, _ = self.lstm(x)
        delta_logits = self.delta_head(h)
        anchor_logit = self.anchor_head(h[:, 0, :])
        soc_pred, delta = self.hard_constraint(delta_logits, current_seq, anchor_logit)
        if return_delta:
            return soc_pred, delta
        return soc_pred


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = HardCoulombLSTM(num_inputs=5, hidden_size=64, num_layers=2)
    print(f'HardCoulombLSTM trainable params: {count_parameters(model):,}')
    print(f'Gamma (SOC/A/s): {model.hard_constraint.gamma:.6e}')
    print(f'Gamma factor: {model.hard_constraint.gamma_factor:.6e}')
    x = torch.randn(4, 100, 5)
    current = torch.randn(4, 100) * 10.0
    output, delta = model(x, current, return_delta=True)
    print(f'Output: {output.shape}, range=[{output.min():.4f}, {output.max():.4f}]')
    print(f'Delta: {delta.shape}, range=[{delta.min():.6f}, {delta.max():.6f}]')
    max_delta_20a = 20.0 * model.hard_constraint.gamma_factor
    print(f'Max |delta_SOC|/step at 20A = {max_delta_20a:.6f} ({max_delta_20a * 100:.4f}%)')
