"""
model_v7_gated_context.py -- Gated Contextual Hard-Coulomb LSTM
================================================================

Sprint 51 architecture: split anchor context into OCV-rest and history streams,
then use a learned gate that is hard-masked by ctx_ocv_rest_valid.

If ctx_ocv_rest_valid == 0, the OCV stream is mathematically disabled:

    g_raw = sigmoid(W[ocv_emb, hist_emb])
    g     = g_raw * ctx_ocv_rest_valid
    ctx   = g * ocv_emb + (1 - g) * hist_emb

The delta path and Hard-Coulomb constraint remain structurally identical.
"""

from __future__ import annotations

import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn

from config import NUM_INPUTS, Q_NOMINAL
from model_v5_coulomb import HardCoulombConstraint

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

OCV_CTX_DIM = 4
HISTORY_CTX_DIM = 10
ANCHOR_CTX_DIM = OCV_CTX_DIM + HISTORY_CTX_DIM
OCV_VALID_INDEX = 0


class GatedContextualHardCoulombLSTM(nn.Module):
    """
    Hard-Coulomb LSTM with a validity-gated contextual anchor.

    Inputs:
      x_seq       : (B, T=100, 5)
      current_seq : (B, T=100), unscaled Amperes
      anchor_ctx  : (B, 14), scaled [OCV-rest features | history features]

    Output:
      soc_pred    : (B, T=100, 1)
    """

    def __init__(
        self,
        num_inputs: int = NUM_INPUTS,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        ctx_emb_dim: int = 32,
        q_nominal: float = Q_NOMINAL,
        safety_factor: float = 1.5,
    ) -> None:
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ctx_emb_dim = ctx_emb_dim

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

        self.ocv_encoder = nn.Sequential(
            nn.Linear(OCV_CTX_DIM, ctx_emb_dim),
            nn.ReLU(),
            nn.LayerNorm(ctx_emb_dim),
        )

        self.history_encoder = nn.Sequential(
            nn.Linear(HISTORY_CTX_DIM, ctx_emb_dim),
            nn.ReLU(),
            nn.LayerNorm(ctx_emb_dim),
        )

        self.context_gate = nn.Linear(ctx_emb_dim * 2, 1)

        self.anchor_head = nn.Sequential(
            nn.Linear(hidden_size + ctx_emb_dim, 32),
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

        for module in [self.delta_head, self.ocv_encoder, self.history_encoder, self.anchor_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.context_gate.weight)
        nn.init.zeros_(self.context_gate.bias)

    def split_context(self, anchor_ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if anchor_ctx.ndim != 2 or anchor_ctx.shape[1] != ANCHOR_CTX_DIM:
            raise ValueError(f"Expected anchor_ctx shape (B, {ANCHOR_CTX_DIM}), got {tuple(anchor_ctx.shape)}")
        ocv_features = anchor_ctx[:, :OCV_CTX_DIM]
        history_features = anchor_ctx[:, OCV_CTX_DIM:]
        ocv_valid = anchor_ctx[:, OCV_VALID_INDEX:OCV_VALID_INDEX + 1].clamp(0.0, 1.0)
        return ocv_features, history_features, ocv_valid

    def encode_context(self, anchor_ctx: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ocv_features, history_features, ocv_valid = self.split_context(anchor_ctx)
        ocv_emb = self.ocv_encoder(ocv_features)
        history_emb = self.history_encoder(history_features)

        raw_gate = torch.sigmoid(self.context_gate(torch.cat([ocv_emb, history_emb], dim=-1)))
        effective_gate = raw_gate * ocv_valid
        final_ctx = effective_gate * ocv_emb + (1.0 - effective_gate) * history_emb

        aux = {
            "ocv_valid": ocv_valid,
            "raw_gate": raw_gate,
            "effective_gate": effective_gate,
        }
        return final_ctx, aux

    def forward(
        self,
        x_seq: torch.Tensor,
        current_seq: torch.Tensor,
        anchor_ctx: torch.Tensor,
        return_aux: bool = False,
    ):
        hidden, _ = self.lstm(x_seq)
        delta_soc_raw = self.delta_head(hidden)

        final_ctx, aux = self.encode_context(anchor_ctx)
        anchor_input = torch.cat([hidden[:, 0, :], final_ctx], dim=-1)
        soc_anchor = self.anchor_head(anchor_input)
        soc_pred = self.hard_constraint(delta_soc_raw, current_seq, soc_anchor)

        if return_aux:
            aux["soc_anchor"] = soc_anchor
            return soc_pred, aux
        return soc_pred


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


if __name__ == "__main__":
    model = GatedContextualHardCoulombLSTM()
    x_seq = torch.randn(4, 100, 5)
    current_seq = torch.randn(4, 100)
    anchor_ctx = torch.rand(4, 14)
    anchor_ctx[1, 0] = 0.0
    output, aux = model(x_seq, current_seq, anchor_ctx, return_aux=True)
    print(f"GatedContextualHardCoulombLSTM parameters: {count_parameters(model):,}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Effective gates: {aux['effective_gate'].flatten().tolist()}")
