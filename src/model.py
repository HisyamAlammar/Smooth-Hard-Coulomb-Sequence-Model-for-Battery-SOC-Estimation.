"""
model.py — PI-TCN SOC Estimator Architecture & Physics-Informed Loss
=====================================================================
Physics-Informed Temporal Convolutional Network for State of Charge
estimation of Li-ion batteries (LG HG2 18650).

Architecture follows Section 4 of the Project Brief:
  Input(B,100,5) → 4×TemporalBlock(dil=1,2,4,8) → LastStep → FC → Sigmoid

Created : 2026-04-08
Framework: PyTorch (FINAL — no TF/Keras)
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


# ──────────────────────────────────────────────────────────────────────
# Helper: Chomp1d — removes trailing padding to preserve causality
# ──────────────────────────────────────────────────────────────────────
class Chomp1d(nn.Module):
    """Remove the extra right-side padding added by Conv1d to enforce causality."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


# ──────────────────────────────────────────────────────────────────────
# TemporalBlock — single TCN residual block
# ──────────────────────────────────────────────────────────────────────
class TemporalBlock(nn.Module):
    """
    One residual block of the TCN:
      2× (Conv1d → WeightNorm → ReLU → Dropout)
      + Residual connection (1×1 conv if channel mismatch)

    Parameters
    ----------
    n_inputs  : int — input channel count
    n_outputs : int — output channel count (= num_filters)
    kernel_size : int — convolution kernel size
    stride    : int — convolution stride (always 1 for TCN)
    dilation  : int — dilation factor for this block
    dropout   : float — dropout probability
    """

    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                 stride: int, dilation: int, dropout: float = 0.2):
        super().__init__()

        # Causal padding: pad left so output length == input length
        padding = (kernel_size - 1) * dilation

        # --- First conv layer ---
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.norm1 = nn.LayerNorm(n_outputs)   # domain-shift robustness
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # --- Second conv layer ---
        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.norm2 = nn.LayerNorm(n_outputs)   # domain-shift robustness
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Full sequential for the main branch
        # Note: LayerNorm expects (B, T, C), so we transpose inside forward
        self.net = None  # built manually in forward() now

        # --- Residual (skip) connection ---
        # 1×1 conv if input channels ≠ output channels
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu_out = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        """Kaiming (He) initialisation for convolutional layers."""
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: main branch (with LayerNorm) + residual."""
        # First conv branch: Conv → Chomp → LayerNorm → ReLU → Dropout
        out = self.conv1(x)
        out = self.chomp1(out)
        out = out.transpose(1, 2)          # (B, C, T) → (B, T, C)
        out = self.norm1(out)
        out = out.transpose(1, 2)          # (B, T, C) → (B, C, T)
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second conv branch
        out = self.conv2(out)
        out = self.chomp2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


# ──────────────────────────────────────────────────────────────────────
# TCN_SOC_Estimator — full model
# ──────────────────────────────────────────────────────────────────────
class TCN_SOC_Estimator(nn.Module):
    """
    Physics-Informed Temporal Convolutional Network for SOC estimation.

    Architecture (Seq2Seq — Sprint 5 pivot):
      1. Transpose input (B, T, C) → (B, C, T) for Conv1d
      2. Stack 4 TemporalBlocks with dilation_rates = [1, 2, 4, 8]
      3. Transpose back (B, C, T) → (B, T, C)
      4. FC applied per-timestep: Linear(n_filters→32) → ReLU → Linear(32→1) → Sigmoid

    Parameters
    ----------
    num_inputs   : int   — number of input features (5)
    num_filters  : int   — channels per TemporalBlock (64)
    kernel_size  : int   — convolution kernel size (7)
    dropout      : float — dropout probability (0.2)
    dilation_rates : list[int] — dilation for each block [1,2,4,8]
    """

    def __init__(self, num_inputs: int = 5, num_filters: int = 64,
                 kernel_size: int = 7, dropout: float = 0.2,
                 dilation_rates: list = None):
        super().__init__()

        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 8]

        # --- TCN backbone ---
        layers = []
        for i, dilation in enumerate(dilation_rates):
            in_channels = num_inputs if i == 0 else num_filters
            layers.append(TemporalBlock(
                n_inputs=in_channels,
                n_outputs=num_filters,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                dropout=dropout,
            ))
        self.tcn = nn.Sequential(*layers)

        # --- Fully-connected head (applied per-timestep) ---
        self.fc = nn.Sequential(
            nn.Linear(num_filters, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),          # output ∈ [0, 1]
        )

        # Compute & store receptive field for reference
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(dilation_rates)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (Sequence-to-Sequence).

        Parameters
        ----------
        x : Tensor of shape (Batch, T=100, C=5)
            [Voltage, Current, Temperature, dV/dt, dI/dt]

        Returns
        -------
        Tensor of shape (Batch, T=100, 1)
            Estimated SOC ∈ [0, 1] at every timestep
        """
        # (B, T=100, C=5) → (B, C=5, T=100) for Conv1d
        x = x.transpose(1, 2)

        # Pass through TCN blocks
        x = self.tcn(x)                  # (B, n_filters, T)

        # Transpose back: (B, n_filters, T) → (B, T, n_filters)
        x = x.transpose(1, 2)

        # Per-timestep regression head: (B, T, n_filters) → (B, T, 1)
        x = self.fc(x)
        return x


# ──────────────────────────────────────────────────────────────────────
# PhysicsInformedLoss — MSE + monotonicity penalty during discharge
# ──────────────────────────────────────────────────────────────────────
class PhysicsInformedLoss(nn.Module):
    """
    Combined loss: L_total = MSE(ŷ, y) + λ · Physics_Penalty

    Physics constraint (Seq2Seq intra-window — Sprint 5):
      During discharge (I < -0.05 A), SOC must NOT increase.
      ΔSOC is computed between temporally consecutive timesteps
      *within* each window, not across batch samples.

      delta_soc = y_pred[:, 1:, 0] - y_pred[:, :-1, 0]   → (B, T-1)
      Penalty = mean(ReLU(delta_soc[discharge_mask]))

    Lambda Schedule (Curriculum):
      lambda_phys is updated externally each epoch via:
        criterion.lambda_phys = min(MAX_LAMBDA, MAX_LAMBDA * (epoch / 30.0))
      MAX_LAMBDA = 5.0 recommended (see train.py / sprint44_ablation.py)

    Discharge threshold: -0.05 A (filters sensor noise, genuine discharge only)

    Parameters
    ----------
    lambda_phys : float — weight of the physics penalty (updated dynamically)
    """
    DISCHARGE_THRESHOLD = -0.05   # Amperes — genuine discharge only

    def __init__(self, lambda_phys: float = 0.1):
        super().__init__()
        self.lambda_phys = lambda_phys
        self.mse = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                current_values: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-informed loss (Seq2Seq intra-window).

        Parameters
        ----------
        y_pred : Tensor (B, T, 1) — predicted SOC at every timestep
        y_true : Tensor (B, T, 1) — ground-truth SOC trajectory
        current_values : Tensor (B, T) — unscaled current (Amperes)

        Returns
        -------
        loss : scalar Tensor
        """
        # --- Defensive shape assertions (catch silent broadcasting bugs) ---
        assert y_pred.dim() == 3 and y_pred.shape[-1] == 1, (
            f"y_pred must be (B, T, 1), got {tuple(y_pred.shape)}")
        assert current_values.dim() == 2, (
            f"current_values must be (B, T), got {tuple(current_values.shape)}")

        # --- Data-driven loss (full sequence) ---
        loss_mse = self.mse(y_pred, y_true)

        # --- Intra-window physics penalty ---
        # ΔSOC between consecutive timesteps within each window
        soc_seq = y_pred[:, :, 0]                              # (B, T)   — squeeze last dim
        delta_soc = soc_seq[:, 1:] - soc_seq[:, :-1]          # (B, T-1)

        # Discharge mask: current < DISCHARGE_THRESHOLD at destination timestep
        # current_values is already (B, T) — no squeeze needed
        discharge_mask = current_values[:, 1:] < self.DISCHARGE_THRESHOLD  # (B, T-1)

        if discharge_mask.any():
            # Penalise SOC *increases* during genuine discharge
            violations = torch.relu(delta_soc[discharge_mask])   # 1D flat of True entries
            loss_phys = violations.mean()
        else:
            loss_phys = torch.tensor(0.0, device=y_pred.device, requires_grad=False)

        return loss_mse + self.lambda_phys * loss_phys


# ──────────────────────────────────────────────────────────────────────
# Convenience: count parameters
# ──────────────────────────────────────────────────────────────────────
def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
