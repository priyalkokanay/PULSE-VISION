
"""
PulseVision: Lightweight temporal CNN for rPPG.

Input:
    RGB sequences shaped (batch, 3, time)

Output:
    Predicted pulse waveform shaped (batch, time)

This model must be trained on synchronized RGB sequences and
reference pulse waveforms before its estimates are meaningful.
"""

import torch
from torch import nn


class ResidualTemporalBlock(nn.Module):
    """Lightweight residual temporal convolution block."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(4, channels)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.activation(self.norm1(x))
        x = self.norm2(self.conv2(x))

        return self.activation(x + residual)


class PulseVisionRPPGNet(nn.Module):
    """
    Compact 1D CNN that predicts a pulse waveform from RGB traces.

    Expected input:
        (B, 3, T), where channels are red, green, blue.

    Output:
        (B, T), one pulse waveform per sequence.
    """

    def __init__(self, channels: int = 32):
        super().__init__()

        if channels % 4 != 0:
            raise ValueError("channels must be divisible by 4")

        self.stem = nn.Sequential(
            nn.Conv1d(
                3,
                channels,
                kernel_size=7,
                padding=3,
            ),
            nn.GroupNorm(4, channels),
            nn.GELU(),
        )

        self.temporal_blocks = nn.Sequential(
            ResidualTemporalBlock(channels, dilation=1),
            ResidualTemporalBlock(channels, dilation=2),
            ResidualTemporalBlock(channels, dilation=4),
            ResidualTemporalBlock(channels, dilation=8),
        )

        self.head = nn.Sequential(
            nn.Conv1d(channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=1),
        )

    def forward(self, rgb):
        if rgb.ndim != 3:
            raise ValueError(
                f"Expected input shape (B, 3, T), got {tuple(rgb.shape)}"
            )

        if rgb.shape[1] != 3:
            raise ValueError(
                f"Expected 3 RGB channels, got {rgb.shape[1]}"
            )

        features = self.stem(rgb)
        features = self.temporal_blocks(features)
        pulse = self.head(features)

        # Remove the single output-channel dimension.
        return pulse.squeeze(1)


def waveform_loss(predicted, target):
    """
    Shape-invariant waveform loss.

    Normalizes each waveform before comparing it, so the model
    learns temporal shape rather than absolute signal amplitude.
    """
    if predicted.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted.shape}, "
            f"target {target.shape}"
        )

    predicted = predicted - predicted.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    predicted = predicted / (
        predicted.std(dim=-1, keepdim=True).clamp_min(1e-6)
    )
    target = target / (
        target.std(dim=-1, keepdim=True).clamp_min(1e-6)
    )

    return nn.functional.mse_loss(predicted, target)


if __name__ == "__main__":
    # Smoke test only: random input is not training data.
    model = PulseVisionRPPGNet()
    sample = torch.randn(2, 3, 900)

    with torch.no_grad():
        output = model(sample)

    print("Model:", model.__class__.__name__)
    print("Input shape:", tuple(sample.shape))
    print("Output shape:", tuple(output.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Device: CPU smoke test passed")
