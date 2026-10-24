
import torch
import torch.nn as nn

print("PyTorch:", torch.__version__)
print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")

# Example input: 3 colour channels, 300 time samples.
x = torch.randn(1, 3, 300)

# A small 1D CNN for temporal signal processing.
model = nn.Sequential(
    nn.Conv1d(3, 16, kernel_size=7, padding=3),
    nn.ReLU(),
    nn.MaxPool1d(2),
    nn.Conv1d(16, 32, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.AdaptiveAvgPool1d(1),
    nn.Flatten(),
    nn.Linear(32, 1),
)

output = model(x)

print("Input shape:", tuple(x.shape))
print("Output shape:", tuple(output.shape))
print("CNN forward pass successful!")
