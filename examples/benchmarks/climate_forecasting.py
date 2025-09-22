# examples/benchmarks/climate_forecasting.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset  # Add this import

from ueq import UQ
from ueq.utils.metrics import coverage, sharpness, expected_calibration_error

# ------------------------------
# 1. Load dataset
# ------------------------------
print("🌡️ Loading Housing dataset...")
data = fetch_california_housing(as_frame=True)
X = data.data.values
y = data.target.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors and create DataLoader
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# ------------------------------
# 2. Define PyTorch model with dropout
# ------------------------------
class MLPDropout(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------
# 3. Wrap with MC Dropout UQ
# ------------------------------
uq = UQ(
    model=lambda: MLPDropout(X_train.shape[1], dropout_rate=0.2),
    method="mc_dropout",
    n_forward_passes=30,
    lr=1e-3,
    epochs=50,
    batch_size=64,
    device="cpu"
)

uq.fit(X_train, y_train)
mean_pred, intervals = uq.predict(X_test)

# ------------------------------
# 4. Metrics
# ------------------------------
print("🌍 Climate Forecasting Benchmark")
print("Coverage:", coverage(y_test, intervals))
print("Sharpness:", sharpness(intervals))
print("ECE:", expected_calibration_error(y_test, intervals))

# ------------------------------
# 5. Visualization
# ------------------------------
plt.figure(figsize=(10, 5))
plt.plot(y_test[:200], label="True", alpha=0.7)
plt.plot(mean_pred[:200], label="Predicted Mean", alpha=0.7)
plt.fill_between(
    np.arange(200),
    intervals[:200, 0],
    intervals[:200, 1],
    color="orange",
    alpha=0.3,
    label="Uncertainty Interval"
)
plt.legend()
plt.title("Climate Forecasting with MC Dropout UQ")
plt.show()
