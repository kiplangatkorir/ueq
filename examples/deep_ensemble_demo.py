import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from ueq import UQ


# -------------------------------
# Define a simple neural network
# -------------------------------
class SimpleNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# -------------------------------
# Demo script
# -------------------------------
def main():
    # synthetic regression data
    X = torch.randn(200, 5)
    y = torch.sum(X, dim=1, keepdim=True) + 0.1 * torch.randn(200, 1)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # define base model
    base_model = SimpleNet(input_dim=5)

    # Wrap in UQ with Deep Ensembles
    # Wrap in UQ with Deep Ensembles
    uq = UQ(lambda: SimpleNet(input_dim=5), method="deep_ensemble", n_models=3)

    # Training config
    criterion = nn.MSELoss()
    optimizer_fn = lambda params: optim.Adam(params, lr=0.01)

    # Fit ensemble
    uq.fit(loader, criterion, optimizer_fn, epochs=10)

    # Test predictions
    X_test = torch.randn(5, 5)
    mean_pred, intervals = uq.predict(X_test)

    print("Predictions:", mean_pred.flatten().tolist())
    print("Uncertainty intervals:", intervals)


if __name__ == "__main__":
    main()
