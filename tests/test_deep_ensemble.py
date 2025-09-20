import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ueq import UQ


# Simple network for testing
class TinyNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_core_deep_ensemble():
    # synthetic dataset
    X = torch.randn(50, 3)
    y = torch.sum(X, dim=1, keepdim=True) + 0.05 * torch.randn(50, 1)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    # construct UQ with Deep Ensemble (3 models)
    uq = UQ(lambda: TinyNet(input_dim=3), method="deep_ensemble", n_models=3)

    criterion = nn.MSELoss()
    optimizer_fn = lambda params: optim.Adam(params, lr=0.01)

    # Train ensemble
    uq.fit(loader, criterion, optimizer_fn, epochs=2)

    # Predict on new inputs
    X_test = torch.randn(5, 3)
    mean, intervals = uq.predict(X_test)

    # checks
    assert mean.shape == (5, 1)
    assert len(intervals) == 5
    assert all(len(iv) == 2 for iv in intervals)
    assert all(iv[1] >= iv[0] for iv in intervals)  # upper ≥ lower
