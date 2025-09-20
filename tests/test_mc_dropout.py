import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ueq.methods.mc_dropout import MCDropoutUQ


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 10)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)


def test_mc_dropout_uncertainty_runs():
    X = torch.randn(50, 5)
    y = torch.sum(X, dim=1, keepdim=True)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10)

    model = SmallNet()
    uq = MCDropoutUQ(model, n_forward_passes=5)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    uq.fit(loader, criterion, optimizer, epochs=1)
    mean, std = uq.predict(torch.randn(5, 5))

    assert mean.shape == (5, 1)
    assert std.shape == (5, 1)
    assert (std >= 0).all()
