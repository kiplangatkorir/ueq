import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ueq import UQ

# Dummy dataset
X = torch.randn(200, 10)
y = torch.sum(X, dim=1, keepdim=True) + 0.1 * torch.randn(200, 1)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Simple model with dropout
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.drop = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

model = Net()

# Wrap with UQ
uq = UQ(model, method="mc_dropout", n_forward_passes=100)

# Train
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
uq.fit(loader, criterion, optimizer, epochs=20)

# Predict with uncertainty
X_test = torch.randn(5, 10)
mean, std = uq.predict(X_test)

print("Predictive mean:", mean)
print("Uncertainty (std):", std)
