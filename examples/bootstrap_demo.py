from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from ueq import UQ

# Synthetic dataset
X, y = make_regression(n_samples=200, n_features=10, noise=5, random_state=42)

# Base model
model = RandomForestRegressor()

# Wrap with Uncertainty Everywhere
uq = UQ(model, method="bootstrap", n_samples=50, random_state=42)

# Fit
uq.fit(X, y)

# Predict with uncertainty
pred, interval = uq.predict(X[:5], return_interval=True)

print("Predictions:", pred)
print("Intervals:", interval)
