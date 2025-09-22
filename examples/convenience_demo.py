from sklearn.datasets import make_regression
from ueq.utils.api import evaluate
from ueq.utils.plotting import plot_intervals

def main():
    # Generate synthetic data
    X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)
    X_train, X_test = X[:70], X[70:]
    y_train, y_test = y[:70], y[70:]

    # Evaluate Bayesian Linear Regression
    results = evaluate(
        method="bayesian_linear",
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        alpha=2.0, beta=25.0
    )

    # Print metrics and plot results
    print("\n📊 Uncertainty Metrics:")
    for metric, value in results["metrics"].items():
        print(f"{metric}: {value:.3f}")
        
    plot_intervals(X_test, y_test, results["mean"], results["intervals"],
                  title="Bayesian Linear Regression with Uncertainty")

if __name__ == "__main__":
    main()