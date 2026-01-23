"""
Example demonstrating new UQ evaluation and visualization features in UEQ v1.0.2

This example shows:
1. Standardized UQ evaluation metrics (Issue #9)
2. Reliability and calibration diagnostics (Issue #10)
3. Enhanced prediction interval visualizations (Issue #17)
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from ueq import UQ
from ueq.utils import (
    evaluate_uncertainty,
    check_calibration,
    plot_intervals,
    plot_reliability_diagram,
    plot_coverage_vs_confidence,
)


def generate_regression_data(n_samples=200, noise_level=2.0, seed=42):
    """Generate synthetic regression data with known noise."""
    np.random.seed(seed)
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    # True relationship: y = 2x + 1
    y_true = 2 * X.flatten() + 1
    # Add heteroscedastic noise (increases with X)
    noise = noise_level * (1 + 0.3 * X.flatten()) * np.random.randn(n_samples)
    y = y_true + noise
    return X, y


def main():
    print("=" * 70)
    print("UEQ v1.0.2 - Enhanced Evaluation & Visualization Demo")
    print("=" * 70)
    
    # 1. Generate data and train model
    print("\n1. Generating synthetic regression data...")
    X, y = generate_regression_data(n_samples=200, noise_level=2.0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # 2. Train UQ model with bootstrap
    print("\n2. Training UQ model with bootstrap method...")
    model = LinearRegression()
    uq = UQ(model, method='bootstrap', n_models=50)
    uq.fit(X_train, y_train)
    print("   Model trained successfully")
    
    # 3. Get predictions
    print("\n3. Generating predictions with uncertainty intervals...")
    y_pred, intervals = uq.predict(X_test, return_interval=True)
    print(f"   Generated {len(intervals)} prediction intervals")
    
    # 4. Evaluate using new standardized metrics (Issue #9)
    print("\n4. Computing standardized UQ evaluation metrics...")
    print("   " + "-" * 60)
    
    # Evaluate all metrics at once
    metrics = evaluate_uncertainty(
        y_true=y_test,
        y_pred=y_pred,
        intervals=intervals,
        alpha=0.05,  # For 95% intervals
        n_bins=10
    )
    
    print(f"   Coverage:              {metrics['coverage']:.4f}")
    print(f"   Sharpness (avg width): {metrics['sharpness']:.4f}")
    print(f"   Interval Width:        {metrics['interval_width']:.4f}")
    print(f"   ECE (calibration):     {metrics['ece']:.4f}")
    print(f"   MCE (calibration):     {metrics['mce']:.4f}")
    print(f"   Interval Score:        {metrics['interval_score']:.4f}")
    print(f"   Miscoverage Rate:      {metrics['miscoverage']:.4f}")
    
    # Evaluate selected metrics only
    selected_metrics = evaluate_uncertainty(
        y_true=y_test,
        y_pred=y_pred,
        intervals=intervals,
        metrics=['coverage', 'sharpness', 'interval_score']
    )
    
    print("\n   Selected metrics only:")
    for name, value in selected_metrics.items():
        print(f"   {name}: {value:.4f}")
    
    # 5. Run calibration diagnostics (Issue #10)
    print("\n5. Running calibration diagnostics...")
    print("   " + "-" * 60)
    
    diagnostics = check_calibration(
        y_true=y_test,
        intervals=intervals,
        confidence=0.95,
        tolerance=0.05
    )
    
    print(f"   Empirical Coverage:    {diagnostics['empirical_coverage']:.4f}")
    print(f"   Nominal Coverage:      {diagnostics['nominal_coverage']:.4f}")
    print(f"   Coverage Error:        {diagnostics['coverage_error']:.4f}")
    print(f"   Mean Interval Width:   {diagnostics['mean_interval_width']:.4f}")
    print(f"   Miscoverage Rate:      {diagnostics['miscoverage_rate']:.4f}")
    print(f"   Well Calibrated:       {diagnostics['is_well_calibrated']}")
    
    if diagnostics['warnings']:
        print("\n   ⚠️  Calibration Warnings:")
        for warning in diagnostics['warnings']:
            print(f"   - {warning}")
    else:
        print("\n   ✓ No calibration warnings - intervals are well calibrated!")
    
    # 6. Generate visualizations (Issue #17)
    print("\n6. Generating visualizations...")
    print("   (Plots saved in non-interactive mode)")
    
    # Plot prediction intervals
    print("   - Plotting prediction intervals...")
    plot_intervals(
        x=X_test.flatten(),
        y_pred=y_pred,
        lower=np.array([iv[0] for iv in intervals]),
        upper=np.array([iv[1] for iv in intervals]),
        y_true=y_test,
        title="UEQ Prediction Intervals with True Values",
        xlabel="X",
        ylabel="Y"
    )
    
    # Plot reliability diagram
    print("   - Plotting reliability diagram...")
    plot_reliability_diagram(
        y_true=y_test,
        y_pred=y_pred,
        intervals=intervals,
        n_bins=10,
        confidence=0.95,
        title="UEQ Reliability Diagram"
    )
    
    # Plot coverage vs confidence
    print("   - Plotting coverage vs interval width...")
    plot_coverage_vs_confidence(
        y_true=y_test,
        intervals=intervals,
        n_bins=15,
        title="Coverage vs Interval Width"
    )
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
    
    # 7. Summary of new features
    print("\n📊 Summary of New Features:")
    print("\n✓ Issue #9: Standardized UQ Evaluation Metrics")
    print("  - 7 comprehensive metrics: coverage, sharpness, ECE, MCE, etc.")
    print("  - Unified evaluate_uncertainty() function")
    print("  - Flexible metric selection")
    
    print("\n✓ Issue #10: Reliability & Calibration Diagnostics")
    print("  - check_calibration() with automatic warnings")
    print("  - Detects undercoverage, overcoverage, uncertainty collapse")
    print("  - Reliability diagrams for visual calibration assessment")
    
    print("\n✓ Issue #17: Enhanced Prediction Interval Visualizations")
    print("  - plot_intervals() with customizable options")
    print("  - plot_reliability_diagram() for calibration curves")
    print("  - plot_coverage_vs_confidence() for diagnostic plots")
    
    return metrics, diagnostics


if __name__ == "__main__":
    metrics, diagnostics = main()
    print("\nAll features demonstrated successfully! ✓")
