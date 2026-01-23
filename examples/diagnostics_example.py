"""
Example: Using Calibration Diagnostics

This example demonstrates how to use the diagnostics module to assess
uncertainty quantification quality for both regression and classification tasks.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # For non-interactive environments
import matplotlib.pyplot as plt

from ueq import UQ, plot_calibration, check_regression_calibration
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

print("=" * 70)
print("Example: Calibration Diagnostics for Uncertainty Quantification")
print("=" * 70)

# ============================================================================
# Part 1: Regression Example
# ============================================================================
print("\n[1] REGRESSION CALIBRATION DIAGNOSTICS")
print("-" * 70)

# Generate synthetic regression data
X, y = make_regression(n_samples=500, n_features=10, noise=5.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model with UQ
model = RandomForestRegressor(n_estimators=10, random_state=42)
uq = UQ(model)
uq.fit(X_train, y_train)

# Get predictions with intervals
y_pred, intervals = uq.predict(X_test, return_interval=True)

print(f"Generated {len(y_test)} test predictions with 95% confidence intervals")

# Quick calibration check (no plot)
print("\nQuick Calibration Check:")
diagnostics = check_regression_calibration(y_test, intervals, confidence=0.95)
print(f"  • Empirical Coverage: {diagnostics['empirical_coverage']:.2%}")
print(f"  • Nominal Coverage:   {diagnostics['nominal_coverage']:.2%}")
print(f"  • Mean Interval Width: {diagnostics['mean_interval_width']:.2f}")
print(f"  • Well Calibrated?    {diagnostics['is_well_calibrated']}")

if diagnostics['warnings']:
    print("\nWarnings:")
    for warning in diagnostics['warnings']:
        print(f"  ⚠ {warning}")

# Full calibration diagnostic plot
print("\nGenerating comprehensive calibration diagnostic plot...")
plt.close('all')
full_diagnostics = plot_calibration(
    y_true=y_test,
    y_pred=y_pred,
    uncertainty=intervals,
    task='regression',
    confidence=0.95,
    n_bins=10,
    title="Regression Calibration Diagnostics - Random Forest",
    return_diagnostics=True
)
plt.savefig('/tmp/regression_calibration_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close('all')
print("✓ Saved plot to /tmp/regression_calibration_diagnostics.png")

print(f"\nFull Diagnostics Summary:")
print(f"  • Coverage Error:     {full_diagnostics['coverage_error']:+.3f}")
print(f"  • Miscoverage Rate:   {full_diagnostics['miscoverage_rate']:.2%}")
print(f"  • Interval Width Std: {full_diagnostics['interval_width_std']:.3f}")

# ============================================================================
# Part 2: Classification Example
# ============================================================================
print("\n" + "=" * 70)
print("[2] CLASSIFICATION CALIBRATION DIAGNOSTICS")
print("-" * 70)

# Generate synthetic classification data
X_clf, y_clf = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=3,
    random_state=42
)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42
)

# Train classifier
clf_model = RandomForestClassifier(n_estimators=50, random_state=42)
clf_model.fit(X_train_clf, y_train_clf)

# Get predictions and probabilities
y_pred_clf = clf_model.predict(X_test_clf)
y_pred_proba = clf_model.predict_proba(X_test_clf)

print(f"Generated {len(y_test_clf)} test predictions with class probabilities")

# Quick calibration check (no plot)
from ueq import check_classification_calibration

print("\nQuick Calibration Check:")
clf_diagnostics = check_classification_calibration(y_test_clf, y_pred_proba)
print(f"  • Overall Accuracy:   {clf_diagnostics['overall_accuracy']:.2%}")
print(f"  • Mean Confidence:    {clf_diagnostics['mean_confidence']:.2%}")
print(f"  • ECE:                {clf_diagnostics['expected_calibration_error']:.4f}")
print(f"  • Well Calibrated?    {clf_diagnostics['is_well_calibrated']}")

if clf_diagnostics['warnings']:
    print("\nWarnings:")
    for warning in clf_diagnostics['warnings']:
        print(f"  ⚠ {warning}")

# Full calibration diagnostic plot
print("\nGenerating comprehensive calibration diagnostic plot...")
plt.close('all')
clf_full_diagnostics = plot_calibration(
    y_true=y_test_clf,
    y_pred=y_pred_clf,
    uncertainty=y_pred_proba,
    task='classification',
    n_bins=10,
    title="Classification Calibration Diagnostics - Random Forest",
    return_diagnostics=True
)
plt.savefig('/tmp/classification_calibration_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close('all')
print("✓ Saved plot to /tmp/classification_calibration_diagnostics.png")

print(f"\nFull Diagnostics Summary:")
print(f"  • Confidence Gap:     {clf_full_diagnostics['confidence_gap']:+.3f}")
print(f"  • Number of Bins:     {len(clf_full_diagnostics['bin_counts'])}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE COMPLETED SUCCESSFULLY")
print("=" * 70)
print("\nKey Takeaways:")
print("  1. Use plot_calibration() for comprehensive visual diagnostics")
print("  2. Use check_*_calibration() for quick programmatic checks")
print("  3. Works seamlessly with ueq.UQ for regression")
print("  4. Supports both regression and classification tasks")
print("  5. Provides automated warnings for common calibration issues")
print("\nNext Steps:")
print("  • Check the generated plots in /tmp/")
print("  • Integrate diagnostics into your ML pipeline")
print("  • Use warnings to improve model calibration")
print("=" * 70)
