"""
Reliability and Calibration Diagnostics for Uncertainty Quantification.

This module provides visual and diagnostic tools to interpret uncertainty behavior,
complementing numerical UQ metrics with reliability diagrams, coverage plots, and
automated warnings for miscalibration issues.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional

# Constants for calibration diagnostics
DEFAULT_TOLERANCE = 0.05  # Default tolerance for coverage deviation
UNCERTAINTY_COLLAPSE_THRESHOLD = 1e-6  # Threshold for detecting uncertainty collapse
CONSTANT_INTERVALS_THRESHOLD = 1e-6  # Threshold for detecting constant intervals
ECE_WARNING_THRESHOLD = 0.1  # ECE threshold for high calibration error
CONFIDENCE_GAP_THRESHOLD = 0.1  # Threshold for confidence gap warnings
OVERCONFIDENT_THRESHOLD = 0.1  # Threshold for detecting overconfident intervals


def plot_calibration(y_true: Union[np.ndarray, List],
                     y_pred: Union[np.ndarray, List],
                     uncertainty: Union[np.ndarray, List, List[Tuple]],
                     task: str = 'regression',
                     confidence: float = 0.95,
                     n_bins: int = 10,
                     figsize: Tuple[int, int] = (14, 10),
                     title: Optional[str] = None,
                     show_warnings: bool = True,
                     return_diagnostics: bool = False) -> Optional[Dict]:
    """
    Comprehensive calibration diagnostic plot for uncertainty quantification.
    
    Creates a multi-panel visualization combining:
    - Reliability diagram (nominal vs empirical coverage)
    - Coverage vs confidence/width analysis
    - Calibration error distribution
    - Diagnostic warnings for miscalibration
    
    This is the main diagnostic API for assessing UQ quality.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True target values.
    y_pred : array-like, shape (n_samples,)
        Point predictions (mean/median for regression, class probabilities for classification).
    uncertainty : array-like
        For regression: either intervals as [(lower, upper), ...] or
                       uncertainty estimates (standard deviations/half-widths).
        For classification: prediction probabilities (n_samples, n_classes) or
                           confidence scores (n_samples,).
    task : str, default='regression'
        Type of task: 'regression' or 'classification'.
    confidence : float, default=0.95
        Expected nominal confidence/coverage level.
    n_bins : int, default=10
        Number of bins for calibration analysis.
    figsize : tuple, default=(14, 10)
        Figure size (width, height).
    title : str, optional
        Main title for the plot. Auto-generated if None.
    show_warnings : bool, default=True
        Whether to display diagnostic warnings on the plot.
    return_diagnostics : bool, default=False
        If True, return diagnostic dictionary along with showing plot.
    
    Returns
    -------
    diagnostics : dict, optional
        Diagnostic information if return_diagnostics=True.
        Contains coverage metrics, warnings, and calibration errors.
    
    Examples
    --------
    For regression:
    
    >>> from ueq.diagnostics import plot_calibration
    >>> diagnostics = plot_calibration(
    ...     y_true=y_test,
    ...     y_pred=predictions,
    ...     uncertainty=intervals,  # or uncertainty estimates
    ...     task='regression',
    ...     return_diagnostics=True
    ... )
    
    For classification:
    
    >>> plot_calibration(
    ...     y_true=y_test,
    ...     y_pred=class_predictions,
    ...     uncertainty=probabilities,
    ...     task='classification'
    ... )
    
    Notes
    -----
    - For regression, uncertainty can be prediction intervals or uncertainty estimates
    - For classification, uses confidence-based calibration
    - Automatically detects and warns about common calibration issues
    - Works with both notebooks and scripts (uses matplotlib)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if task == 'regression':
        diagnostics = _plot_regression_calibration(
            y_true, y_pred, uncertainty, confidence, n_bins,
            figsize, title, show_warnings
        )
    elif task == 'classification':
        diagnostics = _plot_classification_calibration(
            y_true, y_pred, uncertainty, n_bins,
            figsize, title, show_warnings
        )
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'regression' or 'classification'.")
    
    if return_diagnostics:
        return diagnostics
    

def _plot_regression_calibration(y_true, y_pred, uncertainty, confidence,
                                  n_bins, figsize, title, show_warnings):
    """Internal function for regression calibration plots."""
    # Convert uncertainty to intervals if needed
    if isinstance(uncertainty, (list, tuple)) and len(uncertainty) > 0:
        if isinstance(uncertainty[0], (list, tuple)) and len(uncertainty[0]) == 2:
            # Already intervals
            intervals = np.array(uncertainty)
        else:
            # Uncertainty values - create symmetric intervals
            uncertainty = np.asarray(uncertainty)
            intervals = np.column_stack([y_pred - uncertainty, y_pred + uncertainty])
    else:
        uncertainty = np.asarray(uncertainty)
        if uncertainty.ndim == 2 and uncertainty.shape[1] == 2:
            intervals = uncertainty
        else:
            # Assume symmetric intervals
            intervals = np.column_stack([y_pred - uncertainty, y_pred + uncertainty])
    
    lower, upper = intervals[:, 0], intervals[:, 1]
    widths = upper - lower
    covered = (y_true >= lower) & (y_true <= upper)
    
    # Compute diagnostics
    empirical_coverage = covered.mean()
    mean_width = widths.mean()
    miscoverage = 1.0 - empirical_coverage
    
    # Detect issues
    warnings = []
    tolerance = DEFAULT_TOLERANCE
    
    if abs(empirical_coverage - confidence) > tolerance:
        if empirical_coverage < confidence:
            warnings.append(
                f"⚠ UNDERCOVERAGE: {empirical_coverage:.1%} vs {confidence:.1%} nominal"
            )
        else:
            warnings.append(
                f"⚠ OVERCOVERAGE: {empirical_coverage:.1%} vs {confidence:.1%} nominal"
            )
    
    if mean_width < UNCERTAINTY_COLLAPSE_THRESHOLD:
        warnings.append("⚠ UNCERTAINTY COLLAPSE: Intervals extremely narrow")
    
    if np.std(widths) < CONSTANT_INTERVALS_THRESHOLD:
        warnings.append("⚠ CONSTANT INTERVALS: All intervals have similar width")
    
    # Check for overconfident intervals
    if empirical_coverage < confidence - OVERCONFIDENT_THRESHOLD:
        warnings.append("⚠ OVERCONFIDENT INTERVALS: Coverage significantly below nominal")
    
    # Create multi-panel plot
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Reliability diagram
    ax1 = fig.add_subplot(gs[0, 0])
    nominal_levels = np.linspace(0, 1, n_bins)
    empirical_coverages = []
    
    for nom_level in nominal_levels:
        width = widths * nom_level
        center = (upper + lower) / 2
        scaled_lower = center - width / 2
        scaled_upper = center + width / 2
        covered_scaled = (y_true >= scaled_lower) & (y_true <= scaled_upper)
        empirical_coverages.append(covered_scaled.mean())
    
    ax1.plot(nominal_levels, empirical_coverages, 'o-', linewidth=2,
            markersize=8, label='Empirical coverage', color='steelblue')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    ax1.fill_between([0, 1], [0, 1], alpha=0.1, color='green')
    ax1.set_xlabel('Nominal Coverage', fontsize=11)
    ax1.set_ylabel('Empirical Coverage', fontsize=11)
    ax1.set_title('Reliability Diagram', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Panel 2: Coverage vs interval width
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Bin by interval width
    width_bins = np.percentile(widths, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(widths, width_bins[1:-1])
    
    bin_widths = []
    bin_coverages = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_widths.append(widths[mask].mean())
            bin_coverages.append(covered[mask].mean())
            bin_counts.append(mask.sum())
    
    scatter = ax2.scatter(bin_widths, bin_coverages,
                         s=[c*3 for c in bin_counts],
                         alpha=0.6, c=bin_coverages, cmap='RdYlGn',
                         vmin=0, vmax=1, edgecolors='black', linewidths=1)
    ax2.axhline(y=confidence, color='r', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Target ({confidence:.0%})')
    plt.colorbar(scatter, ax=ax2, label='Coverage')
    ax2.set_xlabel('Mean Interval Width', fontsize=11)
    ax2.set_ylabel('Empirical Coverage', fontsize=11)
    ax2.set_title('Coverage vs Interval Width', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # Panel 3: Calibration error distribution
    ax3 = fig.add_subplot(gs[1, 0])
    calibration_errors = np.array(empirical_coverages) - nominal_levels
    colors = ['red' if e < 0 else 'green' for e in calibration_errors]
    bars = ax3.bar(range(len(calibration_errors)), calibration_errors,
                   color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=1.5)
    ax3.set_xlabel('Bin Index', fontsize=11)
    ax3.set_ylabel('Calibration Error\n(Empirical - Nominal)', fontsize=11)
    ax3.set_title('Calibration Error Distribution', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Summary statistics and warnings
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Summary text
    summary_text = f"""
Calibration Summary
{'=' * 40}

Coverage Statistics:
  • Empirical Coverage: {empirical_coverage:.2%}
  • Nominal Coverage:   {confidence:.2%}
  • Coverage Error:     {empirical_coverage - confidence:+.2%}
  • Miscoverage Rate:   {miscoverage:.2%}

Interval Statistics:
  • Mean Width:         {mean_width:.3f}
  • Std Width:          {np.std(widths):.3f}
  • Min Width:          {np.min(widths):.3f}
  • Max Width:          {np.max(widths):.3f}

Calibration Quality:
  • ECE (Est.):         {abs(empirical_coverage - confidence):.4f}
"""
    
    if show_warnings and warnings:
        summary_text += f"\n{'=' * 40}\nDiagnostic Warnings:\n"
        for warning in warnings:
            summary_text += f"  {warning}\n"
    else:
        summary_text += f"\n{'=' * 40}\n✓ Well-calibrated (no issues detected)\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Overall title
    if title is None:
        title = f'Calibration Diagnostics - Regression (n={len(y_true)})'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()
    
    # Return diagnostics
    diagnostics = {
        'empirical_coverage': empirical_coverage,
        'nominal_coverage': confidence,
        'coverage_error': empirical_coverage - confidence,
        'mean_interval_width': mean_width,
        'miscoverage_rate': miscoverage,
        'warnings': warnings,
        'is_well_calibrated': len(warnings) == 0,
        'calibration_errors': calibration_errors.tolist(),
        'interval_width_std': float(np.std(widths))
    }
    
    return diagnostics


def _plot_classification_calibration(y_true, y_pred, uncertainty, n_bins,
                                      figsize, title, show_warnings):
    """Internal function for classification calibration plots."""
    y_true = np.asarray(y_true)
    
    # Extract confidence scores
    if uncertainty.ndim == 2:
        # Probability matrix - use max probability as confidence
        confidence_scores = np.max(uncertainty, axis=1)
        predicted_classes = np.argmax(uncertainty, axis=1)
    else:
        # Already confidence scores
        confidence_scores = np.asarray(uncertainty)
        predicted_classes = np.asarray(y_pred)
    
    # Ensure y_true is in correct format
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # Compute accuracy per confidence bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        mask = (confidence_scores >= lower) & (confidence_scores < upper)
        
        if i == n_bins - 1:  # Include upper bound for last bin
            mask = (confidence_scores >= lower) & (confidence_scores <= upper)
        
        if mask.sum() > 0:
            bin_conf = confidence_scores[mask].mean()
            bin_acc = (predicted_classes[mask] == y_true[mask]).mean()
            bin_confidences.append(bin_conf)
            bin_accuracies.append(bin_acc)
            bin_counts.append(mask.sum())
    
    # Compute ECE (Expected Calibration Error)
    ece = 0.0
    for conf, acc, count in zip(bin_confidences, bin_accuracies, bin_counts):
        ece += abs(conf - acc) * (count / len(y_true))
    
    # Detect issues
    warnings = []
    overall_accuracy = (predicted_classes == y_true).mean()
    mean_confidence = confidence_scores.mean()
    
    if ece > ECE_WARNING_THRESHOLD:
        warnings.append(f"⚠ HIGH CALIBRATION ERROR: ECE = {ece:.3f}")
    
    if mean_confidence > overall_accuracy + CONFIDENCE_GAP_THRESHOLD:
        warnings.append(f"⚠ OVERCONFIDENT: Mean confidence ({mean_confidence:.2%}) >> Accuracy ({overall_accuracy:.2%})")
    
    if mean_confidence < overall_accuracy - CONFIDENCE_GAP_THRESHOLD:
        warnings.append(f"⚠ UNDERCONFIDENT: Mean confidence ({mean_confidence:.2%}) << Accuracy ({overall_accuracy:.2%})")
    
    # Create multi-panel plot
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Reliability diagram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(bin_confidences, bin_accuracies, 'o-', linewidth=2,
            markersize=8, label='Model calibration', color='steelblue')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    ax1.fill_between([0, 1], [0, 1], alpha=0.1, color='green')
    ax1.set_xlabel('Confidence', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Reliability Diagram', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Panel 2: Confidence histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(confidence_scores, bins=20, alpha=0.7, color='steelblue',
            edgecolor='black', density=True)
    ax2.axvline(mean_confidence, color='r', linestyle='--', linewidth=2,
               label=f'Mean: {mean_confidence:.2f}')
    ax2.set_xlabel('Confidence', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Confidence Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Calibration error bars
    ax3 = fig.add_subplot(gs[1, 0])
    calibration_errors = np.array(bin_accuracies) - np.array(bin_confidences)
    colors = ['red' if e < 0 else 'green' for e in calibration_errors]
    ax3.bar(range(len(calibration_errors)), calibration_errors,
           color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=1.5)
    ax3.set_xlabel('Bin Index', fontsize=11)
    ax3.set_ylabel('Calibration Error\n(Accuracy - Confidence)', fontsize=11)
    ax3.set_title('Calibration Error Distribution', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Summary statistics and warnings
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Summary text
    summary_text = f"""
Calibration Summary
{'=' * 40}

Performance:
  • Overall Accuracy:   {overall_accuracy:.2%}
  • Mean Confidence:    {mean_confidence:.2%}
  • Confidence Gap:     {mean_confidence - overall_accuracy:+.2%}

Calibration Metrics:
  • ECE:                {ece:.4f}
  • # Bins:             {len(bin_confidences)}
  • # Samples:          {len(y_true)}

Confidence Stats:
  • Min Confidence:     {np.min(confidence_scores):.3f}
  • Max Confidence:     {np.max(confidence_scores):.3f}
  • Std Confidence:     {np.std(confidence_scores):.3f}
"""
    
    if show_warnings and warnings:
        summary_text += f"\n{'=' * 40}\nDiagnostic Warnings:\n"
        for warning in warnings:
            summary_text += f"  {warning}\n"
    else:
        summary_text += f"\n{'=' * 40}\n✓ Well-calibrated (no issues detected)\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Overall title
    if title is None:
        title = f'Calibration Diagnostics - Classification (n={len(y_true)})'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()
    
    # Return diagnostics
    diagnostics = {
        'expected_calibration_error': ece,
        'overall_accuracy': overall_accuracy,
        'mean_confidence': mean_confidence,
        'confidence_gap': mean_confidence - overall_accuracy,
        'warnings': warnings,
        'is_well_calibrated': len(warnings) == 0,
        'bin_confidences': bin_confidences,
        'bin_accuracies': bin_accuracies,
        'bin_counts': bin_counts
    }
    
    return diagnostics


# Convenience functions for specific diagnostics

def check_regression_calibration(y_true, intervals, confidence=0.95, tolerance=0.05):
    """
    Quick calibration check for regression intervals.
    
    Returns diagnostic warnings without plotting.
    
    Parameters
    ----------
    y_true : array-like
        True values.
    intervals : array-like, shape (n_samples, 2)
        Prediction intervals [(lower, upper), ...].
    confidence : float
        Expected nominal coverage.
    tolerance : float
        Acceptable deviation from nominal coverage.
    
    Returns
    -------
    diagnostics : dict
        Diagnostic information with warnings.
    """
    y_true = np.asarray(y_true)
    intervals = np.asarray(intervals)
    lower, upper = intervals[:, 0], intervals[:, 1]
    
    covered = (y_true >= lower) & (y_true <= upper)
    empirical_coverage = covered.mean()
    mean_width = (upper - lower).mean()
    
    warnings = []
    
    if abs(empirical_coverage - confidence) > tolerance:
        if empirical_coverage < confidence:
            warnings.append(
                f"UNDERCOVERAGE: Empirical coverage ({empirical_coverage:.3f}) "
                f"is below nominal ({confidence:.3f})"
            )
        else:
            warnings.append(
                f"OVERCOVERAGE: Empirical coverage ({empirical_coverage:.3f}) "
                f"exceeds nominal ({confidence:.3f})"
            )
    
    if mean_width < UNCERTAINTY_COLLAPSE_THRESHOLD:
        warnings.append(
            f"UNCERTAINTY COLLAPSE: Mean interval width ({mean_width:.6f}) "
            "is extremely small"
        )
    
    if np.std(upper - lower) < CONSTANT_INTERVALS_THRESHOLD:
        warnings.append(
            "CONSTANT INTERVALS: All intervals have similar width"
        )
    
    return {
        'empirical_coverage': empirical_coverage,
        'nominal_coverage': confidence,
        'coverage_error': empirical_coverage - confidence,
        'mean_interval_width': mean_width,
        'warnings': warnings,
        'is_well_calibrated': len(warnings) == 0
    }


def check_classification_calibration(y_true, y_pred_proba, n_bins=10):
    """
    Quick calibration check for classification predictions.
    
    Returns diagnostic warnings without plotting.
    
    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred_proba : array-like, shape (n_samples, n_classes) or (n_samples,)
        Predicted probabilities or confidence scores.
    n_bins : int
        Number of bins for ECE calculation.
    
    Returns
    -------
    diagnostics : dict
        Diagnostic information with warnings and ECE.
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    if y_pred_proba.ndim == 2:
        confidence_scores = np.max(y_pred_proba, axis=1)
        predicted_classes = np.argmax(y_pred_proba, axis=1)
    else:
        confidence_scores = y_pred_proba
        predicted_classes = (confidence_scores > 0.5).astype(int)
    
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    
    # Compute ECE
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        mask = (confidence_scores >= lower) & (confidence_scores < upper)
        
        if i == n_bins - 1:
            mask = (confidence_scores >= lower) & (confidence_scores <= upper)
        
        if mask.sum() > 0:
            bin_conf = confidence_scores[mask].mean()
            bin_acc = (predicted_classes[mask] == y_true[mask]).mean()
            ece += abs(bin_conf - bin_acc) * (mask.sum() / len(y_true))
    
    overall_accuracy = (predicted_classes == y_true).mean()
    mean_confidence = confidence_scores.mean()
    
    warnings = []
    
    if ece > ECE_WARNING_THRESHOLD:
        warnings.append(f"HIGH CALIBRATION ERROR: ECE = {ece:.3f}")
    
    if mean_confidence > overall_accuracy + CONFIDENCE_GAP_THRESHOLD:
        warnings.append(
            f"OVERCONFIDENT: Mean confidence ({mean_confidence:.2%}) "
            f"exceeds accuracy ({overall_accuracy:.2%})"
        )
    
    return {
        'expected_calibration_error': ece,
        'overall_accuracy': overall_accuracy,
        'mean_confidence': mean_confidence,
        'warnings': warnings,
        'is_well_calibrated': len(warnings) == 0
    }
