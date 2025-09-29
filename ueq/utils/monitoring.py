"""
Model monitoring and uncertainty drift detection utilities.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import warnings


class UQMonitor:
    """
    Monitor uncertainty quantification models for drift and performance degradation.
    
    Parameters
    ----------
    baseline_data : array-like, optional
        Baseline data for drift detection
    baseline_uncertainty : array-like, optional
        Baseline uncertainty estimates
    window_size : int, default=100
        Size of sliding window for monitoring
    drift_threshold : float, default=0.1
        Threshold for drift detection
    """
    
    def __init__(self, baseline_data=None, baseline_uncertainty=None, 
                 window_size=100, drift_threshold=0.1):
        self.baseline_data = baseline_data
        self.baseline_uncertainty = baseline_uncertainty
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        
        # Monitoring history
        self.uncertainty_history = deque(maxlen=window_size)
        self.prediction_history = deque(maxlen=window_size)
        self.drift_scores = deque(maxlen=window_size)
        
        # Statistics
        self.baseline_stats = None
        self.current_stats = None
        
    def update_baseline(self, data, uncertainty):
        """Update baseline statistics for drift detection."""
        self.baseline_data = data
        self.baseline_uncertainty = uncertainty
        self.baseline_stats = self._compute_stats(uncertainty)
        
    def _compute_stats(self, uncertainty):
        """Compute statistical properties of uncertainty estimates."""
        if isinstance(uncertainty, list):
            # Convert intervals to widths
            widths = [u[1] - u[0] for u in uncertainty]
            uncertainty = np.array(widths)
        
        return {
            'mean': np.mean(uncertainty),
            'std': np.std(uncertainty),
            'median': np.median(uncertainty),
            'q25': np.percentile(uncertainty, 25),
            'q75': np.percentile(uncertainty, 75),
            'min': np.min(uncertainty),
            'max': np.max(uncertainty)
        }
    
    def monitor(self, predictions, uncertainty):
        """
        Monitor new predictions and uncertainty estimates.
        
        Parameters
        ----------
        predictions : array-like
            New predictions
        uncertainty : array-like
            New uncertainty estimates
            
        Returns
        -------
        dict
            Monitoring results including drift score and alerts
        """
        # Update history
        self.prediction_history.append(predictions)
        self.uncertainty_history.append(uncertainty)
        
        # Compute current statistics
        self.current_stats = self._compute_stats(uncertainty)
        
        # Compute drift score
        drift_score = self._compute_drift_score()
        self.drift_scores.append(drift_score)
        
        # Generate alerts
        alerts = self._generate_alerts(drift_score)
        
        return {
            'drift_score': drift_score,
            'alerts': alerts,
            'current_stats': self.current_stats,
            'baseline_stats': self.baseline_stats,
            'history_size': len(self.uncertainty_history)
        }
    
    def _compute_drift_score(self):
        """Compute drift score between baseline and current uncertainty."""
        if self.baseline_stats is None or self.current_stats is None:
            return 0.0
        
        # Compute relative change in key statistics
        mean_change = abs(self.current_stats['mean'] - self.baseline_stats['mean']) / self.baseline_stats['mean']
        std_change = abs(self.current_stats['std'] - self.baseline_stats['std']) / self.baseline_stats['std']
        median_change = abs(self.current_stats['median'] - self.baseline_stats['median']) / self.baseline_stats['median']
        
        # Weighted combination
        drift_score = 0.4 * mean_change + 0.3 * std_change + 0.3 * median_change
        
        return drift_score
    
    def _generate_alerts(self, drift_score):
        """Generate alerts based on drift score and other metrics."""
        alerts = []
        
        if drift_score > self.drift_threshold:
            alerts.append({
                'type': 'drift',
                'severity': 'high' if drift_score > 2 * self.drift_threshold else 'medium',
                'message': f'Uncertainty drift detected: {drift_score:.3f} > {self.drift_threshold}',
                'drift_score': drift_score
            })
        
        # Check for extreme uncertainty values
        if self.current_stats and self.baseline_stats:
            if self.current_stats['mean'] > 2 * self.baseline_stats['mean']:
                alerts.append({
                    'type': 'uncertainty_spike',
                    'severity': 'high',
                    'message': 'Uncertainty significantly increased',
                    'current_mean': self.current_stats['mean'],
                    'baseline_mean': self.baseline_stats['mean']
                })
        
        return alerts
    
    def get_summary(self):
        """Get monitoring summary."""
        if not self.drift_scores:
            return "No monitoring data available"
        
        recent_drift = np.mean(list(self.drift_scores)[-10:]) if len(self.drift_scores) >= 10 else np.mean(list(self.drift_scores))
        
        return {
            'total_samples': len(self.uncertainty_history),
            'recent_drift_score': recent_drift,
            'max_drift_score': max(self.drift_scores) if self.drift_scores else 0,
            'alerts_count': sum(1 for score in self.drift_scores if score > self.drift_threshold),
            'status': 'healthy' if recent_drift < self.drift_threshold else 'warning'
        }


class PerformanceMonitor:
    """
    Monitor model performance metrics in production.
    
    Parameters
    ----------
    window_size : int, default=100
        Size of sliding window for performance tracking
    """
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.performance_history = deque(maxlen=window_size)
        self.timing_history = deque(maxlen=window_size)
        
    def log_performance(self, predictions, true_values, inference_time=None):
        """
        Log performance metrics.
        
        Parameters
        ----------
        predictions : array-like
            Model predictions
        true_values : array-like
            True target values
        inference_time : float, optional
            Inference time in seconds
        """
        # Compute basic metrics
        mse = np.mean((predictions - true_values) ** 2)
        mae = np.mean(np.abs(predictions - true_values))
        
        performance = {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'inference_time': inference_time
        }
        
        self.performance_history.append(performance)
        if inference_time is not None:
            self.timing_history.append(inference_time)
    
    def get_performance_summary(self):
        """Get performance summary."""
        if not self.performance_history:
            return "No performance data available"
        
        recent_performance = list(self.performance_history)[-10:]
        
        return {
            'recent_mse': np.mean([p['mse'] for p in recent_performance]),
            'recent_mae': np.mean([p['mae'] for p in recent_performance]),
            'recent_rmse': np.mean([p['rmse'] for p in recent_performance]),
            'avg_inference_time': np.mean(list(self.timing_history)) if self.timing_history else None,
            'total_samples': len(self.performance_history)
        }


def detect_uncertainty_drift(baseline_uncertainty, current_uncertainty, 
                           method='statistical', threshold=0.1):
    """
    Detect uncertainty drift between baseline and current estimates.
    
    Parameters
    ----------
    baseline_uncertainty : array-like
        Baseline uncertainty estimates
    current_uncertainty : array-like
        Current uncertainty estimates
    method : str, default='statistical'
        Drift detection method
    threshold : float, default=0.1
        Drift threshold
        
    Returns
    -------
    dict
        Drift detection results
    """
    if method == 'statistical':
        # Convert intervals to widths if needed
        if isinstance(baseline_uncertainty, list):
            baseline_widths = [u[1] - u[0] for u in baseline_uncertainty]
            current_widths = [u[1] - u[0] for u in current_uncertainty]
        else:
            baseline_widths = baseline_uncertainty
            current_widths = current_uncertainty
        
        # Statistical comparison
        baseline_mean = np.mean(baseline_widths)
        current_mean = np.mean(current_widths)
        
        drift_score = abs(current_mean - baseline_mean) / baseline_mean
        
        return {
            'drift_detected': drift_score > threshold,
            'drift_score': drift_score,
            'baseline_mean': baseline_mean,
            'current_mean': current_mean,
            'method': method
        }
    
    else:
        raise ValueError(f"Unknown drift detection method: {method}")
