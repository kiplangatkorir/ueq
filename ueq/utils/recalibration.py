"""
Drift-aware recalibration and uncertainty inflation module.

Provides automatic recalibration and uncertainty adjustment mechanisms
for production UQ systems under distribution drift.
"""

import numpy as np
from typing import Optional, Union, Tuple, Callable, Dict
from collections import deque
import warnings


class DriftAwareRecalibrator:
    """
    Automatic recalibration module that adjusts UQ estimates under detected drift.
    
    Integrates with drift detection to trigger recalibration when coverage
    violations or significant distribution shifts are detected.
    
    Parameters
    ----------
    uq_method : object
        UQ method object with fit(), predict(), and update() methods.
        Should support online calibration (e.g., AdaptiveConformalUQ, OnlineConformalUQ).
    drift_detector : callable, optional
        Function that takes (X, uncertainty) and returns drift score.
        If None, uses built-in drift detection based on uncertainty statistics.
    drift_threshold : float, default=0.1
        Drift score threshold above which recalibration is triggered.
    coverage_check_interval : int, default=50
        How often to check empirical coverage for recalibration triggers.
    target_coverage : float, default=0.9
        Target empirical coverage level (1 - alpha).
    coverage_tolerance : float, default=0.05
        Acceptable deviation from target coverage before recalibration.
    min_calibration_samples : int, default=50
        Minimum samples needed before recalibration.
    
    Attributes
    ----------
    recalibration_history : list
        History of recalibration events with timestamps and metrics.
    drift_scores : deque
        Recent drift scores.
    coverage_history : deque
        Recent empirical coverage values.
    
    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from ueq.methods.online_conformal import AdaptiveConformalUQ
    >>> 
    >>> model = LinearRegression()
    >>> uq = AdaptiveConformalUQ(model, alpha=0.1)
    >>> recalibrator = DriftAwareRecalibrator(uq, drift_threshold=0.1)
    >>> 
    >>> # Initial fit
    >>> recalibrator.fit(X_train, y_train, X_calib, y_calib)
    >>> 
    >>> # Stream data with automatic recalibration
    >>> for X_batch, y_batch in data_stream:
    ...     preds, intervals = recalibrator.predict(X_batch, return_interval=True)
    ...     recalibrator.update(X_batch, y_batch)  # Triggers recalibration if needed
    """
    
    def __init__(self, uq_method, drift_detector: Optional[Callable] = None,
                 drift_threshold: float = 0.1, coverage_check_interval: int = 50,
                 target_coverage: float = 0.9, coverage_tolerance: float = 0.05,
                 min_calibration_samples: int = 50):
        self.uq_method = uq_method
        self.drift_detector = drift_detector
        self.drift_threshold = drift_threshold
        self.coverage_check_interval = coverage_check_interval
        self.target_coverage = target_coverage
        self.coverage_tolerance = coverage_tolerance
        self.min_calibration_samples = min_calibration_samples
        
        # History tracking
        self.recalibration_history = []
        self.drift_scores = deque(maxlen=100)
        self.coverage_history = deque(maxlen=100)
        self.baseline_uncertainty_stats = None
        
        # Counters
        self.samples_since_recalibration = 0
        self.total_recalibrations = 0
        
    def fit(self, X_train, y_train, X_calib, y_calib):
        """
        Initial fit of the UQ method.
        
        Parameters
        ----------
        X_train : array-like
            Training features.
        y_train : array-like
            Training targets.
        X_calib : array-like
            Calibration features.
        y_calib : array-like
            Calibration targets.
        
        Returns
        -------
        self : object
            Fitted recalibrator.
        """
        self.uq_method.fit(X_train, y_train, X_calib, y_calib)
        
        # Establish baseline uncertainty statistics
        preds, intervals = self.uq_method.predict(X_calib, return_interval=True)
        self.baseline_uncertainty_stats = self._compute_uncertainty_stats(intervals)
        
        return self
    
    def predict(self, X, return_interval=False):
        """
        Predict with current UQ estimates.
        
        Parameters
        ----------
        X : array-like
            Input features.
        return_interval : bool
            Whether to return prediction intervals.
        
        Returns
        -------
        predictions : array-like
            Point predictions.
        intervals : list of tuples, optional
            Prediction intervals if return_interval=True.
        """
        return self.uq_method.predict(X, return_interval=return_interval)
    
    def update(self, X_new, y_new):
        """
        Update with new data, checking for drift and triggering recalibration if needed.
        
        Parameters
        ----------
        X_new : array-like
            New input features.
        y_new : array-like
            New true labels.
        """
        # Get predictions before update
        preds, intervals = self.uq_method.predict(X_new, return_interval=True)
        
        # Compute empirical coverage
        covered = np.array([(y >= l and y <= u) for y, (l, u) in zip(y_new, intervals)])
        empirical_coverage = covered.mean()
        self.coverage_history.append(empirical_coverage)
        
        # Detect drift
        drift_score = self._detect_drift(X_new, intervals)
        self.drift_scores.append(drift_score)
        
        # Update UQ method
        if hasattr(self.uq_method, 'update'):
            self.uq_method.update(X_new, y_new)
        
        self.samples_since_recalibration += len(y_new)
        
        # Check if recalibration is needed
        if self._should_recalibrate(drift_score, empirical_coverage):
            self._trigger_recalibration(X_new, y_new, drift_score, empirical_coverage)
    
    def _detect_drift(self, X, intervals) -> float:
        """
        Detect drift using custom detector or built-in uncertainty-based detection.
        
        Parameters
        ----------
        X : array-like
            Input features.
        intervals : list of tuples
            Prediction intervals.
        
        Returns
        -------
        drift_score : float
            Drift score (higher means more drift).
        """
        if self.drift_detector is not None:
            # Use custom drift detector
            uncertainty = np.array([u - l for l, u in intervals])
            return self.drift_detector(X, uncertainty)
        else:
            # Built-in uncertainty-based drift detection
            current_stats = self._compute_uncertainty_stats(intervals)
            
            if self.baseline_uncertainty_stats is None:
                return 0.0
            
            # Compute relative change in key statistics
            mean_change = abs(current_stats['mean'] - self.baseline_uncertainty_stats['mean'])
            mean_change /= (self.baseline_uncertainty_stats['mean'] + 1e-8)
            
            std_change = abs(current_stats['std'] - self.baseline_uncertainty_stats['std'])
            std_change /= (self.baseline_uncertainty_stats['std'] + 1e-8)
            
            # Weighted drift score
            drift_score = 0.6 * mean_change + 0.4 * std_change
            
            return drift_score
    
    def _compute_uncertainty_stats(self, intervals) -> Dict[str, float]:
        """Compute statistics of uncertainty estimates."""
        widths = np.array([u - l for l, u in intervals])
        
        return {
            'mean': np.mean(widths),
            'std': np.std(widths),
            'median': np.median(widths),
            'q25': np.percentile(widths, 25),
            'q75': np.percentile(widths, 75)
        }
    
    def _should_recalibrate(self, drift_score: float, empirical_coverage: float) -> bool:
        """
        Determine if recalibration should be triggered.
        
        Parameters
        ----------
        drift_score : float
            Current drift score.
        empirical_coverage : float
            Recent empirical coverage.
        
        Returns
        -------
        should_recalibrate : bool
            Whether to trigger recalibration.
        """
        # Need minimum samples before recalibrating
        if self.samples_since_recalibration < self.min_calibration_samples:
            return False
        
        # Check drift threshold
        if drift_score > self.drift_threshold:
            return True
        
        # Check coverage violation
        if len(self.coverage_history) >= self.coverage_check_interval:
            recent_coverage = np.mean(list(self.coverage_history)[-self.coverage_check_interval:])
            if abs(recent_coverage - self.target_coverage) > self.coverage_tolerance:
                return True
        
        return False
    
    def _trigger_recalibration(self, X_new, y_new, drift_score, empirical_coverage):
        """
        Trigger recalibration and log the event.
        
        Parameters
        ----------
        X_new : array-like
            Recent data to use for recalibration.
        y_new : array-like
            Recent labels.
        drift_score : float
            Drift score that triggered recalibration.
        empirical_coverage : float
            Current empirical coverage.
        """
        # Log recalibration event
        event = {
            'total_recalibrations': self.total_recalibrations + 1,
            'samples_processed': self.samples_since_recalibration,
            'drift_score': drift_score,
            'empirical_coverage': empirical_coverage,
            'target_coverage': self.target_coverage,
            'reason': []
        }
        
        if drift_score > self.drift_threshold:
            event['reason'].append(f'drift_detected (score={drift_score:.3f})')
        
        if abs(empirical_coverage - self.target_coverage) > self.coverage_tolerance:
            event['reason'].append(f'coverage_violation ({empirical_coverage:.3f} vs {self.target_coverage:.3f})')
        
        self.recalibration_history.append(event)
        
        # Update baseline statistics
        preds, intervals = self.uq_method.predict(X_new, return_interval=True)
        self.baseline_uncertainty_stats = self._compute_uncertainty_stats(intervals)
        
        # Reset counter
        self.samples_since_recalibration = 0
        self.total_recalibrations += 1
        
        warnings.warn(
            f"Recalibration triggered (#{self.total_recalibrations}): "
            f"{', '.join(event['reason'])}", UserWarning
        )
    
    def get_recalibration_summary(self) -> Dict:
        """
        Get summary of recalibration history.
        
        Returns
        -------
        summary : dict
            Summary statistics about recalibrations.
        """
        if not self.recalibration_history:
            return {
                'total_recalibrations': 0,
                'status': 'no_recalibrations_yet'
            }
        
        recent_drift = np.mean(list(self.drift_scores)[-10:]) if len(self.drift_scores) >= 10 else 0
        recent_coverage = np.mean(list(self.coverage_history)[-10:]) if len(self.coverage_history) >= 10 else 0
        
        return {
            'total_recalibrations': self.total_recalibrations,
            'recent_drift_score': recent_drift,
            'recent_coverage': recent_coverage,
            'target_coverage': self.target_coverage,
            'samples_since_last_recalibration': self.samples_since_recalibration,
            'status': 'healthy' if recent_drift < self.drift_threshold else 'drift_detected'
        }


class UncertaintyInflator:
    """
    Inflate uncertainty estimates under detected drift or OoD conditions.
    
    Prevents overconfidence by widening intervals/sets when model operates
    outside its training distribution.
    
    Parameters
    ----------
    base_uq_method : object
        Base UQ method that provides initial uncertainty estimates.
    inflation_strategy : {"multiplicative", "additive", "adaptive"}, default="multiplicative"
        How to inflate uncertainty:
        - "multiplicative": multiply interval width by inflation factor
        - "additive": add constant to interval width
        - "adaptive": inflation proportional to drift severity
    max_inflation : float, default=2.0
        Maximum inflation factor or additive constant.
    drift_threshold : float, default=0.1
        Drift score above which inflation starts.
    
    Examples
    --------
    >>> from ueq.methods.conformal import ConformalUQ
    >>> 
    >>> base_uq = ConformalUQ(model, alpha=0.1)
    >>> inflator = UncertaintyInflator(base_uq, inflation_strategy="multiplicative",
    ...                                max_inflation=2.0)
    >>> 
    >>> inflator.fit(X_train, y_train, X_calib, y_calib)
    >>> 
    >>> # Predictions with automatic inflation under drift
    >>> preds, intervals = inflator.predict(X_test, drift_score=0.3, return_interval=True)
    """
    
    def __init__(self, base_uq_method, inflation_strategy: str = "multiplicative",
                 max_inflation: float = 2.0, drift_threshold: float = 0.1):
        self.base_uq_method = base_uq_method
        self.inflation_strategy = inflation_strategy
        self.max_inflation = max_inflation
        self.drift_threshold = drift_threshold
        
        # Validate strategy
        valid_strategies = ["multiplicative", "additive", "adaptive"]
        if inflation_strategy not in valid_strategies:
            raise ValueError(f"inflation_strategy must be one of {valid_strategies}")
        
        # Track inflation history
        self.inflation_history = []
    
    def fit(self, X_train, y_train, X_calib, y_calib):
        """Fit the base UQ method."""
        self.base_uq_method.fit(X_train, y_train, X_calib, y_calib)
        return self
    
    def predict(self, X, drift_score: Optional[float] = None, 
                return_interval=False):
        """
        Predict with inflated uncertainty based on drift score.
        
        Parameters
        ----------
        X : array-like
            Input features.
        drift_score : float, optional
            Drift severity score. If None, no inflation applied.
        return_interval : bool
            Whether to return prediction intervals.
        
        Returns
        -------
        predictions : array-like
            Point predictions.
        intervals : list of tuples, optional
            Inflated prediction intervals if return_interval=True.
        """
        # Get base predictions and intervals
        result = self.base_uq_method.predict(X, return_interval=True)
        
        if isinstance(result, tuple):
            predictions, intervals = result
        else:
            # Classification - prediction sets
            if not return_interval:
                return result
            predictions = result
            intervals = result
        
        # Compute inflation factor
        inflation_factor = self._compute_inflation_factor(drift_score)
        
        # Apply inflation
        inflated_intervals = self._inflate_intervals(intervals, inflation_factor)
        
        # Track inflation
        self.inflation_history.append({
            'drift_score': drift_score,
            'inflation_factor': inflation_factor
        })
        
        return (predictions, inflated_intervals) if return_interval else predictions
    
    def _compute_inflation_factor(self, drift_score: Optional[float]) -> float:
        """
        Compute inflation factor based on drift score and strategy.
        
        Parameters
        ----------
        drift_score : float or None
            Drift severity score.
        
        Returns
        -------
        inflation_factor : float
            Factor by which to inflate uncertainty.
        """
        if drift_score is None or drift_score <= self.drift_threshold:
            return 1.0  # No inflation
        
        if self.inflation_strategy == "multiplicative":
            # Linear interpolation from 1.0 to max_inflation
            excess_drift = drift_score - self.drift_threshold
            inflation = 1.0 + (self.max_inflation - 1.0) * min(excess_drift / self.drift_threshold, 1.0)
            return min(inflation, self.max_inflation)
        
        elif self.inflation_strategy == "additive":
            # Return additive constant based on drift
            excess_drift = drift_score - self.drift_threshold
            return self.max_inflation * min(excess_drift / self.drift_threshold, 1.0)
        
        elif self.inflation_strategy == "adaptive":
            # Exponential inflation for severe drift
            excess_drift = drift_score - self.drift_threshold
            inflation = 1.0 + (self.max_inflation - 1.0) * (1 - np.exp(-excess_drift))
            return min(inflation, self.max_inflation)
        
        return 1.0
    
    def _inflate_intervals(self, intervals, inflation_factor: float):
        """
        Apply inflation to prediction intervals.
        
        Parameters
        ----------
        intervals : list of tuples
            Original prediction intervals.
        inflation_factor : float
            Inflation factor.
        
        Returns
        -------
        inflated_intervals : list of tuples
            Inflated intervals.
        """
        if inflation_factor <= 1.0:
            return intervals
        
        inflated = []
        for lower, upper in intervals:
            center = (lower + upper) / 2
            width = upper - lower
            
            if self.inflation_strategy in ["multiplicative", "adaptive"]:
                # Multiplicative inflation
                new_width = width * inflation_factor
            else:  # additive
                # Additive inflation
                new_width = width + inflation_factor
            
            new_lower = center - new_width / 2
            new_upper = center + new_width / 2
            inflated.append((new_lower, new_upper))
        
        return inflated
    
    def get_inflation_summary(self) -> Dict:
        """
        Get summary of inflation history.
        
        Returns
        -------
        summary : dict
            Summary statistics about uncertainty inflation.
        """
        if not self.inflation_history:
            return {
                'mean_inflation': 1.0,
                'max_inflation': 1.0,
                'inflation_events': 0
            }
        
        factors = [h['inflation_factor'] for h in self.inflation_history]
        
        return {
            'mean_inflation': np.mean(factors),
            'max_inflation': np.max(factors),
            'inflation_events': sum(1 for f in factors if f > 1.0),
            'total_predictions': len(factors)
        }
