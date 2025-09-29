# UEQ Production Deployment Guide

## Table of Contents

1. [Production Architecture](#production-architecture)
2. [Model Monitoring](#model-monitoring)
3. [Performance Optimization](#performance-optimization)
4. [Deployment Patterns](#deployment-patterns)
5. [Monitoring and Alerting](#monitoring-and-alerting)
6. [Scaling Strategies](#scaling-strategies)
7. [Best Practices](#best-practices)

## Production Architecture

### 1. Service Architecture

```python
from ueq import UQ, UQMonitor, PerformanceMonitor, BatchProcessor
import logging
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

class ProductionUQService:
    """
    Production-ready UQ service with monitoring, optimization, and error handling.
    """
    
    def __init__(self, 
                 model, 
                 method="auto",
                 baseline_data=None,
                 baseline_uncertainty=None,
                 config=None):
        """
        Initialize production UQ service.
        
        Parameters
        ----------
        model : object or list
            Model(s) to wrap with UQ
        method : str
            UQ method to use
        baseline_data : array-like, optional
            Baseline data for drift detection
        baseline_uncertainty : array-like, optional
            Baseline uncertainty estimates
        config : dict, optional
            Service configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize UQ model
        self.uq = UQ(model, method=method)
        
        # Initialize monitoring
        self.monitor = UQMonitor(
            baseline_data=baseline_data,
            baseline_uncertainty=baseline_uncertainty,
            drift_threshold=self.config.get('drift_threshold', 0.1),
            window_size=self.config.get('window_size', 100)
        )
        
        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor(
            window_size=self.config.get('perf_window_size', 100)
        )
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(
            batch_size=self.config.get('batch_size', 1000),
            n_jobs=self.config.get('n_jobs', 1)
        )
        
        # Service state
        self.is_healthy = True
        self.last_health_check = time.time()
        self.request_count = 0
        self.error_count = 0
        
    def fit(self, *args, **kwargs):
        """Fit the UQ model."""
        try:
            self.uq.fit(*args, **kwargs)
            self.logger.info("Model fitted successfully")
            return self
        except Exception as e:
            self.logger.error(f"Model fitting failed: {e}")
            raise
    
    def predict(self, 
                X, 
                return_uncertainty=True,
                monitor=True,
                batch_size=None):
        """
        Production prediction with monitoring and optimization.
        
        Parameters
        ----------
        X : array-like
            Input data
        return_uncertainty : bool
            Whether to return uncertainty estimates
        monitor : bool
            Whether to monitor predictions
        batch_size : int, optional
            Batch size for large datasets
            
        Returns
        -------
        dict
            Prediction results with metadata
        """
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Handle large datasets
            if batch_size and len(X) > batch_size:
                predictions = self._predict_large_dataset(X, batch_size, return_uncertainty)
            else:
                predictions = self._predict_single(X, return_uncertainty)
            
            # Monitor if requested
            if monitor:
                monitoring_results = self._monitor_predictions(predictions)
            else:
                monitoring_results = None
            
            # Log performance
            inference_time = time.time() - start_time
            self.performance_monitor.log_performance(
                predictions.get('predictions', np.array([])),
                np.zeros(len(X)),  # Placeholder for true values
                inference_time
            )
            
            # Update health status
            self._update_health_status(monitoring_results)
            
            return {
                'predictions': predictions.get('predictions'),
                'uncertainty': predictions.get('uncertainty'),
                'monitoring': monitoring_results,
                'inference_time': inference_time,
                'request_id': self.request_count,
                'status': 'success'
            }
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Prediction failed: {e}")
            return {
                'error': str(e),
                'request_id': self.request_count,
                'status': 'error'
            }
    
    def _predict_single(self, X, return_uncertainty):
        """Single prediction."""
        if return_uncertainty:
            predictions, uncertainty = self.uq.predict(X, return_interval=True)
            return {'predictions': predictions, 'uncertainty': uncertainty}
        else:
            predictions = self.uq.predict(X, return_interval=False)
            return {'predictions': predictions}
    
    def _predict_large_dataset(self, X, batch_size, return_uncertainty):
        """Large dataset prediction with batching."""
        def process_batch(batch):
            if return_uncertainty:
                return self.uq.predict(batch, return_interval=True)
            else:
                return self.uq.predict(batch, return_interval=False)
        
        results = self.batch_processor.process_batches(X, process_batch)
        
        if return_uncertainty:
            # Merge predictions and uncertainties
            predictions = self.batch_processor.merge_results([r[0] for r in results])
            uncertainties = self.batch_processor.merge_results([r[1] for r in results])
            return {'predictions': predictions, 'uncertainty': uncertainties}
        else:
            predictions = self.batch_processor.merge_results(results)
            return {'predictions': predictions}
    
    def _monitor_predictions(self, predictions):
        """Monitor predictions for drift and performance."""
        if 'uncertainty' in predictions:
            return self.monitor.monitor(
                predictions['predictions'], 
                predictions['uncertainty']
            )
        else:
            return self.monitor.monitor(
                predictions['predictions'], 
                None
            )
    
    def _update_health_status(self, monitoring_results):
        """Update service health status."""
        if monitoring_results:
            # Check for drift alerts
            if monitoring_results['alerts']:
                for alert in monitoring_results['alerts']:
                    if alert['severity'] == 'high':
                        self.is_healthy = False
                        self.logger.warning(f"High severity alert: {alert['message']}")
            
            # Check drift score
            if monitoring_results['drift_score'] > self.config.get('drift_threshold', 0.1):
                self.is_healthy = False
                self.logger.warning(f"Drift detected: {monitoring_results['drift_score']:.3f}")
        
        # Check error rate
        error_rate = self.error_count / max(self.request_count, 1)
        if error_rate > self.config.get('max_error_rate', 0.05):
            self.is_healthy = False
            self.logger.warning(f"High error rate: {error_rate:.3f}")
        
        self.last_health_check = time.time()
    
    def get_health_status(self):
        """Get comprehensive health status."""
        monitor_summary = self.monitor.get_summary()
        perf_summary = self.performance_monitor.get_performance_summary()
        
        return {
            'is_healthy': self.is_healthy,
            'last_health_check': self.last_health_check,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'monitoring': monitor_summary,
            'performance': perf_summary,
            'uptime': time.time() - self.last_health_check
        }
    
    def get_model_info(self):
        """Get model information."""
        return self.uq.get_info()
    
    def benchmark(self, X, methods=None):
        """Benchmark service performance."""
        from ueq.utils.performance import PerformanceProfiler
        
        profiler = PerformanceProfiler()
        
        if methods is None:
            methods = {
                'predict': lambda x: self.predict(x, return_uncertainty=False, monitor=False),
                'predict_with_uncertainty': lambda x: self.predict(x, return_uncertainty=True, monitor=False)
            }
        
        return profiler.benchmark_methods(methods, X)
```

### 2. Configuration Management

```python
import yaml
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class UQServiceConfig:
    """Configuration for UQ service."""
    
    # Model configuration
    method: str = "auto"
    model_params: Dict[str, Any] = None
    
    # Monitoring configuration
    drift_threshold: float = 0.1
    window_size: int = 100
    perf_window_size: int = 100
    
    # Performance configuration
    batch_size: int = 1000
    n_jobs: int = 1
    max_error_rate: float = 0.05
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: str = None
    
    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, config_path: str):
        """Save configuration to YAML file."""
        with open(config_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

# Example configuration file (config.yaml)
config_example = """
method: "bootstrap"
model_params:
  n_models: 100
  random_state: 42

drift_threshold: 0.1
window_size: 100
perf_window_size: 100

batch_size: 1000
n_jobs: 4
max_error_rate: 0.05

log_level: "INFO"
log_file: "ueq_service.log"
"""
```

## Model Monitoring

### 1. Real-time Drift Detection

```python
class DriftDetector:
    """Advanced drift detection for UQ models."""
    
    def __init__(self, 
                 baseline_data=None,
                 baseline_uncertainty=None,
                 methods=['statistical', 'ks_test', 'wasserstein']):
        self.baseline_data = baseline_data
        self.baseline_uncertainty = baseline_uncertainty
        self.methods = methods
        self.drift_history = []
        
    def detect_drift(self, current_data, current_uncertainty):
        """Detect drift using multiple methods."""
        results = {}
        
        for method in self.methods:
            if method == 'statistical':
                results[method] = self._statistical_drift(current_uncertainty)
            elif method == 'ks_test':
                results[method] = self._ks_test_drift(current_uncertainty)
            elif method == 'wasserstein':
                results[method] = self._wasserstein_drift(current_uncertainty)
        
        # Combine results
        overall_drift = self._combine_drift_results(results)
        self.drift_history.append(overall_drift)
        
        return {
            'overall_drift': overall_drift,
            'method_results': results,
            'drift_trend': self._analyze_drift_trend()
        }
    
    def _statistical_drift(self, current_uncertainty):
        """Statistical drift detection."""
        from ueq.utils.monitoring import detect_uncertainty_drift
        return detect_uncertainty_drift(
            self.baseline_uncertainty, 
            current_uncertainty, 
            method='statistical'
        )
    
    def _ks_test_drift(self, current_uncertainty):
        """Kolmogorov-Smirnov test for drift."""
        from scipy import stats
        
        if isinstance(current_uncertainty, list):
            current_widths = [u[1] - u[0] for u in current_uncertainty]
            baseline_widths = [u[1] - u[0] for u in self.baseline_uncertainty]
        else:
            current_widths = current_uncertainty
            baseline_widths = self.baseline_uncertainty
        
        statistic, p_value = stats.ks_2samp(baseline_widths, current_widths)
        
        return {
            'drift_detected': p_value < 0.05,
            'statistic': statistic,
            'p_value': p_value,
            'method': 'ks_test'
        }
    
    def _wasserstein_drift(self, current_uncertainty):
        """Wasserstein distance for drift detection."""
        from scipy.stats import wasserstein_distance
        
        if isinstance(current_uncertainty, list):
            current_widths = [u[1] - u[0] for u in current_uncertainty]
            baseline_widths = [u[1] - u[0] for u in self.baseline_uncertainty]
        else:
            current_widths = current_uncertainty
            baseline_widths = self.baseline_uncertainty
        
        distance = wasserstein_distance(baseline_widths, current_widths)
        
        return {
            'drift_detected': distance > 0.1,  # Threshold
            'distance': distance,
            'method': 'wasserstein'
        }
    
    def _combine_drift_results(self, results):
        """Combine multiple drift detection results."""
        drift_scores = []
        for method, result in results.items():
            if 'drift_score' in result:
                drift_scores.append(result['drift_score'])
            elif 'statistic' in result:
                drift_scores.append(result['statistic'])
            elif 'distance' in result:
                drift_scores.append(result['distance'])
        
        return np.mean(drift_scores) if drift_scores else 0.0
    
    def _analyze_drift_trend(self):
        """Analyze drift trend over time."""
        if len(self.drift_history) < 3:
            return 'insufficient_data'
        
        recent_drift = np.mean(self.drift_history[-3:])
        overall_drift = np.mean(self.drift_history)
        
        if recent_drift > overall_drift * 1.2:
            return 'increasing'
        elif recent_drift < overall_drift * 0.8:
            return 'decreasing'
        else:
            return 'stable'
```

### 2. Performance Monitoring

```python
class PerformanceTracker:
    """Comprehensive performance tracking for UQ services."""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.metrics_history = []
        self.alert_thresholds = {
            'latency_p95': 1.0,  # seconds
            'error_rate': 0.05,
            'memory_usage': 0.8,  # fraction
            'cpu_usage': 0.8      # fraction
        }
    
    def track_request(self, 
                     predictions, 
                     true_values, 
                     inference_time, 
                     memory_usage=None,
                     cpu_usage=None):
        """Track a single request."""
        import psutil
        
        # Compute metrics
        mse = np.mean((predictions - true_values) ** 2)
        mae = np.mean(np.abs(predictions - true_values))
        
        # Get system metrics if not provided
        if memory_usage is None:
            memory_usage = psutil.virtual_memory().percent / 100
        if cpu_usage is None:
            cpu_usage = psutil.cpu_percent() / 100
        
        metrics = {
            'timestamp': time.time(),
            'mse': mse,
            'mae': mae,
            'inference_time': inference_time,
            'memory_usage': memory_usage,
            'cpu_usage': cpu_usage,
            'throughput': 1.0 / inference_time if inference_time > 0 else 0
        }
        
        self.metrics_history.append(metrics)
        
        # Keep only recent history
        if len(self.metrics_history) > self.window_size:
            self.metrics_history = self.metrics_history[-self.window_size:]
        
        # Check for alerts
        alerts = self._check_alerts(metrics)
        
        return {
            'metrics': metrics,
            'alerts': alerts
        }
    
    def _check_alerts(self, metrics):
        """Check for performance alerts."""
        alerts = []
        
        if metrics['inference_time'] > self.alert_thresholds['latency_p95']:
            alerts.append({
                'type': 'high_latency',
                'severity': 'warning',
                'message': f"High latency: {metrics['inference_time']:.3f}s"
            })
        
        if metrics['memory_usage'] > self.alert_thresholds['memory_usage']:
            alerts.append({
                'type': 'high_memory',
                'severity': 'critical',
                'message': f"High memory usage: {metrics['memory_usage']:.1%}"
            })
        
        if metrics['cpu_usage'] > self.alert_thresholds['cpu_usage']:
            alerts.append({
                'type': 'high_cpu',
                'severity': 'warning',
                'message': f"High CPU usage: {metrics['cpu_usage']:.1%}"
            })
        
        return alerts
    
    def get_performance_summary(self):
        """Get performance summary."""
        if not self.metrics_history:
            return "No performance data available"
        
        recent_metrics = self.metrics_history[-10:]
        
        return {
            'recent_mse': np.mean([m['mse'] for m in recent_metrics]),
            'recent_mae': np.mean([m['mae'] for m in recent_metrics]),
            'avg_latency': np.mean([m['inference_time'] for m in recent_metrics]),
            'p95_latency': np.percentile([m['inference_time'] for m in recent_metrics], 95),
            'avg_throughput': np.mean([m['throughput'] for m in recent_metrics]),
            'avg_memory_usage': np.mean([m['memory_usage'] for m in recent_metrics]),
            'avg_cpu_usage': np.mean([m['cpu_usage'] for m in recent_metrics]),
            'total_requests': len(self.metrics_history)
        }
```

## Performance Optimization

### 1. Batch Processing Strategies

```python
class OptimizedBatchProcessor:
    """Optimized batch processing for UQ models."""
    
    def __init__(self, 
                 model,
                 optimal_batch_size=None,
                 memory_limit_gb=8,
                 n_jobs=1):
        self.model = model
        self.memory_limit_gb = memory_limit_gb
        self.n_jobs = n_jobs
        
        # Find optimal batch size if not provided
        if optimal_batch_size is None:
            self.optimal_batch_size = self._find_optimal_batch_size()
        else:
            self.optimal_batch_size = optimal_batch_size
    
    def _find_optimal_batch_size(self):
        """Find optimal batch size for the model."""
        from ueq.utils.performance import optimize_batch_size
        
        # Create sample data
        sample_data = np.random.randn(1000, 10)
        
        return optimize_batch_size(
            self.model,
            sample_data,
            min_batch=32,
            max_batch=2048,
            step=32,
            target_time=0.5
        )
    
    def process_large_dataset(self, X, return_uncertainty=True):
        """Process large dataset with optimization."""
        n_samples = len(X)
        
        # Determine batch size based on memory
        available_memory = self._get_available_memory()
        memory_based_batch_size = self._calculate_memory_based_batch_size(
            X, available_memory
        )
        
        # Use the smaller of optimal and memory-based batch size
        batch_size = min(self.optimal_batch_size, memory_based_batch_size)
        
        # Process in batches
        results = []
        for i in range(0, n_samples, batch_size):
            batch = X[i:i + batch_size]
            
            if return_uncertainty:
                pred, unc = self.model.predict(batch, return_interval=True)
                results.append((pred, unc))
            else:
                pred = self.model.predict(batch, return_interval=False)
                results.append(pred)
        
        # Merge results
        if return_uncertainty:
            predictions = np.concatenate([r[0] for r in results], axis=0)
            uncertainties = []
            for r in results:
                uncertainties.extend(r[1])
            return predictions, uncertainties
        else:
            return np.concatenate(results, axis=0)
    
    def _get_available_memory(self):
        """Get available system memory."""
        import psutil
        return psutil.virtual_memory().available / (1024**3)  # GB
    
    def _calculate_memory_based_batch_size(self, X, available_memory_gb):
        """Calculate batch size based on available memory."""
        # Estimate memory per sample (rough approximation)
        bytes_per_sample = X.nbytes / len(X)
        
        # Reserve some memory for other operations
        usable_memory = available_memory_gb * 0.7  # 70% of available memory
        
        # Calculate batch size
        max_batch_size = int((usable_memory * 1024**3) / bytes_per_sample)
        
        return min(max_batch_size, 2048)  # Cap at 2048
```

### 2. Caching and Optimization

```python
class UQCache:
    """Intelligent caching for UQ predictions."""
    
    def __init__(self, max_size=1000, ttl=3600):
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
    
    def get(self, key):
        """Get cached result."""
        if key in self.cache:
            # Check if expired
            if time.time() - self.creation_times[key] > self.ttl:
                self._remove(key)
                return None
            
            # Update access time
            self.access_times[key] = time.time()
            return self.cache[key]
        
        return None
    
    def set(self, key, value):
        """Set cached result."""
        # Remove oldest if cache is full
        if len(self.cache) >= self.max_size:
            self._remove_oldest()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
        self.creation_times[key] = time.time()
    
    def _remove(self, key):
        """Remove item from cache."""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            del self.creation_times[key]
    
    def _remove_oldest(self):
        """Remove least recently used item."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), 
                        key=lambda k: self.access_times[k])
        self._remove(oldest_key)
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_times.clear()
        self.creation_times.clear()
    
    def stats(self):
        """Get cache statistics."""
        if not self.cache:
            return {
                'size': 0,
                'hit_rate': 0,
                'avg_age': 0
            }
        
        current_time = time.time()
        ages = [current_time - t for t in self.creation_times.values()]
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': sum(self.access_times.values()) / max(len(self.cache), 1),
            'avg_age': np.mean(ages),
            'oldest_age': max(ages),
            'newest_age': min(ages)
        }
```

## Deployment Patterns

### 1. REST API Service

```python
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Global service instance
uq_service = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    if uq_service is None:
        return jsonify({'status': 'unhealthy', 'error': 'Service not initialized'}), 500
    
    health = uq_service.get_health_status()
    status_code = 200 if health['is_healthy'] else 503
    
    return jsonify(health), status_code

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    try:
        data = request.get_json()
        X = np.array(data['X'])
        return_uncertainty = data.get('return_uncertainty', True)
        monitor = data.get('monitor', True)
        
        result = uq_service.predict(
            X, 
            return_uncertainty=return_uncertainty,
            monitor=monitor
        )
        
        if result['status'] == 'error':
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_large', methods=['POST'])
def predict_large():
    """Large dataset prediction endpoint."""
    try:
        data = request.get_json()
        X = np.array(data['X'])
        batch_size = data.get('batch_size', 1000)
        
        result = uq_service.predict(
            X, 
            return_uncertainty=True,
            monitor=True,
            batch_size=batch_size
        )
        
        if result['status'] == 'error':
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information."""
    if uq_service is None:
        return jsonify({'error': 'Service not initialized'}), 500
    
    info = uq_service.get_model_info()
    return jsonify(info)

@app.route('/benchmark', methods=['POST'])
def benchmark():
    """Benchmark endpoint."""
    try:
        data = request.get_json()
        X = np.array(data['X'])
        
        results = uq_service.benchmark(X)
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def initialize_service(model, config_path=None):
    """Initialize the UQ service."""
    global uq_service
    
    if config_path:
        config = UQServiceConfig.from_yaml(config_path)
    else:
        config = UQServiceConfig()
    
    uq_service = ProductionUQService(
        model=model,
        method=config.method,
        config=config.__dict__
    )
    
    return uq_service

if __name__ == '__main__':
    # Initialize service
    from sklearn.ensemble import RandomForestRegressor
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    service = initialize_service(model)
    
    # Train model (in production, this would be done separately)
    # service.fit(X_train, y_train)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
```

### 2. Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 ueq_user && chown -R ueq_user:ueq_user /app
USER ueq_user

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD ["python", "app.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  ueq-service:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - ueq-service
    restart: unless-stopped
```

### 3. Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ueq-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ueq-service
  template:
    metadata:
      labels:
        app: ueq-service
    spec:
      containers:
      - name: ueq-service
        image: ueq-service:latest
        ports:
        - containerPort: 5000
        env:
        - name: FLASK_ENV
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: ueq-service
spec:
  selector:
    app: ueq-service
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

## Monitoring and Alerting

### 1. Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrics
REQUEST_COUNT = Counter('ueq_requests_total', 'Total requests', ['method', 'status'])
REQUEST_DURATION = Histogram('ueq_request_duration_seconds', 'Request duration')
DRIFT_SCORE = Gauge('ueq_drift_score', 'Current drift score')
ERROR_RATE = Gauge('ueq_error_rate', 'Current error rate')
MEMORY_USAGE = Gauge('ueq_memory_usage_bytes', 'Memory usage')
CPU_USAGE = Gauge('ueq_cpu_usage_percent', 'CPU usage')

class PrometheusMonitor:
    """Prometheus metrics for UQ service."""
    
    def __init__(self, port=8000):
        self.port = port
        start_http_server(port)
    
    def record_request(self, method, status, duration):
        """Record request metrics."""
        REQUEST_COUNT.labels(method=method, status=status).inc()
        REQUEST_DURATION.observe(duration)
    
    def update_drift_score(self, score):
        """Update drift score metric."""
        DRIFT_SCORE.set(score)
    
    def update_error_rate(self, rate):
        """Update error rate metric."""
        ERROR_RATE.set(rate)
    
    def update_system_metrics(self, memory_usage, cpu_usage):
        """Update system metrics."""
        MEMORY_USAGE.set(memory_usage)
        CPU_USAGE.set(cpu_usage)
```

### 2. Alerting Rules

```yaml
# prometheus-alerts.yaml
groups:
- name: ueq-alerts
  rules:
  - alert: UEQHighDrift
    expr: ueq_drift_score > 0.2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High uncertainty drift detected"
      description: "UEQ drift score is {{ $value }}, above threshold of 0.2"
  
  - alert: UEQHighErrorRate
    expr: ueq_error_rate > 0.05
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "UEQ error rate is {{ $value }}, above threshold of 0.05"
  
  - alert: UEQHighLatency
    expr: histogram_quantile(0.95, ueq_request_duration_seconds) > 1.0
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High latency detected"
      description: "95th percentile latency is {{ $value }}s"
  
  - alert: UEQHighMemoryUsage
    expr: ueq_memory_usage_bytes > 8e9  # 8GB
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage detected"
      description: "Memory usage is {{ $value }} bytes"
```

## Scaling Strategies

### 1. Horizontal Scaling

```python
class UQLoadBalancer:
    """Load balancer for UQ services."""
    
    def __init__(self, service_urls, health_check_interval=30):
        self.service_urls = service_urls
        self.health_check_interval = health_check_interval
        self.healthy_services = []
        self.service_metrics = {}
        self.last_health_check = 0
        
    def get_healthy_service(self):
        """Get a healthy service instance."""
        if time.time() - self.last_health_check > self.health_check_interval:
            self._update_healthy_services()
        
        if not self.healthy_services:
            raise Exception("No healthy services available")
        
        # Simple round-robin for now
        return self.healthy_services[0]
    
    def _update_healthy_services(self):
        """Update list of healthy services."""
        import requests
        
        healthy = []
        for url in self.service_urls:
            try:
                response = requests.get(f"{url}/health", timeout=5)
                if response.status_code == 200:
                    healthy.append(url)
            except:
                continue
        
        self.healthy_services = healthy
        self.last_health_check = time.time()
    
    def predict(self, X, **kwargs):
        """Predict using load-balanced service."""
        service_url = self.get_healthy_service()
        
        import requests
        response = requests.post(
            f"{service_url}/predict",
            json={'X': X.tolist(), **kwargs}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Prediction failed: {response.text}")
```

### 2. Caching Layer

```python
class UQCacheLayer:
    """Caching layer for UQ predictions."""
    
    def __init__(self, cache_backend='redis', ttl=3600):
        self.ttl = ttl
        
        if cache_backend == 'redis':
            import redis
            self.cache = redis.Redis(host='localhost', port=6379, db=0)
        else:
            # In-memory cache
            self.cache = {}
    
    def get(self, key):
        """Get cached result."""
        if hasattr(self.cache, 'get'):
            # Redis
            result = self.cache.get(key)
            return json.loads(result) if result else None
        else:
            # In-memory
            return self.cache.get(key)
    
    def set(self, key, value):
        """Set cached result."""
        if hasattr(self.cache, 'set'):
            # Redis
            self.cache.set(key, json.dumps(value), ex=self.ttl)
        else:
            # In-memory
            self.cache[key] = value
    
    def predict_with_cache(self, X, predict_func, **kwargs):
        """Predict with caching."""
        # Create cache key
        cache_key = self._create_cache_key(X, kwargs)
        
        # Try to get from cache
        cached_result = self.get(cache_key)
        if cached_result:
            return cached_result
        
        # Predict and cache
        result = predict_func(X, **kwargs)
        self.set(cache_key, result)
        
        return result
    
    def _create_cache_key(self, X, kwargs):
        """Create cache key from input and parameters."""
        import hashlib
        
        # Hash input data and parameters
        data_str = str(X.tolist()) + str(sorted(kwargs.items()))
        return hashlib.md5(data_str.encode()).hexdigest()
```

## Best Practices

### 1. Model Versioning

```python
class ModelVersionManager:
    """Manage model versions in production."""
    
    def __init__(self, model_storage_path):
        self.model_storage_path = model_storage_path
        self.current_version = None
        self.version_history = []
    
    def save_model(self, uq_model, version, metadata=None):
        """Save model version."""
        import pickle
        import json
        
        version_path = f"{self.model_storage_path}/v{version}"
        os.makedirs(version_path, exist_ok=True)
        
        # Save model
        with open(f"{version_path}/model.pkl", 'wb') as f:
            pickle.dump(uq_model, f)
        
        # Save metadata
        if metadata:
            with open(f"{version_path}/metadata.json", 'w') as f:
                json.dump(metadata, f)
        
        # Update version history
        self.version_history.append({
            'version': version,
            'timestamp': time.time(),
            'metadata': metadata
        })
        
        self.current_version = version
    
    def load_model(self, version=None):
        """Load model version."""
        import pickle
        
        if version is None:
            version = self.current_version
        
        if version is None:
            raise Exception("No model version specified")
        
        version_path = f"{self.model_storage_path}/v{version}"
        model_path = f"{version_path}/model.pkl"
        
        if not os.path.exists(model_path):
            raise Exception(f"Model version {version} not found")
        
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def list_versions(self):
        """List available model versions."""
        return self.version_history
```

### 2. A/B Testing

```python
class UQABTest:
    """A/B testing for UQ models."""
    
    def __init__(self, model_a, model_b, traffic_split=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self.results = {'A': [], 'B': []}
    
    def predict(self, X, user_id=None):
        """Predict using A/B test."""
        # Determine which model to use
        if user_id:
            # Consistent assignment based on user ID
            use_model_a = hash(user_id) % 100 < self.traffic_split * 100
        else:
            # Random assignment
            use_model_a = np.random.random() < self.traffic_split
        
        # Get prediction
        if use_model_a:
            result = self.model_a.predict(X, return_interval=True)
            model_used = 'A'
        else:
            result = self.model_b.predict(X, return_interval=True)
            model_used = 'B'
        
        # Store result for analysis
        self.results[model_used].append({
            'timestamp': time.time(),
            'result': result
        })
        
        return result, model_used
    
    def analyze_results(self):
        """Analyze A/B test results."""
        if not self.results['A'] or not self.results['B']:
            return "Insufficient data for analysis"
        
        # Compare performance metrics
        # This is a simplified analysis - in practice, you'd want more sophisticated metrics
        
        return {
            'model_a_samples': len(self.results['A']),
            'model_b_samples': len(self.results['B']),
            'traffic_split': self.traffic_split
        }
```

### 3. Error Handling and Recovery

```python
class UQErrorHandler:
    """Error handling and recovery for UQ services."""
    
    def __init__(self, max_retries=3, backoff_factor=2):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.error_counts = {}
    
    def handle_prediction_error(self, func, *args, **kwargs):
        """Handle prediction errors with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                # Log error
                error_type = type(e).__name__
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
                
                # Wait before retry
                if attempt < self.max_retries - 1:
                    wait_time = self.backoff_factor ** attempt
                    time.sleep(wait_time)
        
        # All retries failed
        raise Exception(f"Prediction failed after {self.max_retries} attempts: {last_exception}")
    
    def get_error_stats(self):
        """Get error statistics."""
        return self.error_counts
    
    def reset_error_counts(self):
        """Reset error counts."""
        self.error_counts.clear()
```

This production guide provides comprehensive coverage of deploying UEQ in production environments, including monitoring, optimization, scaling, and best practices. The examples are designed to be practical and immediately usable in real-world scenarios.

## 🐛 Issues and Support

If you encounter any problems while deploying UEQ in production:

1. **Check the [API documentation](API.md)** for detailed method signatures
2. **Search existing [issues](https://github.com/kiplangatkorir/ueq/issues)** to see if your problem is already reported
3. **Create a new issue** with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and UEQ version (`ueq.__version__`)
   - Error messages (if any)
   - Production environment details (if relevant)

**We're here to help!** 🚀
