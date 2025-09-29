# UEQ Documentation

Welcome to the comprehensive documentation for **Uncertainty Everywhere (UEQ)** - a unified Python library for Uncertainty Quantification (UQ).

## 📚 Documentation Overview

This documentation provides everything you need to understand, use, and deploy UEQ in production environments.

### 📖 Documentation Structure

| Document | Description | Audience |
|----------|-------------|----------|
| **[API.md](API.md)** | Complete API reference with all classes, methods, and parameters | Developers, API users |
| **[TUTORIAL.md](TUTORIAL.md)** | Step-by-step tutorials and usage examples | Beginners, learners |
| **[PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md)** | Production deployment, monitoring, and scaling | DevOps, ML engineers |
| **[EXAMPLES.md](EXAMPLES.md)** | Comprehensive examples and use cases | All users |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/kiplangatkorir/ueq.git
cd ueq
pip install -e .
```

### Basic Usage

```python
from ueq import UQ
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Generate data
X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)

# Create model
model = LinearRegression()

# Wrap with UEQ (auto-detection)
uq = UQ(model)  # Automatically selects bootstrap method

# Fit and predict
uq.fit(X, y)
predictions, intervals = uq.predict(X[:10], return_interval=True)

print("Predictions:", predictions)
print("Uncertainty intervals:", intervals)
```

## 🎯 Key Features

### **Auto-Detection System**
- Automatically detects model types (sklearn, PyTorch, constructors)
- Selects optimal UQ methods based on model characteristics
- Zero-configuration uncertainty quantification

### **Cross-Framework Support**
- **Scikit-learn**: Bootstrap, Conformal Prediction
- **PyTorch**: MC Dropout, Deep Ensembles
- **Cross-framework**: Combine models from different frameworks
- **Bayesian**: Closed-form Bayesian linear regression

### **Production Features**
- **Model Monitoring**: Real-time drift detection
- **Performance Optimization**: Batch processing, memory efficiency
- **Health Monitoring**: Service health and alerting
- **Scalability**: Horizontal scaling and load balancing

## 📋 UQ Methods

| Method | Framework | Use Case | Coverage | Speed |
|--------|-----------|----------|----------|-------|
| **Bootstrap** | sklearn | General purpose | Ensemble-based | Fast |
| **Conformal** | sklearn | Distribution-free | Guaranteed | Fast |
| **MC Dropout** | PyTorch | Deep learning | Bayesian approximation | Medium |
| **Deep Ensemble** | PyTorch | Best uncertainty | Multiple models | Slow |
| **Bayesian Linear** | None | Linear problems | Closed-form | Very fast |
| **Cross-framework** | Any | Multi-framework | Unified | Variable |

## 🔧 API Overview

### Core Classes

```python
# Main UQ interface
from ueq import UQ

# Monitoring and performance
from ueq import UQMonitor, PerformanceMonitor, BatchProcessor

# Metrics and evaluation
from ueq import coverage, sharpness, expected_calibration_error
```

### Key Methods

```python
# Basic usage
uq = UQ(model, method="auto")
uq.fit(X_train, y_train)
predictions, intervals = uq.predict(X_test, return_interval=True)

# Production features
uq.predict_large_dataset(X_large, batch_size=1000)
monitoring_results = uq.monitor(X_new, y_new)
benchmark_results = uq.benchmark(X_test)

# Model information
info = uq.get_info()
```

## 📊 Uncertainty Quality Metrics

### Coverage
Fraction of true values within prediction intervals.

```python
from ueq import coverage
cov = coverage(y_true, intervals)
print(f"Coverage: {cov:.3f} (target: 0.95)")
```

### Sharpness
Average width of prediction intervals (lower is better).

```python
from ueq import sharpness
sharp = sharpness(intervals)
print(f"Sharpness: {sharp:.3f}")
```

### Calibration
Expected and Maximum Calibration Error.

```python
from ueq import expected_calibration_error, maximum_calibration_error
ece = expected_calibration_error(y_true, intervals)
mce = maximum_calibration_error(y_true, intervals)
print(f"ECE: {ece:.3f}, MCE: {mce:.3f}")
```

## 🏭 Production Deployment

### Service Architecture

```python
from ueq import UQ, UQMonitor, PerformanceMonitor

class ProductionUQService:
    def __init__(self, model):
        self.uq = UQ(model)
        self.monitor = UQMonitor()
        self.performance_monitor = PerformanceMonitor()
    
    def predict(self, X):
        predictions, uncertainty = self.uq.predict(X, return_interval=True)
        monitoring = self.monitor.monitor(predictions, uncertainty)
        return predictions, uncertainty, monitoring
```

### Monitoring and Alerting

```python
# Real-time drift detection
monitor = UQMonitor(baseline_data=X_train, baseline_uncertainty=baseline_unc)
results = monitor.monitor(new_predictions, new_uncertainty)

if results['drift_score'] > 0.1:
    print("⚠️ Drift detected - consider retraining")
```

### Performance Optimization

```python
# Large dataset processing
predictions = uq.predict_large_dataset(X_large, batch_size=1000)

# Batch processing
from ueq import BatchProcessor
processor = BatchProcessor(batch_size=500, n_jobs=4)
results = processor.process_batches(X_large, predict_func)
```

## 🔍 Model Monitoring

### Drift Detection

```python
from ueq import UQMonitor

# Create monitor
monitor = UQMonitor(
    baseline_data=X_train,
    baseline_uncertainty=baseline_unc,
    drift_threshold=0.1
)

# Monitor new data
results = monitor.monitor(predictions, uncertainty)
print(f"Drift score: {results['drift_score']:.3f}")
print(f"Alerts: {len(results['alerts'])}")
```

### Performance Tracking

```python
from ueq import PerformanceMonitor

# Track performance
perf_monitor = PerformanceMonitor()
perf_monitor.log_performance(predictions, true_values, inference_time)

# Get summary
summary = perf_monitor.get_performance_summary()
print(f"Average latency: {summary['avg_latency']:.3f}s")
```

## 📈 Scaling Strategies

### Horizontal Scaling

```python
# Load balancer for multiple UQ services
class UQLoadBalancer:
    def __init__(self, service_urls):
        self.service_urls = service_urls
        self.healthy_services = []
    
    def predict(self, X):
        service_url = self.get_healthy_service()
        return self.call_service(service_url, X)
```

### Caching Layer

```python
# Redis caching for predictions
from ueq import UQCache

cache = UQCache(cache_backend='redis', ttl=3600)
result = cache.predict_with_cache(X, predict_func)
```

## 🎓 Learning Path

### For Beginners
1. Start with **[TUTORIAL.md](TUTORIAL.md)** - Basic usage and concepts
2. Try the examples in **[EXAMPLES.md](EXAMPLES.md)** - Hands-on learning
3. Explore the **[API.md](API.md)** - Understand the interface

### For Developers
1. Read **[API.md](API.md)** - Complete API reference
2. Study **[EXAMPLES.md](EXAMPLES.md)** - Advanced use cases
3. Review **[PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md)** - Deployment patterns

### For ML Engineers
1. Focus on **[PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md)** - Production deployment
2. Use **[EXAMPLES.md](EXAMPLES.md)** - Real-world scenarios
3. Reference **[API.md](API.md)** - Technical details

## 🔗 External Resources

### Related Libraries
- **scikit-learn**: Machine learning models
- **PyTorch**: Deep learning framework
- **scipy**: Statistical functions
- **matplotlib**: Visualization

### Research Papers
- Conformal Prediction: [Vovk et al., 2005](https://link.springer.com/book/10.1007/978-3-319-04013-4)
- Bootstrap Methods: [Efron & Tibshirani, 1993](https://www.crcpress.com/An-Introduction-to-the-Bootstrap/Efron-Tibshirani/p/book/9780412042317)
- MC Dropout: [Gal & Ghahramani, 2016](https://arxiv.org/abs/1506.02142)
- Deep Ensembles: [Lakshminarayanan et al., 2017](https://arxiv.org/abs/1612.01474)

## 🤝 Contributing

We welcome contributions! Please see the main repository for:
- Issue reporting
- Feature requests
- Pull requests
- Code of conduct

## 📄 License

This project is licensed under the MIT License - see the main repository for details.

## 🆘 Support

- **Documentation**: This comprehensive guide
- **Examples**: Run the examples in `examples/` directory
- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Join community discussions

---

**Happy Uncertainty Quantification! 🎉**

*For the latest updates and news, follow the project on GitHub.*
