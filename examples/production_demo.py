#!/usr/bin/env python3
"""
Production Features Demo

This example demonstrates the production-ready features of UEQ v1.0.1:
- Model monitoring and drift detection
- Performance optimization for large datasets
- Batch processing and parallelization
- Production deployment patterns
"""

import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

from ueq import UQ
from ueq.utils.monitoring import UQMonitor, PerformanceMonitor
from ueq.utils.performance import BatchProcessor, PerformanceProfiler

def main():
    print("🏭 Production Features Demo")
    print("=" * 50)
    
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=10, noise=5, random_state=42)
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    # Create and train model
    print("\n🤖 Training UQ Model...")
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    uq = UQ(model, method="bootstrap", n_models=20)
    uq.fit(X_train, y_train)
    
    print("✅ Model trained successfully")
    
    # 1. Model Monitoring and Drift Detection
    print("\n📊 Model Monitoring and Drift Detection")
    print("-" * 40)
    
    # Get baseline uncertainty
    baseline_pred, baseline_uncertainty = uq.predict(X_train[:100], return_interval=True)
    
    # Create monitor
    monitor = UQMonitor(
        baseline_data=X_train[:100],
        baseline_uncertainty=baseline_uncertainty,
        drift_threshold=0.1
    )
    
    # Monitor new data
    new_pred, new_uncertainty = uq.predict(X_test[:50], return_interval=True)
    monitoring_results = monitor.monitor(new_pred, new_uncertainty)
    
    print(f"Drift Score: {monitoring_results['drift_score']:.3f}")
    print(f"Alerts: {len(monitoring_results['alerts'])}")
    
    if monitoring_results['alerts']:
        for alert in monitoring_results['alerts']:
            print(f"  - {alert['type']}: {alert['message']}")
    else:
        print("  ✅ No alerts - model is healthy")
    
    # 2. Performance Optimization
    print("\n⚡ Performance Optimization")
    print("-" * 30)
    
    # Benchmark performance
    profiler = PerformanceProfiler()
    
    def predict_func(data):
        return uq.predict(data, return_interval=False)
    
    def predict_with_uncertainty_func(data):
        return uq.predict(data, return_interval=True)
    
    methods = {
        'predict': predict_func,
        'predict_with_uncertainty': predict_with_uncertainty_func
    }
    
    benchmark_results = profiler.benchmark_methods(methods, X_test[:100])
    
    for method_name, results in benchmark_results.items():
        if results['success']:
            print(f"{method_name}: {results['timing']['execution_time']:.3f}s")
        else:
            print(f"{method_name}: Failed - {results['error']}")
    
    # 3. Batch Processing for Large Datasets
    print("\n📦 Batch Processing for Large Datasets")
    print("-" * 40)
    
    # Create large dataset
    X_large = np.random.randn(5000, 10)
    
    # Process in batches
    batch_processor = BatchProcessor(batch_size=500, n_jobs=1)
    
    def process_batch(batch):
        return uq.predict(batch, return_interval=False)
    
    start_time = time.time()
    batch_results = batch_processor.process_batches(X_large, process_batch)
    batch_time = time.time() - start_time
    
    # Merge results
    final_predictions = batch_processor.merge_results(batch_results)
    
    print(f"Processed {len(X_large)} samples in {batch_time:.3f}s")
    print(f"Batch results: {len(batch_results)} batches")
    print(f"Final predictions shape: {final_predictions.shape}")
    
    # 4. Production Deployment Pattern
    print("\n🚀 Production Deployment Pattern")
    print("-" * 35)
    
    class ProductionUQService:
        """Production UQ service with monitoring and optimization."""
        
        def __init__(self, model, baseline_data=None, baseline_uncertainty=None):
            self.uq = UQ(model)
            self.monitor = UQMonitor(
                baseline_data=baseline_data,
                baseline_uncertainty=baseline_uncertainty
            )
            self.performance_monitor = PerformanceMonitor()
            self.batch_processor = BatchProcessor(batch_size=1000)
            
        def predict(self, X, return_uncertainty=True):
            """Production prediction with monitoring."""
            start_time = time.time()
            
            # Get predictions
            if return_uncertainty:
                predictions, uncertainty = self.uq.predict(X, return_interval=True)
            else:
                predictions = self.uq.predict(X, return_interval=False)
                uncertainty = None
            
            # Monitor
            monitoring_results = self.monitor.monitor(predictions, uncertainty)
            
            # Log performance
            inference_time = time.time() - start_time
            self.performance_monitor.log_performance(
                predictions, np.zeros(len(predictions)), inference_time
            )
            
            return {
                'predictions': predictions,
                'uncertainty': uncertainty,
                'monitoring': monitoring_results,
                'inference_time': inference_time
            }
        
        def predict_large(self, X, batch_size=1000):
            """Large dataset prediction."""
            def process_batch(batch):
                return self.uq.predict(batch, return_interval=False)
            
            results = self.batch_processor.process_batches(X, process_batch)
            return self.batch_processor.merge_results(results)
        
        def get_health_status(self):
            """Get service health status."""
            monitor_summary = self.monitor.get_summary()
            perf_summary = self.performance_monitor.get_performance_summary()
            
            return {
                'monitoring': monitor_summary,
                'performance': perf_summary,
                'status': 'healthy' if monitor_summary.get('status') == 'healthy' else 'warning'
            }
    
    # Create production service
    service = ProductionUQService(
        model=RandomForestRegressor(n_estimators=20, random_state=42),
        baseline_data=X_train[:100],
        baseline_uncertainty=baseline_uncertainty
    )
    
    # Train the service
    service.uq.fit(X_train, y_train)
    
    # Test production predictions
    print("Testing production service...")
    results = service.predict(X_test[:10], return_uncertainty=True)
    
    print(f"Predictions shape: {results['predictions'].shape}")
    print(f"Inference time: {results['inference_time']:.3f}s")
    print(f"Drift score: {results['monitoring']['drift_score']:.3f}")
    
    # Get health status
    health = service.get_health_status()
    print(f"Service status: {health['status']}")
    
    # 5. Real-time Monitoring Dashboard
    print("\n📈 Real-time Monitoring Dashboard")
    print("-" * 35)
    
    # Simulate real-time monitoring
    print("Simulating real-time monitoring...")
    
    for i in range(5):
        # Simulate new data
        new_data = X_test[i*10:(i+1)*10]
        new_results = service.predict(new_data, return_uncertainty=True)
        
        drift_score = new_results['monitoring']['drift_score']
        alerts = new_results['monitoring']['alerts']
        
        status = "🟢" if drift_score < 0.1 else "🟡" if drift_score < 0.2 else "🔴"
        
        print(f"Batch {i+1}: {status} Drift: {drift_score:.3f}, Alerts: {len(alerts)}")
        
        if alerts:
            for alert in alerts:
                print(f"  ⚠️  {alert['message']}")
    
    print("\n🎉 Production features demo completed!")
    print("=" * 50)

def demo_advanced_monitoring():
    """Demonstrate advanced monitoring capabilities."""
    
    print("\n🔍 Advanced Monitoring Demo")
    print("=" * 30)
    
    # Create model with different uncertainty patterns
    model1 = LinearRegression()
    model2 = RandomForestRegressor(n_estimators=10, random_state=42)
    
    uq1 = UQ(model1, method="bootstrap", n_models=10)
    uq2 = UQ(model2, method="bootstrap", n_models=10)
    
    # Generate data
    X, y = make_regression(n_samples=200, n_features=5, noise=5, random_state=42)
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]
    
    # Train models
    uq1.fit(X_train, y_train)
    uq2.fit(X_train, y_train)
    
    # Get baseline uncertainties
    _, baseline_unc1 = uq1.predict(X_train[:50], return_interval=True)
    _, baseline_unc2 = uq2.predict(X_train[:50], return_interval=True)
    
    # Create monitors
    monitor1 = UQMonitor(baseline_uncertainty=baseline_unc1, drift_threshold=0.15)
    monitor2 = UQMonitor(baseline_uncertainty=baseline_unc2, drift_threshold=0.15)
    
    # Monitor both models
    _, unc1 = uq1.predict(X_test[:30], return_interval=True)
    _, unc2 = uq2.predict(X_test[:30], return_interval=True)
    
    results1 = monitor1.monitor(np.zeros(30), unc1)
    results2 = monitor2.monitor(np.zeros(30), unc2)
    
    print(f"Model 1 (Linear): Drift = {results1['drift_score']:.3f}")
    print(f"Model 2 (Random Forest): Drift = {results2['drift_score']:.3f}")
    
    # Compare models
    if results1['drift_score'] < results2['drift_score']:
        print("✅ Model 1 shows more stable uncertainty")
    else:
        print("✅ Model 2 shows more stable uncertainty")

if __name__ == "__main__":
    main()
    demo_advanced_monitoring()
