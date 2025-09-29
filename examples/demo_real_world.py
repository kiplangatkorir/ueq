#!/usr/bin/env python3
"""
🔥 UEQ v1.0.1 Phoenix - Real-World Demo 🔥

This demo shows how to install UEQ from PyPI and solve a real-world problem:
House Price Prediction with Uncertainty Quantification

Problem: Predict house prices with confidence intervals for real estate investment decisions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🔥 UEQ v1.0.1 Phoenix - Real-World Demo 🔥")
    print("=" * 60)
    print("Problem: House Price Prediction with Uncertainty Quantification")
    print("Dataset: California Housing (Real estate data)")
    print("Goal: Predict house prices with confidence intervals")
    print("=" * 60)
    
    # Step 1: Load and prepare data
    print("\n📊 Step 1: Loading California Housing Dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {data.feature_names}")
    print(f"Target: House prices (in $100,000s)")
    print(f"Price range: ${y.min()*100000:.0f} - ${y.max()*100000:.0f}")
    
    # Step 2: Data preprocessing
    print("\n🔧 Step 2: Data Preprocessing...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    
    # Step 3: Demonstrate UEQ Auto-Detection
    print("\n🧠 Step 3: UEQ Auto-Detection in Action...")
    
    # Import UEQ (simulating pip install)
    try:
        from ueq import UQ
        print("✅ UEQ imported successfully!")
        print(f"UEQ version: {UQ.__module__}")
    except ImportError:
        print("❌ UEQ not found. Please install with: pip install ueq")
        return
    
    # Create different models to show auto-detection
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42)
    }
    
    print("\n🔍 Auto-detecting model types and selecting UQ methods...")
    
    uq_models = {}
    for name, model in models.items():
        uq = UQ(model)  # Auto-detection magic!
        info = uq.get_info()
        print(f"  {name}: {info['model_type']} → {info['method']}")
        uq_models[name] = uq
    
    # Step 4: Train models and get predictions
    print("\n🚀 Step 4: Training Models and Getting Predictions...")
    
    results = {}
    
    for name, uq in uq_models.items():
        print(f"\n  Training {name}...")
        
        # Fit model
        uq.fit(X_train_scaled, y_train)
        
        # Get predictions with uncertainty
        predictions, intervals = uq.predict(X_test_scaled, return_interval=True)
        
        # Calculate metrics
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        
        # Calculate coverage
        from ueq import coverage, sharpness
        cov = coverage(y_test, intervals)
        sharp = sharpness(intervals)
        
        results[name] = {
            'predictions': predictions,
            'intervals': intervals,
            'mse': mse,
            'mae': mae,
            'coverage': cov,
            'sharpness': sharp
        }
        
        print(f"    MSE: ${mse*100000:.0f}")
        print(f"    MAE: ${mae*100000:.0f}")
        print(f"    Coverage: {cov:.3f}")
        print(f"    Sharpness: {sharp:.3f}")
    
    # Step 5: Cross-Framework Ensemble Demo
    print("\n🔄 Step 5: Cross-Framework Ensemble Demo...")
    
    # Create PyTorch model
    class HousePriceNet(nn.Module):
        def __init__(self, input_size=8):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.dropout = nn.Dropout(0.2)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            return self.fc3(x)
    
    # Create cross-framework ensemble
    ensemble_models = [
        LinearRegression(),
        RandomForestRegressor(n_estimators=30, random_state=42),
        HousePriceNet
    ]
    
    print("  Creating cross-framework ensemble...")
    ensemble_uq = UQ(ensemble_models)
    ensemble_info = ensemble_uq.get_info()
    
    print(f"  Ensemble type: {ensemble_info['model_type']}")
    print(f"  Method: {ensemble_info['method']}")
    print(f"  Number of models: {ensemble_info['n_models']}")
    print(f"  Model classes: {ensemble_info['model_classes']}")
    
    # Train ensemble
    print("  Training ensemble...")
    ensemble_uq.fit(X_train_scaled, y_train)
    
    # Get ensemble predictions
    ensemble_pred, ensemble_intervals = ensemble_uq.predict(X_test_scaled, return_interval=True)
    
    # Calculate ensemble metrics
    ensemble_mse = np.mean((ensemble_pred - y_test) ** 2)
    ensemble_mae = np.mean(np.abs(ensemble_pred - y_test))
    ensemble_cov = coverage(y_test, ensemble_intervals)
    ensemble_sharp = sharpness(ensemble_intervals)
    
    results['Cross-Framework Ensemble'] = {
        'predictions': ensemble_pred,
        'intervals': ensemble_intervals,
        'mse': ensemble_mse,
        'mae': ensemble_mae,
        'coverage': ensemble_cov,
        'sharpness': ensemble_sharp
    }
    
    print(f"    Ensemble MSE: ${ensemble_mse*100000:.0f}")
    print(f"    Ensemble MAE: ${ensemble_mae*100000:.0f}")
    print(f"    Ensemble Coverage: {ensemble_cov:.3f}")
    print(f"    Ensemble Sharpness: {ensemble_sharp:.3f}")
    
    # Step 6: Production Features Demo
    print("\n🏭 Step 6: Production Features Demo...")
    
    # Model monitoring
    from ueq import UQMonitor
    
    # Use Linear Regression for monitoring demo
    monitor_uq = UQ(LinearRegression())
    monitor_uq.fit(X_train_scaled, y_train)
    
    # Get baseline uncertainty
    baseline_pred, baseline_unc = monitor_uq.predict(X_train_scaled[:100], return_interval=True)
    
    # Create monitor
    monitor = UQMonitor(
        baseline_data=X_train_scaled[:100],
        baseline_uncertainty=baseline_unc,
        drift_threshold=0.1
    )
    
    # Monitor new data (simulate new incoming data)
    new_data = X_test_scaled[:50]
    new_pred, new_unc = monitor_uq.predict(new_data, return_interval=True)
    
    # Monitor for drift
    monitoring_results = monitor.monitor(new_pred, new_unc)
    
    print(f"  Drift Score: {monitoring_results['drift_score']:.3f}")
    print(f"  Alerts: {len(monitoring_results['alerts'])}")
    
    if monitoring_results['alerts']:
        for alert in monitoring_results['alerts']:
            print(f"    ⚠️ {alert['type']}: {alert['message']}")
    else:
        print("    ✅ No alerts - model is healthy")
    
    # Performance optimization demo
    print("\n  Performance optimization demo...")
    
    # Large dataset simulation
    X_large = np.random.randn(5000, 8)
    X_large_scaled = scaler.transform(X_large)
    
    # Regular prediction
    import time
    start_time = time.time()
    large_pred = monitor_uq.predict(X_large_scaled, return_interval=False)
    regular_time = time.time() - start_time
    
    # Large dataset prediction
    start_time = time.time()
    large_pred_optimized = monitor_uq.predict_large_dataset(X_large_scaled, batch_size=1000)
    optimized_time = time.time() - start_time
    
    print(f"    Regular prediction time: {regular_time:.3f}s")
    print(f"    Optimized prediction time: {optimized_time:.3f}s")
    print(f"    Speedup: {regular_time/optimized_time:.1f}x")
    
    # Step 7: Results Analysis and Visualization
    print("\n📈 Step 7: Results Analysis...")
    
    # Create comparison table
    print("\n📊 Model Comparison:")
    print("-" * 80)
    print(f"{'Model':<25} {'MSE ($)':<12} {'MAE ($)':<12} {'Coverage':<10} {'Sharpness':<10}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{name:<25} {result['mse']*100000:<12.0f} {result['mae']*100000:<12.0f} "
              f"{result['coverage']:<10.3f} {result['sharpness']:<10.3f}")
    
    # Find best model
    best_mse = min(results.items(), key=lambda x: x[1]['mse'])
    best_coverage = min(results.items(), key=lambda x: abs(x[1]['coverage'] - 0.95))
    
    print(f"\n🏆 Best MSE: {best_mse[0]} (${best_mse[1]['mse']*100000:.0f})")
    print(f"🎯 Best Coverage: {best_coverage[0]} ({best_coverage[1]['coverage']:.3f})")
    
    # Step 8: Real-World Decision Making
    print("\n💼 Step 8: Real-World Decision Making...")
    
    # Use the best model for investment decisions
    best_model_name = best_mse[0]
    best_result = results[best_model_name]
    
    print(f"Using {best_model_name} for investment decisions...")
    
    # Sample predictions for investment analysis
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    
    print("\n🏠 Sample House Price Predictions:")
    print("-" * 60)
    print(f"{'House':<8} {'Predicted':<12} {'Lower Bound':<12} {'Upper Bound':<12} {'Confidence':<12}")
    print("-" * 60)
    
    for i, idx in enumerate(sample_indices):
        pred = best_result['predictions'][idx]
        lower, upper = best_result['intervals'][idx]
        confidence = (upper - lower) / pred * 100
        
        print(f"House {i+1:<3} ${pred*100000:<11.0f} ${lower*100000:<11.0f} "
              f"${upper*100000:<11.0f} {confidence:<11.1f}%")
    
    # Investment recommendations
    print("\n💡 Investment Recommendations:")
    
    # Find houses with good value (low price, high confidence)
    value_scores = []
    for i in range(len(X_test)):
        pred = best_result['predictions'][i]
        lower, upper = best_result['intervals'][i]
        confidence_width = upper - lower
        value_score = pred / confidence_width  # Higher is better
        value_scores.append((i, pred, confidence_width, value_score))
    
    # Sort by value score
    value_scores.sort(key=lambda x: x[3], reverse=True)
    
    print("Top 3 investment opportunities:")
    for i, (idx, pred, conf_width, score) in enumerate(value_scores[:3]):
        print(f"  {i+1}. House {idx}: ${pred*100000:.0f} "
              f"(Confidence: ±${conf_width*100000/2:.0f}, Value Score: {score:.2f})")
    
    # Step 9: Summary and Conclusions
    print("\n🎉 Step 9: Demo Summary and Conclusions...")
    
    print("\n✅ What we demonstrated:")
    print("  1. Easy installation: pip install ueq")
    print("  2. Auto-detection: UEQ automatically selected optimal UQ methods")
    print("  3. Multiple models: Linear, Ridge, Random Forest with uncertainty")
    print("  4. Cross-framework ensemble: Combined sklearn + PyTorch models")
    print("  5. Production features: Monitoring, drift detection, performance optimization")
    print("  6. Real-world application: House price prediction for investment decisions")
    
    print("\n🔥 Phoenix Features in Action:")
    print("  🧠 Auto-detection: Zero configuration, just works!")
    print("  🔄 Cross-framework: Unified uncertainty across different ML frameworks")
    print("  🏭 Production-ready: Monitoring, optimization, scaling")
    print("  ⚡ Performance: Batch processing for large datasets")
    
    print("\n💼 Business Value:")
    print("  • Confidence intervals for risk assessment")
    print("  • Model monitoring for production reliability")
    print("  • Cross-framework ensembles for robust predictions")
    print("  • Performance optimization for scalability")
    
    print("\n🚀 Ready for Production!")
    print("UEQ Phoenix provides enterprise-grade uncertainty quantification")
    print("with zero configuration and production-ready features.")
    
    print("\n" + "=" * 60)
    print("🔥 UEQ v1.0.1 Phoenix Demo Complete! 🔥")
    print("=" * 60)

if __name__ == "__main__":
    main()
