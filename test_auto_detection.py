#!/usr/bin/env python3
"""
Test script for Phase 1 auto-detection functionality.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from ueq import UQ

def test_auto_detection():
    """Test auto-detection of different model types."""
    
    print("🧪 Testing Phase 1 Auto-Detection")
    print("=" * 50)
    
    # Test 1: Scikit-learn regressor (should auto-select bootstrap)
    print("\n1. Testing sklearn regressor...")
    sklearn_model = LinearRegression()
    uq1 = UQ(sklearn_model)  # method="auto" by default
    info1 = uq1.get_info()
    print(f"   Model type: {info1['model_type']}")
    print(f"   Selected method: {info1['method']}")
    print(f"   Model class: {info1['model_class']}")
    assert info1['model_type'] == 'sklearn_regressor'
    assert info1['method'] == 'bootstrap'
    
    # Test 2: Scikit-learn classifier (should auto-select conformal)
    print("\n2. Testing sklearn classifier...")
    sklearn_clf = RandomForestClassifier()
    uq2 = UQ(sklearn_clf)
    info2 = uq2.get_info()
    print(f"   Model type: {info2['model_type']}")
    print(f"   Selected method: {info2['method']}")
    print(f"   Model class: {info2['model_class']}")
    assert info2['model_type'] == 'sklearn_classifier'
    assert info2['method'] == 'conformal'
    
    # Test 3: PyTorch model (should auto-select mc_dropout)
    print("\n3. Testing PyTorch model...")
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(5, 1)
        def forward(self, x):
            return self.fc(x)
    
    pytorch_model = SimpleNet()
    uq3 = UQ(pytorch_model)
    info3 = uq3.get_info()
    print(f"   Model type: {info3['model_type']}")
    print(f"   Selected method: {info3['method']}")
    print(f"   Model class: {info3['model_class']}")
    assert info3['model_type'] == 'pytorch'
    assert info3['method'] == 'mc_dropout'
    
    # Test 4: Constructor function (should auto-select deep_ensemble)
    print("\n4. Testing constructor function...")
    def create_model():
        return SimpleNet()
    
    uq4 = UQ(create_model)
    info4 = uq4.get_info()
    print(f"   Model type: {info4['model_type']}")
    print(f"   Selected method: {info4['method']}")
    print(f"   Model class: {info4['model_class']}")
    assert info4['model_type'] == 'constructor'
    assert info4['method'] == 'deep_ensemble'
    
    # Test 5: No model (should auto-select bayesian_linear)
    print("\n5. Testing no model...")
    uq5 = UQ()
    info5 = uq5.get_info()
    print(f"   Model type: {info5['model_type']}")
    print(f"   Selected method: {info5['method']}")
    print(f"   Model class: {info5['model_class']}")
    assert info5['model_type'] == 'none'
    assert info5['method'] == 'bayesian_linear'
    
    # Test 6: Explicit method override
    print("\n6. Testing explicit method override...")
    uq6 = UQ(sklearn_model, method="conformal")
    info6 = uq6.get_info()
    print(f"   Model type: {info6['model_type']}")
    print(f"   Selected method: {info6['method']}")
    print(f"   Model class: {info6['model_class']}")
    assert info6['model_type'] == 'sklearn_regressor'
    assert info6['method'] == 'conformal'  # Override worked
    
    print("\n✅ All auto-detection tests passed!")
    print("=" * 50)

def test_backward_compatibility():
    """Test that existing code still works."""
    
    print("\n🔄 Testing backward compatibility...")
    
    # Test old API still works
    sklearn_model = LinearRegression()
    uq_old = UQ(sklearn_model, method="bootstrap")  # Explicit method
    info_old = uq_old.get_info()
    print(f"   Old API - Method: {info_old['method']}")
    assert info_old['method'] == 'bootstrap'
    
    # Test that default behavior changed from "bootstrap" to "auto"
    uq_new = UQ(sklearn_model)  # No method specified
    info_new = uq_new.get_info()
    print(f"   New API - Method: {info_new['method']}")
    assert info_new['method'] == 'bootstrap'  # Should still be bootstrap for sklearn regressor
    
    print("✅ Backward compatibility maintained!")

if __name__ == "__main__":
    test_auto_detection()
    test_backward_compatibility()
    print("\n🎉 Phase 1 implementation successful!")

