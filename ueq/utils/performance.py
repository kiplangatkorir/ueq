"""
Performance optimization utilities for large-scale uncertainty quantification.
"""

import numpy as np
import time
from typing import Union, List, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import warnings


class BatchProcessor:
    """
    Efficient batch processing for large datasets.
    
    Parameters
    ----------
    batch_size : int, default=1000
        Size of batches for processing
    n_jobs : int, default=1
        Number of parallel jobs
    backend : str, default='threading'
        Backend for parallel processing ('threading', 'multiprocessing')
    """
    
    def __init__(self, batch_size=1000, n_jobs=1, backend='threading'):
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.backend = backend
        
    def process_batches(self, data, process_func, **kwargs):
        """
        Process data in batches with optional parallelization.
        
        Parameters
        ----------
        data : array-like
            Input data to process
        process_func : callable
            Function to process each batch
        **kwargs
            Additional arguments for process_func
            
        Returns
        -------
        list
            Results from all batches
        """
        n_samples = len(data)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        batches = []
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)
            batches.append(data[start_idx:end_idx])
        
        if self.n_jobs == 1:
            # Sequential processing
            results = []
            for batch in batches:
                result = process_func(batch, **kwargs)
                results.append(result)
        else:
            # Parallel processing
            if self.backend == 'threading':
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    results = list(executor.map(
                        partial(process_func, **kwargs), batches
                    ))
            elif self.backend == 'multiprocessing':
                with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                    results = list(executor.map(
                        partial(process_func, **kwargs), batches
                    ))
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
        
        return results
    
    def merge_results(self, results, merge_func=None):
        """
        Merge results from batch processing.
        
        Parameters
        ----------
        results : list
            Results from batch processing
        merge_func : callable, optional
            Custom merge function
            
        Returns
        -------
        array-like
            Merged results
        """
        if merge_func is not None:
            return merge_func(results)
        
        # Default merge: concatenate arrays
        if isinstance(results[0], np.ndarray):
            return np.concatenate(results, axis=0)
        elif isinstance(results[0], list):
            merged = []
            for result in results:
                merged.extend(result)
            return merged
        else:
            return results


class PerformanceProfiler:
    """
    Profile performance of uncertainty quantification methods.
    
    Parameters
    ----------
    warmup_runs : int, default=3
        Number of warmup runs before timing
    """
    
    def __init__(self, warmup_runs=3):
        self.warmup_runs = warmup_runs
        self.timings = {}
        
    def profile(self, func, *args, **kwargs):
        """
        Profile a function's performance.
        
        Parameters
        ----------
        func : callable
            Function to profile
        *args
            Arguments for the function
        **kwargs
            Keyword arguments for the function
            
        Returns
        -------
        tuple
            (result, timing_info)
        """
        # Warmup runs
        for _ in range(self.warmup_runs):
            try:
                func(*args, **kwargs)
            except:
                pass
        
        # Actual timing
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        timing_info = {
            'execution_time': end_time - start_time,
            'timestamp': start_time
        }
        
        return result, timing_info
    
    def benchmark_methods(self, methods, data, **kwargs):
        """
        Benchmark multiple UQ methods.
        
        Parameters
        ----------
        methods : dict
            Dictionary of method names to functions
        data : array-like
            Test data
        **kwargs
            Additional arguments for methods
            
        Returns
        -------
        dict
            Benchmark results
        """
        results = {}
        
        for name, method in methods.items():
            try:
                result, timing = self.profile(method, data, **kwargs)
                results[name] = {
                    'result': result,
                    'timing': timing,
                    'success': True
                }
            except Exception as e:
                results[name] = {
                    'error': str(e),
                    'success': False
                }
        
        return results


def optimize_batch_size(model, data, min_batch=32, max_batch=2048, 
                       step=32, target_time=1.0):
    """
    Find optimal batch size for a model.
    
    Parameters
    ----------
    model : object
        Model to optimize
    data : array-like
        Sample data
    min_batch : int, default=32
        Minimum batch size to test
    max_batch : int, default=2048
        Maximum batch size to test
    step : int, default=32
        Step size for batch size search
    target_time : float, default=1.0
        Target execution time in seconds
        
    Returns
    -------
    int
        Optimal batch size
    """
    best_batch_size = min_batch
    best_time = float('inf')
    
    for batch_size in range(min_batch, max_batch + 1, step):
        try:
            # Create batch
            batch = data[:batch_size]
            
            # Time the prediction
            start_time = time.time()
            if hasattr(model, 'predict'):
                model.predict(batch)
            else:
                model(batch)
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Check if this is better
            if execution_time < best_time and execution_time <= target_time:
                best_time = execution_time
                best_batch_size = batch_size
                
        except Exception as e:
            warnings.warn(f"Failed to test batch size {batch_size}: {e}")
            continue
    
    return best_batch_size


def memory_efficient_predict(model, data, batch_size=1000, 
                           return_uncertainty=True):
    """
    Memory-efficient prediction for large datasets.
    
    Parameters
    ----------
    model : object
        Model with predict method
    data : array-like
        Input data
    batch_size : int, default=1000
        Batch size for processing
    return_uncertainty : bool, default=True
        Whether to return uncertainty estimates
        
    Returns
    -------
    tuple or array
        Predictions and optionally uncertainty estimates
    """
    n_samples = len(data)
    predictions = []
    uncertainties = []
    
    for i in range(0, n_samples, batch_size):
        batch = data[i:i + batch_size]
        
        if return_uncertainty:
            try:
                pred, unc = model.predict(batch, return_interval=True)
                predictions.append(pred)
                uncertainties.append(unc)
            except:
                # Fallback to single prediction
                pred = model.predict(batch)
                predictions.append(pred)
        else:
            pred = model.predict(batch)
            predictions.append(pred)
    
    # Concatenate results
    final_predictions = np.concatenate(predictions, axis=0)
    
    if return_uncertainty and uncertainties:
        # Handle different uncertainty formats
        if isinstance(uncertainties[0], list):
            final_uncertainties = []
            for unc in uncertainties:
                final_uncertainties.extend(unc)
        else:
            final_uncertainties = np.concatenate(uncertainties, axis=0)
        
        return final_predictions, final_uncertainties
    else:
        return final_predictions


def parallel_ensemble_predict(models, data, n_jobs=None):
    """
    Parallel prediction using multiple models.
    
    Parameters
    ----------
    models : list
        List of models
    data : array-like
        Input data
    n_jobs : int, optional
        Number of parallel jobs
        
    Returns
    -------
    list
        Predictions from all models
    """
    if n_jobs is None:
        n_jobs = min(len(models), mp.cpu_count())
    
    def predict_single(model):
        return model.predict(data)
    
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        predictions = list(executor.map(predict_single, models))
    
    return predictions


class CacheManager:
    """
    Simple cache manager for expensive computations.
    
    Parameters
    ----------
    max_size : int, default=100
        Maximum number of cached items
    """
    
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.cache = {}
        self.access_count = {}
        
    def get(self, key):
        """Get item from cache."""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def set(self, key, value):
        """Set item in cache."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            lru_key = min(self.access_count.keys(), 
                         key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        self.cache[key] = value
        self.access_count[key] = 0
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_count.clear()
    
    def stats(self):
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': sum(self.access_count.values()) / max(len(self.cache), 1)
        }
