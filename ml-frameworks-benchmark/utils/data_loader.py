"""
Data Loader Utility for ML Frameworks Benchmark
Provides consistent data loading across PyTorch, TensorFlow, and JAX
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


class BenchmarkDataLoader:
    """
    Loads EEG brainwave data for benchmarking ML frameworks.
    Provides data in formats compatible with PyTorch, TensorFlow, and JAX.
    """
    
    def __init__(self, data_path: Optional[str] = None, use_synthetic: bool = True):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to real EEG data (if available)
            use_synthetic: If True, generate synthetic data for testing
        """
        self.data_path = data_path
        self.use_synthetic = use_synthetic
        self.n_samples = 1000
        self.n_features = 64  # EEG channels
        self.n_classes = 6    # backward, forward, left, right, takeoff, landing
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load training and test data.
        
        Returns:
            X_train, X_test, y_train, y_test as numpy arrays
        """
        if self.use_synthetic:
            return self._generate_synthetic_data()
        else:
            return self._load_real_data()
    
    def _generate_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic EEG-like data for testing"""
        np.random.seed(42)
        
        # Generate synthetic features (simulating EEG signals)
        X = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        
        # Generate labels
        y = np.random.randint(0, self.n_classes, size=self.n_samples)
        
        # Split train/test (80/20)
        split_idx = int(0.8 * self.n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Generated synthetic data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        return X_train, X_test, y_train, y_test
    
    def _load_real_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load real EEG data from files"""
        if self.data_path is None:
            raise ValueError("data_path must be provided when use_synthetic=False")
        
        # TODO: Implement real data loading when data becomes available
        raise NotImplementedError("Real data loading will be implemented when data path is available")
    
    def get_class_names(self):
        """Return class label names"""
        return ['backward', 'forward', 'left', 'right', 'takeoff', 'landing']


if __name__ == "__main__":
    # Test the data loader
    loader = BenchmarkDataLoader(use_synthetic=True)
    X_train, X_test, y_train, y_test = loader.load_data()
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Classes: {loader.get_class_names()}")