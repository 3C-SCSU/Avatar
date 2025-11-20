"""
Data Loader Utility for ML Frameworks Benchmark
Provides consistent data loading across PyTorch, TensorFlow, and JAX
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import glob


class BenchmarkDataLoader:
    """
    Loads EEG brainwave data for benchmarking ML frameworks.
    Provides data in formats compatible with PyTorch, TensorFlow, and JAX.
    """
    
    def __init__(self, data_path: Optional[str] = None, use_synthetic: bool = False):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to real EEG data (default: ~/Documents/data)
            use_synthetic: If True, generate synthetic data for testing
        """
        if data_path is None:
            data_path = str(Path.home() / "Documents" / "data" / "group03")
        
        self.data_path = data_path
        self.use_synthetic = use_synthetic
        self.n_samples_synthetic = 1000
        self.n_features = 8  # 8 EEG channels from BrainFlow
        self.n_classes = 6    # backward, forward, left, right, takeoff, landing
        self.class_names = ['backwards', 'forward', 'left', 'right', 'takeoff', 'landing']
        
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
        X = np.random.randn(self.n_samples_synthetic, self.n_features * 8).astype(np.float32)
        
        # Generate labels
        y = np.random.randint(0, self.n_classes, size=self.n_samples_synthetic)
        
        # Split train/test (80/20)
        split_idx = int(0.8 * self.n_samples_synthetic)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Generated synthetic data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        return X_train, X_test, y_train, y_test
    
    def _load_real_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load real EEG data from CSV files"""
        print(f"Loading real EEG data from: {self.data_path}")
        
        X_list = []
        y_list = []
        
        # Map class names to labels
        class_to_label = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Iterate through each class folder
        data_root = Path(self.data_path)
        
        # Process first individual for now (can expand later)
        individual_path = data_root / "individual01"
        
        if not individual_path.exists():
            raise ValueError(f"Data path not found: {individual_path}")
        
        for class_name in self.class_names:
            class_path = individual_path / class_name
            
            if not class_path.exists():
                print(f"Warning: {class_name} folder not found, skipping")
                continue
            
            # Get all CSV files for this class
            csv_files = list(class_path.glob("BrainFlow-RAW*.csv"))
            print(f"Loading {len(csv_files)} files for class: {class_name}")
            
            for csv_file in csv_files[:5]:  # Limit to 5 files per class for speed
                try:
                    # Read CSV (tab-separated)
                    df = pd.read_csv(csv_file, sep='\t', header=None)
                    
                    # Extract EEG channels (columns 1-8, as column 0 is timestamp)
                    eeg_data = df.iloc[:, 1:9].values
                    
                    # Take mean of every 250 samples (1 second windows at 250Hz)
                    window_size = 250
                    n_windows = len(eeg_data) // window_size
                    
                    for i in range(n_windows):
                        window = eeg_data[i*window_size:(i+1)*window_size]
                        # Use mean and std as features
                        features = np.concatenate([
                            window.mean(axis=0),
                            window.std(axis=0)
                        ])
                        X_list.append(features)
                        y_list.append(class_to_label[class_name])
                
                except Exception as e:
                    print(f"Error loading {csv_file.name}: {e}")
                    continue
        
        # Convert to arrays
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        
        # Shuffle data
        np.random.seed(42)
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Split train/test (80/20)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Loaded real data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        print(f"Feature shape: {X_train.shape[1]} features (8 channels × 2 stats)")
        return X_train, X_test, y_train, y_test
    
    def get_class_names(self):
        """Return class label names"""
        return self.class_names


if __name__ == "__main__":
    # Test with real data
    print("Testing with REAL EEG data...")
    loader = BenchmarkDataLoader(use_synthetic=False)
    X_train, X_test, y_train, y_test = loader.load_data()
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Classes: {loader.get_class_names()}")
    print(f"Label distribution: {np.bincount(y_train)}")