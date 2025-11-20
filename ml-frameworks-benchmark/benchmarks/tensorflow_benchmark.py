"""
TensorFlow Benchmark for EEG Classification
Measures performance of TensorFlow on brainwave data
"""

import tensorflow as tf
import time
import sys
from pathlib import Path
import numpy as np

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.data_loader import BenchmarkDataLoader


def create_model(n_features=16, n_classes=6):
    """Create TensorFlow CNN model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((n_features, 1), input_shape=(n_features,)),
        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def run_benchmark(device_type='cpu', epochs=10):
    """Run complete TensorFlow benchmark"""
    print(f"\n{'='*60}")
    print(f"TensorFlow Benchmark - Device: {device_type.upper()}")
    print(f"{'='*60}\n")
    
    # Set device
    if device_type == 'cpu':
        tf.config.set_visible_devices([], 'GPU')
        print("Using device: CPU\n")
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Using device: GPU ({len(gpus)} available)\n")
        else:
            print("GPU requested but not available, using CPU\n")
            device_type = 'cpu'
    
    # Load data
    print("Loading data...")
    loader = BenchmarkDataLoader(use_synthetic=False)
    X_train, X_test, y_train, y_test = loader.load_data()
    
    # Create model
    print("Creating model...")
    model = create_model(n_features=16, n_classes=6)
    
    # Train
    print(f"\nTraining for {epochs} epochs...")
    total_train_start = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    total_train_time = time.time() - total_train_start
    avg_epoch_time = total_train_time / epochs
    
    # Evaluate
    print("\nEvaluating...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    test_accuracy *= 100  # Convert to percentage
    
    # Measure inference time
    print("Measuring inference time...")
    inference_times = []
    for i in range(len(X_test)):
        start_time = time.time()
        _ = model.predict(X_test[i:i+1], verbose=0)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
    
    avg_inference_time = np.mean(inference_times)
    
    # Results
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"Total Training Time: {total_train_time:.2f}s")
    print(f"Average Epoch Time: {avg_epoch_time:.2f}s")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Average Inference Time: {avg_inference_time*1000:.2f}ms")
    print(f"{'='*60}\n")
    
    return {
        'framework': 'TensorFlow',
        'device': device_type,
        'total_train_time': total_train_time,
        'avg_epoch_time': avg_epoch_time,
        'accuracy': test_accuracy,
        'avg_inference_time': avg_inference_time * 1000,  # Convert to ms
        'epochs': epochs
    }


if __name__ == "__main__":
    # Run CPU benchmark
    results_cpu = run_benchmark(device_type='cpu', epochs=5)
    
    # Try GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        results_gpu = run_benchmark(device_type='gpu', epochs=5)