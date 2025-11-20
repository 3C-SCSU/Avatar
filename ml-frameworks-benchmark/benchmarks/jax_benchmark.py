"""
JAX Benchmark for EEG Classification
Measures performance of JAX on brainwave data
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
import time
import sys
from pathlib import Path
import numpy as np

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.data_loader import BenchmarkDataLoader


def create_layer_params(rng, n_in, n_out):
    """Initialize layer parameters"""
    w_key, b_key = random.split(rng)
    return {
        'w': random.normal(w_key, (n_in, n_out)) * 0.01,
        'b': jnp.zeros(n_out)
    }


def create_model_params(rng, layer_sizes):
    """Create model parameters for all layers"""
    params = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        rng, layer_rng = random.split(rng)
        params.append(create_layer_params(layer_rng, n_in, n_out))
    return params


def relu(x):
    """ReLU activation"""
    return jnp.maximum(0, x)


def predict(params, x):
    """Forward pass"""
    for i, layer in enumerate(params[:-1]):
        x = relu(jnp.dot(x, layer['w']) + layer['b'])
    
    # Output layer (no activation)
    final_layer = params[-1]
    logits = jnp.dot(x, final_layer['w']) + final_layer['b']
    
    return logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)


def loss(params, x, y):
    """Cross-entropy loss"""
    preds = predict(params, x)
    return -jnp.mean(preds[jnp.arange(len(y)), y])


@jit
def update(params, x, y, lr=0.001):
    """Single gradient descent step"""
    grads = grad(loss)(params, x, y)
    return [
        {'w': p['w'] - lr * g['w'], 'b': p['b'] - lr * g['b']}
        for p, g in zip(params, grads)
    ]


def accuracy(params, x, y):
    """Compute accuracy"""
    preds = predict(params, x)
    predicted_class = jnp.argmax(preds, axis=1)
    return jnp.mean(predicted_class == y)


def run_benchmark(device_type='cpu', epochs=10):
    """Run complete JAX benchmark"""
    print(f"\n{'='*60}")
    print(f"JAX Benchmark - Device: {device_type.upper()}")
    print(f"{'='*60}\n")
    
    # Set device
    if device_type == 'gpu':
        devices = jax.devices('gpu')
        if devices:
            print(f"Using device: GPU ({len(devices)} available)\n")
        else:
            print("GPU requested but not available, using CPU\n")
            device_type = 'cpu'
    else:
        print("Using device: CPU\n")
    
    # Load data
    print("Loading data...")
    loader = BenchmarkDataLoader(use_synthetic=False)
    X_train, X_test, y_train, y_test = loader.load_data()
    
    # Convert to JAX arrays
    X_train = jnp.array(X_train)
    y_train = jnp.array(y_train)
    X_test = jnp.array(X_test)
    y_test = jnp.array(y_test)
    
    # Create model
    print("Creating model...")
    rng = random.PRNGKey(0)
    layer_sizes = [16, 64, 32, 6]  # Input, hidden layers, output
    params = create_model_params(rng, layer_sizes)
    
    # Train
    print(f"\nTraining for {epochs} epochs...")
    total_train_start = time.time()
    training_times = []
    
    batch_size = 32
    n_batches = len(X_train) // batch_size
    
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0
        
        # Shuffle data
        perm = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]
        
        for i in range(n_batches):
            batch_x = X_train_shuffled[i*batch_size:(i+1)*batch_size]
            batch_y = y_train_shuffled[i*batch_size:(i+1)*batch_size]
            
            params = update(params, batch_x, batch_y)
            epoch_loss += loss(params, batch_x, batch_y)
        
        epoch_time = time.time() - start_time
        training_times.append(epoch_time)
        avg_loss = epoch_loss / n_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
    
    total_train_time = time.time() - total_train_start
    
    # Evaluate
    print("\nEvaluating...")
    test_accuracy = accuracy(params, X_test, y_test) * 100
    
    # Measure inference time
    print("Measuring inference time...")
    inference_times = []
    for i in range(len(X_test)):
        start_time = time.time()
        _ = predict(params, X_test[i:i+1])
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
    
    avg_inference_time = np.mean(inference_times)
    
    # Results
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"Total Training Time: {total_train_time:.2f}s")
    print(f"Average Epoch Time: {sum(training_times)/len(training_times):.2f}s")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Average Inference Time: {avg_inference_time*1000:.2f}ms")
    print(f"{'='*60}\n")
    
    return {
        'framework': 'JAX',
        'device': device_type,
        'total_train_time': total_train_time,
        'avg_epoch_time': sum(training_times)/len(training_times),
        'accuracy': float(test_accuracy),
        'avg_inference_time': avg_inference_time * 1000,
        'epochs': epochs
    }


if __name__ == "__main__":
    # Run CPU benchmark
    results_cpu = run_benchmark(device_type='cpu', epochs=5)
    
    # Try GPU if available
    try:
        gpu_devices = jax.devices('gpu')
        if gpu_devices:
            results_gpu = run_benchmark(device_type='gpu', epochs=5)
    except:
        print("\nGPU not available, skipping GPU benchmark")