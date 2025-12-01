"""
Inference script for GaussianNB JAX model on EEG movement classification.

Usage:
    python inference_gaussiannb.py <path/to/model.pkl> <path/to/data>
"""

import pickle
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import sys

# Import the model class from gaussiannb_jax module
from gaussiannb_jax import GaussianNBJAX, load_model


def standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply pre-computed standardization to test data."""
    return (X - mean) / (std + 1e-8)


def predict_on_data(model: GaussianNBJAX, X: np.ndarray, 
                     index_to_label: dict) -> tuple:
    """Generate predictions and class probabilities.
    
    Args:
        model: Fitted GaussianNBJAX instance.
        X: Feature matrix (n_samples, n_features).
        index_to_label: Mapping from class index to class name.
    
    Returns:
        Tuple of (predicted_labels, probabilities, class_names)
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    class_names = [index_to_label[i] for i in sorted(index_to_label.keys())]
    
    return y_pred, y_proba, class_names


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inference_gaussiannb.py <path/to/model.pkl> [data_path]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, state = load_model(model_path)
    index_to_label = state['index_to_label']
    
    print(f"Model classes: {[index_to_label[i] for i in sorted(index_to_label.keys())]}")
    print(f"Expected input features: {model.n_features_}")
    print(f"Accuracy on training data: {state.get('metadata', {}).get('accuracy', 'N/A')}")
    
    print("\n--- Ready for inference ---")
    print("Call predict_on_data(model, X_new, index_to_label) to classify new samples.")
