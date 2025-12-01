# GaussianNB Naive Bayes - PyTorch Implementation

## Overview

This folder contains a Gaussian Naive Bayes (GaussianNB) classifier implementation using PyTorch for brain wave (EEG) data classification to predict drone commands. This is a custom PyTorch implementation that uses the Gaussian Probability Density Function and Bayes' theorem for classification.

### Files

- **gaussiannb_model.py**: Contains the `GaussianNB` class - a custom `torch.nn.Module` implementation
- **gaussiannb_train.ipynb**: Jupyter notebook for training the model with EEG data
- **gaussiannb_trained.pth**: Trained model file (generated after training)
- **README.md**: This documentation file

## How It Works

### Gaussian Naive Bayes Algorithm

GaussianNB is a probabilistic classifier based on Bayes' theorem with the assumption that features follow a Gaussian (normal) distribution and are independent.

**Key Steps:**

1. **Training (fit method)**:
   - Calculate mean (μ) and variance (σ²) for each feature within each class
   - Calculate class priors P(y) = count(class) / total_samples
   - Store these as non-trainable buffers (no backpropagation needed)

2. **Prediction (forward method)**:
   - For each class, calculate the Gaussian Probability Density Function:
```
     P(x|y) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
```
   - Apply Bayes' theorem:
```
     P(y|x) ∝ P(x|y) * P(y)
```
   - Use log probabilities for numerical stability
   - Return class with highest posterior probability

### PyTorch Implementation Details

- **Custom nn.Module**: Does not use traditional gradient descent or backpropagation
- **Buffers**: Stores μ, σ², and priors as registered buffers (persist with model state)
- **Tensor Operations**: Uses PyTorch operations (torch.mean, torch.var, torch.log) for GPU acceleration
- **Numerical Stability**: Uses log-probabilities to avoid numerical underflow

## Key Features

- Pure PyTorch implementation with no scikit-learn dependencies for inference
- GPU-compatible (can move model to CUDA for faster prediction on large batches)
- Stores learned parameters (means, variances, priors) in model state
- Can save/load trained models using `torch.save()` and `torch.load()`
- Includes `predict_proba()` method for probability estimates

## Usage

### Training

1. **Update data path** in `gaussiannb_train.ipynb`:
```python
   directory_path = "/path/to/your/brainwave_readings/"
```

2. **Run the notebook** to:
   - Load brain wave .txt files
   - Extract labels from filenames (backward, forward, landing, left, right, takeoff)
   - Train the GaussianNB model
   - Evaluate accuracy on test set
   - Save trained model

3. **Brain wave data format**:
   - `.txt` files with CSV format (skip first 4 header lines)
   - Filenames must contain command labels
   - Default: 32 EEG feature columns

### Inference
```python
import torch
from gaussiannb_model import GaussianNB

# Load trained model
checkpoint = torch.load('gaussiannb_trained.pth')
model = GaussianNB(
    num_features=checkpoint['num_features'],
    num_classes=checkpoint['num_classes']
)
model.load_state_dict(checkpoint['model_state_dict'])

# Make predictions
X_new = torch.FloatTensor(your_new_data)  # Shape: (n_samples, n_features)
predictions = model(X_new)  # Returns class indices

# Get probabilities
probabilities = model.predict_proba(X_new)  # Shape: (n_samples, n_classes)
```

## Mathematical Background

### Bayes' Theorem
```
P(y|X) = [P(X|y) * P(y)] / P(X)
```

Where:
- `P(y|X)`: Posterior probability of class y given features X
- `P(X|y)`: Likelihood of features X given class y (Gaussian PDF)
- `P(y)`: Prior probability of class y
- `P(X)`: Evidence (constant for all classes, can be ignored for classification)

### Naive Assumption

Features are conditionally independent given the class:
```
P(X|y) = P(x₁|y) * P(x₂|y) * ... * P(xₙ|y)
```

### Gaussian PDF

For continuous features:
```
P(xᵢ|y) = (1/√(2πσᵢ²)) * exp(-(xᵢ - μᵢ)²/(2σᵢ²))
```

## Performance Notes

- **Advantages**:
  - Fast training (just calculate statistics, no iterative optimization)
  - Fast inference
  - Works well with small datasets
  - Probabilistic predictions
  - No hyperparameters to tune

- **Limitations**:
  - Assumes features are independent (rarely true for EEG data)
  - Assumes Gaussian distribution for features
  - May underperform compared to deep learning on complex patterns

## Integration with Avatar GUI

This model can be integrated into GUI5.py alongside Random Forest and Deep Learning models. See the Avatar wiki page "Inserting a new model" for integration instructions.

## Author

**Muhammad Arsalan** - PyTorch GaussianNB Implementation for Issue #505

## References

- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
- PyTorch Documentation: https://pytorch.org/docs/
- Naive Bayes Classifier: https://en.wikipedia.org/wiki/Naive_Bayes_classifier
- Gaussian Distribution: https://en.wikipedia.org/wiki/Normal_distribution
- Scikit-learn GaussianNB: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html