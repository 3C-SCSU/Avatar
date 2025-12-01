# Gaussian Naive Bayes JAX Implementation

A JAX-based Gaussian Naive Bayes classifier for EEG movement classification.

## Files

- **`gaussiannb_jax.py`**: Core model implementation with save/load utilities
- **`gaussiannb_jax_train.ipynb`**: Full training pipeline notebook
- **`inference_gaussiannb.py`**: Inference script template
- **`README.md`**: This file

## Quick Start

### 1. Update Data Path

In `gaussiannb_jax_train.ipynb`, replace:
```python
BASE_DIR = 'path/to/your/data'
```

with your actual data directory containing movement-labeled subdirectories (backward, forward, landing, left, right, takeoff), each with CSV files.

### 2. Run Training Notebook

```bash
jupyter notebook gaussiannb_jax_train.ipynb
```

Execute all cells to:
- Load EEG data
- Engineer features (variance filtering, statistics, transforms)
- Train GaussianNB with uniform class priors
- Evaluate on test set
- Export model to `gaussiannb_jax_model.pkl`

### 3. Use Trained Model

Load and use the model:
```python
from gaussiannb_jax import load_model

model, state = load_model('gaussiannb_jax_model.pkl')
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

## Model Details

### GaussianNBJAX

**Equations**:

For each class $c$ and feature $j$:
- Mean: $\mu_{cj} = \frac{1}{n_c} \sum_{i \in c} X_{ij}$
- Variance: $\sigma_{cj}^2 = \frac{1}{n_c} \sum_{i \in c} X_{ij}^2 - \mu_{cj}^2 + \lambda$

Log-likelihood for sample $x$:
$$\log P(x|c) = \sum_j \left[ -\frac{(x_j - \mu_{cj})^2}{2\sigma_{cj}^2} - \frac{1}{2}\log(2\pi\sigma_{cj}^2) \right] + \log P(c)$$

### Feature Engineering

1. **Variance Filter**: Removes near-constant raw channels (threshold: $10^{-3}$)
2. **Derived Stats**: Per-row mean, std, absolute mean, energy
3. **Nonlinear Transforms**: Absolute values, log1p of absolute values
4. **Standardization**: Final z-score normalization

Input: 32 raw channels → Output: 55 engineered features

### Training Config

- **VAR_SMOOTHING**: $10^{-9}$ (regularization)
- **UNIFORM_PRIORS**: True (equal class priors)
- **TRAIN_RATIO**: 0.8 (80% train, 20% test)
- **RNG_SEED**: 42

## Performance

- **Accuracy**: ~0.29 (6-class balanced classification)
- **Best Recall**: Landing (0.64)
- **Best Precision**: Forward (0.59)

## Data Format

Expected CSV structure: Tab-delimited numeric values (no header).

Directory layout:
```
path/to/your/data/
  ├── backward/
  │   ├── *.csv
  ├── forward/
  │   ├── *.csv
  ├── landing/
  ├── left/
  ├── right/
  └── takeoff/
```

## API Reference

### `GaussianNBJAX`

```python
class GaussianNBJAX:
    def __init__(self, var_smoothing=1e-9):
        """Initialize model."""
    
    def fit(self, X, y):
        """Fit model (X: n_samples×n_features, y: labels 0..C-1)."""
    
    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
    
    def predict_log_proba(self, X) -> np.ndarray:
        """Predict log-probabilities."""
```

### Utilities

```python
def save_model(path: str, model: GaussianNBJAX, meta: dict):
    """Save model to pickle."""

def load_model(path: str) -> tuple[GaussianNBJAX, dict]:
    """Load model and metadata from pickle."""
```

## Next Steps

- **Improve Accuracy**: Add windowed Welch bandpower (delta/theta/alpha/beta/gamma)
- **Temporal Features**: Aggregate statistics over fixed time windows
- **Class Balancing**: Use weighted priors or resampling
- **Dimensionality Reduction**: Apply PCA before classification

## Requirements

- numpy >= 1.24
- jax >= 0.4.20
- pandas (for data loading)
- pickle (standard library)


