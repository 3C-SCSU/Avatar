"""
JAX Deep Learning Implementation for Brain-Computer Interface Applications
===========================================================================

Author: Yash Patel (GitHub: Yash272001)
Date: January 2025
Project: Avatar BCI Platform - Neural Interface Control Systems
Repository: https://github.com/3C-SCSU/Avatar

MAIN CONTRIBUTION:
This implementation demonstrates a neural network approach using JAX and Flax 
for EEG-based brain-computer interface control, providing a deep learning 
alternative to the Random Forest implementation.

REFERENCES:
===========

[1] Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., 
    Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., 
    Wanderman-Milne, S., & Zhang, Q. (2018). JAX: composable 
    transformations of Python+NumPy programs (Version 0.3.13). 
    http://github.com/google/jax
    Used: jit, random, value_and_grad

[2] Google Research. (2024). JAX Documentation. 
    https://jax.readthedocs.io/en/latest/
    Used: API reference for all JAX transformations

[3] Babuschkin, I., Baumli, K., Bell, A., Bhupatiraju, S., Bruce, J., 
    Buchlovsky, P., ... & Viola, F. (2020). The DeepMind JAX Ecosystem. 
    http://github.com/deepmind
    Used: Ecosystem context

[4] Heek, J., Levskaya, A., Oliver, A., Ritter, M., Rondepierre, B., 
    Steiner, A., & van Zee, M. (2020). Flax: A neural network library 
    and ecosystem for JAX (Version 0.8.0). http://github.com/google/flax
    Used: nn.Dense, nn.BatchNorm, nn.Dropout, nn.relu

[5] Babuschkin, I., Hennigan, T., Norman, M., et al. (2020). Optax: 
    composable gradient transformation and optimisation, in JAX. 
    http://github.com/deepmind/optax
    Used: optax.adam, optax.exponential_decay, optax.clip_by_global_norm

[6] Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating 
    Deep Network Training by Reducing Internal Covariate Shift. 
    Proceedings of the 32nd International Conference on Machine Learning, 
    448-456.
    Used: nn.BatchNorm implementation in MLPClassifier

[7] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & 
    Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural 
    Networks from Overfitting. Journal of Machine Learning Research, 
    15(1), 1929-1958.
    Used: nn.Dropout implementation in MLPClassifier

[8] Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic 
    Optimization. arXiv preprint arXiv:1412.6980.
    Used: optax.adam optimizer in create_train_state

[9] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). 
    Array programming with NumPy. Nature, 585(7825), 357-362. 
    https://doi.org/10.1038/s41586-020-2649-2
    Used: np.array, np.mean, standardization

[10] McKinney, W. (2010). Data structures for statistical computing in 
     python. Proceedings of the 9th Python in Science Conference, 445, 51-56.
     Used: pd.read_csv, pd.DataFrame

[11] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. 
     Computing in Science & Engineering, 9(3), 90-95.
     Used: plt.subplots, plt.savefig

[12] Sanei, S., & Chambers, J. A. (2013). EEG signal processing. 
     John Wiley & Sons.
     Used: EEG processing context

[13] Wolpaw, J. R., Birbaumer, N., McFarland, D. J., Pfurtscheller, G., 
     & Vaughan, T. M. (2002). Brain-computer interfaces for communication 
     and control. Clinical Neurophysiology, 113(6), 767-791.
     Used: BCI principles

NOTE:
Development tools (GitHub Copilot, Claude AI, ChatGPT) were used as 
assistive coding aids, but all technical content is derived from and 
cited to the original academic sources listed above.
"""



import jax
import jax.numpy as jnp
from jax import random
import flax
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
import pandas as pd
import os
import time
import pickle
from typing import Sequence, Any
from functools import partial
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

print("="*70)
print("JAX DEEP LEARNING FOR NAO EEG CONTROL")
print("="*70)
print(f"JAX version: {jax.__version__}")
print(f"Device: {jax.default_backend().upper()}")
print(f"Author: Yash272001")
print("="*70)

# Configuration
BASE_DATA_PATH = r"C:\Users\yaskk\JAX Random Forest NAO Control\Professor Data\data\data"
OUTPUT_PATH = os.path.join(BASE_DATA_PATH, 'output')
PLOTS_PATH = os.path.join(OUTPUT_PATH, 'plots')

HIDDEN_DIMS = [256, 128, 64, 32]
LEARNING_RATE = 0.001
BATCH_SIZE = 256
N_EPOCHS = 100
DROPOUT_RATE = 0.1
NOISE_AUGMENTATION = 0.01
EARLY_STOPPING_PATIENCE = 15
GRADIENT_CLIP_NORM = 1.0
SEED = 42
DESIRED_ROWS = 160

jax.config.update("jax_enable_x64", False)

# Set matplotlib style with fallback
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('default')

# ==================================================
# DATA PROCESSING
# ==================================================

def normalize_class_label(label):
    """Normalize class labels to standard format"""
    label_lower = label.lower()
    if 'backward' in label_lower:
        return 'backward'
    elif 'forward' in label_lower:
        return 'forward'
    elif 'landing' in label_lower:
        return 'landing'
    elif 'left' in label_lower:
        return 'left'
    elif 'right' in label_lower:
        return 'right'
    elif 'take' in label_lower or 'takeoff' in label_lower:
        return 'takeoff'
    return label

def read_csv_flexible(file_path):
    """Try multiple separators to read CSV"""
    for sep in [',', '\t', r'\s+']:
        try:
            df = pd.read_csv(file_path, sep=sep)
            if not df.empty and df.shape[1] > 1:
                return df
        except:
            continue
    try:
        return pd.read_csv(file_path, header=None)
    except:
        raise ValueError(f"Could not parse CSV: {file_path}")

def load_and_process_data(base_path, desired_rows):
    """Load and process EEG data from CSV files"""
    print("\nLoading EEG data...")
    
    all_samples = []
    all_labels = []
    successful_files = 0
    
    for root, dirs, files in os.walk(base_path):
        if 'output' in root:
            continue
        csv_files = [f for f in files if f.endswith('.csv')]
        if csv_files:
            class_label_raw = os.path.basename(root)
            if class_label_raw.startswith(('group', 'individual', 'Test')):
                continue
            class_label = normalize_class_label(class_label_raw)
            
            for csv_file in csv_files:
                try:
                    file_path = os.path.join(root, csv_file)
                    df = read_csv_flexible(file_path)
                    if df.empty:
                        continue
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) == 0:
                        continue
                    df = df[numeric_cols]
                    
                    if len(df) < desired_rows:
                        padding = pd.DataFrame(0, index=range(desired_rows - len(df)), 
                                             columns=df.columns)
                        df = pd.concat([df, padding], ignore_index=True)
                    elif len(df) > desired_rows:
                        df = df.iloc[:desired_rows]
                    
                    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
                    
                    for idx, row in df.iterrows():
                        all_samples.append(row.values.astype(np.float32))
                        all_labels.append(class_label)
                    
                    successful_files += 1
                    if successful_files % 200 == 0:
                        print(f"  Processed {successful_files} files...")
                except:
                    pass
    
    print(f"Loaded {successful_files} files, {len(all_samples):,} samples")
    
    max_features = max(len(s) for s in all_samples)
    X_list = []
    for sample in all_samples:
        if len(sample) < max_features:
            padded = np.zeros(max_features, dtype=np.float32)
            padded[:len(sample)] = sample
            X_list.append(padded)
        else:
            X_list.append(sample[:max_features])
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(all_labels)
    
    # Standardize features
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X = (X - X_mean) / X_std
    
    class_names = sorted(np.unique(y))
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    y_encoded = np.array([label_to_idx[label] for label in y], dtype=np.int32)
    
    print(f"Dataset shape: X={X.shape}, y={y_encoded.shape}")
    print(f"Classes: {class_names}\n")
    
    return X, y_encoded, class_names, label_to_idx, X_mean, X_std

# ==================================================
# JAX/FLAX NEURAL NETWORK
# ==================================================

class MLPClassifier(nn.Module):
    """Multi-layer perceptron with batch normalization"""
    hidden_dims: Sequence[int]
    n_classes: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(features=dim, name=f'dense_{i}')(x)
            x = nn.BatchNorm(
                use_running_average=not training,
                momentum=0.9,
                epsilon=1e-5,
                name=f'bn_{i}'
            )(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        x = nn.Dense(features=self.n_classes, name='output')(x)
        return x

@flax.struct.dataclass
class TrainStateWithBatchStats(train_state.TrainState):
    """Training state with batch statistics for BatchNorm"""
    batch_stats: Any

def create_train_state(rng, input_dim, hidden_dims, n_classes):
    """Create training state with optimizer"""
    model = MLPClassifier(hidden_dims=hidden_dims, n_classes=n_classes, dropout_rate=DROPOUT_RATE)
    
    # Initialize with training=True to create batch_stats
    variables = model.init(rng, jnp.ones([1, input_dim]), training=True)
    params = variables['params']
    batch_stats = variables['batch_stats']
    
    schedule = optax.exponential_decay(
        init_value=LEARNING_RATE,
        transition_steps=1000,
        decay_rate=0.96,
        staircase=False
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(GRADIENT_CLIP_NORM),
        optax.adam(schedule)
    )
    
    return TrainStateWithBatchStats.create(
        apply_fn=model.apply,
        params=params,
        batch_stats=batch_stats,
        tx=tx
    )

def augment_data(batch_x, rng, noise_level=NOISE_AUGMENTATION):
    """Add Gaussian noise for data augmentation"""
    noise = random.normal(rng, batch_x.shape) * noise_level
    return batch_x + noise

@partial(jax.jit, static_argnums=(4,))
def train_step(state, batch_x, batch_y, rng, n_classes):
    """Training step with batch normalization"""
    aug_rng, dropout_rng = random.split(rng)
    batch_x_aug = augment_data(batch_x, aug_rng)
    
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        logits, new_model_state = state.apply_fn(
            variables,
            batch_x_aug,
            training=True,
            mutable=['batch_stats'],
            rngs={'dropout': dropout_rng}
        )
        one_hot = jax.nn.one_hot(batch_y, n_classes)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
        return loss, (logits, new_model_state)
    
    (loss, (logits, new_model_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    state = state.apply_gradients(
        grads=grads,
        batch_stats=new_model_state['batch_stats']
    )
    
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == batch_y)
    
    return state, loss, accuracy

@jax.jit
def eval_step(state, batch_x, batch_y):
    """Evaluation step"""
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(variables, batch_x, training=False)
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == batch_y)
    return accuracy, predictions

def create_batches(X, y, batch_size, rng):
    """Create shuffled batches (including last partial batch)"""
    n_samples = X.shape[0]
    perm = random.permutation(rng, n_samples)
    X_shuffled = X[perm]
    y_shuffled = y[perm]
    
    # Include last partial batch
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        yield X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx]

def train_model(X_train, y_train, X_val, y_val, n_epochs, batch_size):
    """Train the neural network"""
    # Derive n_classes from data
    n_classes = int(jnp.max(y_train)) + 1
    
    print("\nTraining JAX Deep Learning Model")
    print(f"Architecture: Input({X_train.shape[1]}) -> {HIDDEN_DIMS} -> Output({n_classes})")
    print(f"Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"Batch size: {batch_size}, Epochs: {n_epochs}\n")
    
    rng = random.PRNGKey(SEED)
    rng, init_rng = random.split(rng)
    
    state = create_train_state(
        init_rng,
        X_train.shape[1],
        HIDDEN_DIMS,
        n_classes
    )
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'epochs': []
    }
    
    best_val_acc = 0.0
    best_params = None
    best_batch_stats = None
    no_improve_count = 0
    
    track_interval = 10
    
    for epoch in range(n_epochs):
        rng, epoch_rng = random.split(rng)
        train_loss = 0.0
        train_acc = 0.0
        n_batches = 0
        
        for batch_x, batch_y in create_batches(X_train, y_train, batch_size, epoch_rng):
            rng, step_rng = random.split(rng)
            state, loss, acc = train_step(state, batch_x, batch_y, step_rng, n_classes)
            train_loss += loss
            train_acc += acc
            n_batches += 1
        
        # Cast JAX scalars to Python floats
        train_loss = float(train_loss / n_batches)
        train_acc = float(train_acc / n_batches)
        
        val_acc, _ = eval_step(state, X_val, y_val)
        val_acc = float(val_acc)
        
        if (epoch + 1) % track_interval == 0 or epoch == 0:
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['epochs'].append(epoch + 1)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = state.params
            best_batch_stats = state.batch_stats
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train: {train_acc*100:.2f}% | "
                  f"Val: {val_acc*100:.2f}% | "
                  f"Best: {best_val_acc*100:.2f}%")
        
        if no_improve_count >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            state = state.replace(params=best_params, batch_stats=best_batch_stats)
            break
    
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    
    if best_params is not None:
        state = state.replace(params=best_params, batch_stats=best_batch_stats)
    
    return state, best_val_acc, history

# ==================================================
# VISUALIZATION
# ==================================================

def plot_training_curves(history, output_path):
    """Plot training curves"""
    print("Generating training curves...")
    os.makedirs(output_path, exist_ok=True)
    
    epochs = history['epochs']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    ax1.plot(epochs, history['train_loss'], 'o-', linewidth=2, markersize=4, color='#1f77b4', label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax1.set_title('Training Loss Over Time', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, [a*100 for a in history['train_acc']], 'o-', linewidth=2, markersize=4, 
             color='#2ca02c', label='Training Accuracy')
    ax2.plot(epochs, [a*100 for a in history['val_acc']], 's-', linewidth=2, markersize=4, 
            color='#d62728', label='Validation Accuracy')
    ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Accuracy Over Time', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    curves_path = os.path.join(output_path, 'training_curves_deep_learning.png')
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {curves_path}")

def plot_training_history_comprehensive(history, output_path):
    """Plot comprehensive training history"""
    print("Generating comprehensive training history...")
    os.makedirs(output_path, exist_ok=True)
    
    if not history['train_acc']:
        return
    
    epochs = history['epochs']
    train_loss = history['train_loss']
    train_acc = [s * 100 for s in history['train_acc']]
    val_acc = [s * 100 for s in history['val_acc']] if history['val_acc'] else None
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax_main = fig.add_subplot(gs[0, :])
    ax1 = ax_main
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(epochs, train_loss, 'o-', linewidth=2, markersize=3, 
                     color='#1f77b4', label='Training Loss')
    line2 = ax2.plot(epochs, train_acc, 's-', linewidth=2, markersize=3, 
                     color='#2ca02c', label='Training Accuracy')
    if val_acc:
        line3 = ax2.plot(epochs, val_acc, '^-', linewidth=2, markersize=3, 
                        color='#d62728', label='Validation Accuracy')
    
    ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=13, fontweight='bold', color='#1f77b4')
    ax2.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold', color='#2ca02c')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    ax1.set_title('Training History - NAO Robot EEG Control\n\nLoss and Accuracy', 
                 fontsize=15, fontweight='bold')
    
    lines = line1 + line2 + (line3 if val_acc else [])
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, train_loss, 'o-', linewidth=2, markersize=3, color='#1f77b4')
    ax3.fill_between(epochs, train_loss, alpha=0.3, color='#1f77b4')
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax3.set_title('Training Loss Trend', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, train_acc, 's-', linewidth=2, markersize=3, 
            color='#2ca02c', label='Train')
    ax4.fill_between(epochs, train_acc, alpha=0.3, color='#2ca02c')
    if val_acc:
        ax4.plot(epochs, val_acc, '^-', linewidth=2, markersize=3, 
                color='#d62728', label='Validation')
        ax4.fill_between(epochs, val_acc, alpha=0.3, color='#d62728')
    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    final_train_acc = history['train_acc'][-1] * 100
    final_val_acc = history['val_acc'][-1] * 100 if history['val_acc'] else 0
    final_train_loss = train_loss[-1]
    
    summary_text = f"""
    TRAINING SUMMARY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Total Epochs: {len(epochs)}
    Final Training Accuracy: {final_train_acc:.2f}%
    Final Validation Accuracy: {final_val_acc:.2f}%
    Final Training Loss: {final_train_loss:.4f}
    Best Training Accuracy: {max(history['train_acc'])*100:.2f}%
    Best Validation Accuracy: {max(history['val_acc'])*100 if history['val_acc'] else 0:.2f}%
    """
    
    ax5.text(0.5, 0.5, summary_text, transform=ax5.transAxes,
            fontsize=12, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
            family='monospace')
    
    history_path = os.path.join(output_path, 'training_history_comprehensive_deep_learning.png')
    plt.savefig(history_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {history_path}")

def plot_confusion_matrix_enhanced(y_true, y_pred, class_names, output_path):
    """Plot enhanced confusion matrix"""
    print("Generating confusion matrix...")
    os.makedirs(output_path, exist_ok=True)
    
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    for i in range(n_classes):
        for j in range(n_classes):
            if i == j:
                color = plt.cm.Greens(cm_norm[i, j])
            else:
                color = plt.cm.Reds(cm_norm[i, j] * 0.8)
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, facecolor=color, edgecolor='white', linewidth=2))
            
            text_color = 'white' if (i == j and cm_norm[i, j] > 0.5) or (i != j and cm_norm[i, j] > 0.4) else 'black'
            ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)',
                   ha='center', va='center', fontsize=11, fontweight='bold', color=text_color)
    
    ax.set_xlim(-0.5, n_classes-0.5)
    ax.set_ylim(n_classes-0.5, -0.5)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix - JAX Deep Learning\nNAO Robot EEG Control', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    cm_path = os.path.join(output_path, 'confusion_matrix_deep_learning.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {cm_path}")
    
    return cm

def plot_per_class_metrics(y_true, y_pred, class_names, output_path):
    """Plot per-class metrics"""
    print("Generating per-class metrics...")
    os.makedirs(output_path, exist_ok=True)
    
    n_classes = len(class_names)
    precision = np.zeros(n_classes, dtype=np.float32)
    recall = np.zeros(n_classes, dtype=np.float32)
    f1 = np.zeros(n_classes, dtype=np.float32)
    
    for class_idx in range(n_classes):
        mask_true = y_true == class_idx
        mask_pred = y_pred == class_idx
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(np.logical_not(mask_true) & mask_pred)
        fn = np.sum(mask_true & np.logical_not(mask_pred))
        
        precision[class_idx] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[class_idx] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[class_idx] = 2 * precision[class_idx] * recall[class_idx] / (precision[class_idx] + recall[class_idx]) if (precision[class_idx] + recall[class_idx]) > 0 else 0
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(n_classes)
    width = 0.25
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#5DA5DA', alpha=0.9)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#60BD68', alpha=0.9)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#F17CB0', alpha=0.9)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Classes', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=0, ha='center')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    metrics_path = os.path.join(output_path, 'per_class_metrics_deep_learning.png')
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {metrics_path}")

def print_classification_report(y_true, y_pred, class_names):
    """Print classification metrics"""
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(f"\n{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
    print("-" * 70)
    
    for class_idx, class_name in enumerate(class_names):
        mask_true = y_true == class_idx
        mask_pred = y_pred == class_idx
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(np.logical_not(mask_true) & mask_pred)
        fn = np.sum(mask_true & np.logical_not(mask_pred))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        supp = np.sum(mask_true)
        
        print(f"{class_name:<12} {prec:>11.4f} {rec:>11.4f} {f1:>11.4f} {int(supp):>11}")
    
    accuracy = np.mean(y_true == y_pred)
    print("-" * 70)
    print(f"{'Overall':<12} {'':<12} {'':<12} {'':<12} {len(y_true):>11}")
    print(f"{'Accuracy':<12} {accuracy:>11.4f}")
    print("=" * 70)

# ==================================================
# MAIN
# ==================================================

def main():
    """Main execution"""
    try:
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        os.makedirs(PLOTS_PATH, exist_ok=True)
        
        X, y, class_names, label_to_idx, X_mean, X_std = load_and_process_data(BASE_DATA_PATH, DESIRED_ROWS)
        
        print("Splitting data...")
        n_samples = X.shape[0]
        n_test = int(n_samples * 0.2)
        n_val = int(n_samples * 0.1)
        
        np.random.seed(SEED)
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_samples - n_test - n_val]
        val_idx = indices[n_samples - n_test - n_val:n_samples - n_test]
        test_idx = indices[n_samples - n_test:]
        
        X_train, y_train = jnp.array(X[train_idx]), jnp.array(y[train_idx])
        X_val, y_val = jnp.array(X[val_idx]), jnp.array(y[val_idx])
        X_test, y_test = jnp.array(X[test_idx]), jnp.array(y[test_idx])
        
        print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}\n")
        
        start_time = time.time()
        state, best_acc, history = train_model(
            X_train, y_train, X_val, y_val,
            N_EPOCHS, BATCH_SIZE
        )
        training_time = time.time() - start_time
        
        print("="*70)
        print("EVALUATION")
        print("="*70)
        test_acc, predictions = eval_step(state, X_test, y_test)
        test_acc = float(test_acc)
        
        print(f"Training time: {training_time:.1f}s ({training_time/60:.1f} min)")
        print(f"Test accuracy: {test_acc*100:.2f}%")
        print(f"NAO control success rate: {test_acc*100:.2f}%\n")
        
        y_pred = np.array(predictions)
        y_test_np = np.array(y_test)
        
        plot_training_curves(history, PLOTS_PATH)
        plot_training_history_comprehensive(history, PLOTS_PATH)
        cm = plot_confusion_matrix_enhanced(y_test_np, y_pred, class_names, PLOTS_PATH)
        plot_per_class_metrics(y_test_np, y_pred, class_names, PLOTS_PATH)
        
        print_classification_report(y_test_np, y_pred, class_names)
        
        model_data = {
            'params': state.params,
            'batch_stats': state.batch_stats,
            'class_names': class_names,
            'label_to_idx': label_to_idx,
            'n_features': X.shape[1],
            'hidden_dims': HIDDEN_DIMS,
            'n_classes': int(jnp.max(y_train)) + 1,
            'X_mean': X_mean,
            'X_std': X_std,
            'training_time': training_time,
            'test_accuracy': test_acc,
            'confusion_matrix': cm,
            'training_history': history,
            'author': 'Yash272001',
            'date': '2025-01-19',
            'jax_version': jax.__version__
        }
        
        model_path = os.path.join(OUTPUT_PATH, 'jax_deep_learning_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\nModel saved: {model_path}")
        
        print("\n" + "="*70)
        print("COMPLETE")
        print("="*70)
        print("Generated plots:")
        print("  1. training_curves_deep_learning.png")
        print("  2. training_history_comprehensive_deep_learning.png")
        print("  3. confusion_matrix_deep_learning.png")
        print("  4. per_class_metrics_deep_learning.png")
        print("="*70)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()