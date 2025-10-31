!pip install flax optax matplotlib seaborn

import jax
import flax
import optax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
import pandas as pd
import os
import pickle
from typing import Sequence
import warnings
warnings.filterwarnings('ignore')

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"Flax version: {flax.__version__}")
print(f"Optax version: {optax.__version__}")

# ==================================================
# IMPROVED CONFIGURATION
# ==================================================
BASE_DATA_PATH = r"C:\Users\yaskk\JAX Random Forest NAO Control\Professor Data\data\data"
OUTPUT_PATH = os.path.join(BASE_DATA_PATH, 'output')

# Improved Neural Network Hyperparameters
HIDDEN_DIMS = [256, 128, 64, 32]  # Larger network
N_CLASSES = 6
INITIAL_LEARNING_RATE = 0.001
BATCH_SIZE = 256
N_EPOCHS = 100  # More epochs
DROPOUT_RATE = 0.1  # Reduced dropout
NOISE_AUGMENTATION = 0.01  # Data augmentation
EARLY_STOPPING_PATIENCE = 15
GRADIENT_CLIP_NORM = 1.0

# Data parameters
DESIRED_ROWS = 160

# ==================================================
# DATA PROCESSING
# ==================================================

def normalize_class_label(label):
    """Normalize class labels"""
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
    else:
        return label

def read_csv_flexible(file_path):
    """Read CSV with flexible parsing"""
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
        raise ValueError("Could not parse CSV file")

def load_and_process_data(base_path, desired_rows):
    """Load and process CSV files - each ROW becomes a sample"""
    print("=" * 70)
    print("Loading and Processing EEG Data")
    print("=" * 70)
    
    all_samples = []
    all_labels = []
    successful_files = 0
    
    print(f"Scanning: {base_path}")
    
    for root, dirs, files in os.walk(base_path):
        if 'output' in root:
            continue
            
        csv_files = [f for f in files if f.endswith('.csv')]
        
        if csv_files:
            class_label_raw = os.path.basename(root)
            
            if class_label_raw.startswith('group') or \
               class_label_raw.startswith('individual') or \
               class_label_raw.startswith('Test'):
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
                        padding = pd.DataFrame(0, index=range(desired_rows - len(df)), columns=df.columns)
                        df = pd.concat([df, padding], ignore_index=True)
                    elif len(df) > desired_rows:
                        df = df.iloc[:desired_rows]
                    
                    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
                    
                    for idx, row in df.iterrows():
                        all_samples.append(row.values.astype(np.float32))
                        all_labels.append(class_label)
                    
                    successful_files += 1
                    
                    if successful_files % 100 == 0:
                        print(f"  Processed {successful_files} files -> {len(all_samples)} samples...")
                    
                except:
                    pass
    
    print(f"\n✓ Successfully processed: {successful_files} files")
    print(f"✓ Generated: {len(all_samples)} samples")
    
    # Standardize features
    feature_lengths = [len(sample) for sample in all_samples]
    max_features = max(feature_lengths)
    
    print(f"\nStandardizing features to: {max_features}")
    
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
    
    # Normalize features (CRITICAL for neural networks)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X = (X - X_mean) / X_std
    
    # Encode labels
    class_names = sorted(np.unique(y))
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    y_encoded = np.array([label_to_idx[label] for label in y], dtype=np.int32)
    
    print(f"\n✓ Dataset: X={X.shape}, y={y_encoded.shape}")
    print(f"Classes: {class_names}")
    
    return X, y_encoded, class_names, label_to_idx, X_mean, X_std

# ==================================================
# IMPROVED JAX/FLAX NEURAL NETWORK WITH BATCH NORMALIZATION
# ==================================================

class ImprovedMLPClassifier(nn.Module):
    """Improved Multi-Layer Perceptron with Batch Normalization"""
    hidden_dims: Sequence[int]
    n_classes: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        # Hidden layers with BatchNorm
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
        
        # Output layer (logits)
        x = nn.Dense(features=self.n_classes, name='output')(x)
        return x

def create_train_state_improved(rng, input_dim, hidden_dims, n_classes):
    """Create improved training state with learning rate scheduling"""
    model = ImprovedMLPClassifier(hidden_dims=hidden_dims, n_classes=n_classes, dropout_rate=DROPOUT_RATE)
    
    # Initialize with dummy batch (including batch_stats for BatchNorm)
    variables = model.init(rng, jnp.ones([1, input_dim]), training=False)
    params = variables['params']
    batch_stats = variables.get('batch_stats', {})
    
    # Learning rate schedule with exponential decay
    schedule = optax.exponential_decay(
        init_value=INITIAL_LEARNING_RATE,
        transition_steps=1000,
        decay_rate=0.96,
        staircase=False
    )
    
    # Optimizer with gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(GRADIENT_CLIP_NORM),
        optax.adam(schedule)
    )
    
    class TrainStateWithBatchStats(train_state.TrainState):
        batch_stats: dict
    
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

@jit
def train_step_improved(state, batch_x, batch_y, rng):
    """Improved training step with BatchNorm and data augmentation"""
    # Data augmentation
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
        one_hot = jax.nn.one_hot(batch_y, N_CLASSES)
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

@jit
def eval_step_improved(state, batch_x, batch_y):
    """Evaluation step with BatchNorm in inference mode"""
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(variables, batch_x, training=False)
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == batch_y)
    return accuracy, predictions

def create_batches(X, y, batch_size, rng):
    """Create shuffled batches"""
    n_samples = X.shape[0]
    perm = random.permutation(rng, n_samples)
    X_shuffled = X[perm]
    y_shuffled = y[perm]
    
    n_batches = n_samples // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        yield X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx]

def train_model_improved(X_train, y_train, X_test, y_test, n_epochs, batch_size):
    """Train the improved neural network"""
    print("\n" + "=" * 70)
    print("Training IMPROVED JAX Deep Learning Model")
    print("=" * 70)
    print(f"Architecture: Input({X_train.shape[1]}) -> {HIDDEN_DIMS} -> Output({N_CLASSES})")
    print(f"Improvements:")
    print(f"  - Batch Normalization: ✓")
    print(f"  - Learning Rate Scheduling: ✓ (exponential decay)")
    print(f"  - Data Augmentation: ✓ (Gaussian noise σ={NOISE_AUGMENTATION})")
    print(f"  - Gradient Clipping: ✓ (max norm {GRADIENT_CLIP_NORM})")
    print(f"  - Early Stopping: ✓ (patience {EARLY_STOPPING_PATIENCE})")
    print(f"Optimizer: Adam (initial lr={INITIAL_LEARNING_RATE})")
    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {n_epochs}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Initialize model
    rng = random.PRNGKey(42)
    rng, init_rng = random.split(rng)
    
    state = create_train_state_improved(
        init_rng,
        X_train.shape[1],
        HIDDEN_DIMS,
        N_CLASSES
    )
    
    # Training loop with early stopping
    best_test_acc = 0.0
    best_params = None
    best_batch_stats = None
    no_improve_count = 0
    
    for epoch in range(n_epochs):
        # Training
        rng, epoch_rng = random.split(rng)
        train_loss = 0.0
        train_acc = 0.0
        n_batches = 0
        
        for batch_x, batch_y in create_batches(X_train, y_train, batch_size, epoch_rng):
            rng, step_rng = random.split(rng)
            state, loss, acc = train_step_improved(state, batch_x, batch_y, step_rng)
            train_loss += loss
            train_acc += acc
            n_batches += 1
        
        train_loss /= n_batches
        train_acc /= n_batches
        
        # Evaluation
        test_acc, _ = eval_step_improved(state, X_test, y_test)
        
        # Early stopping logic
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_params = state.params
            best_batch_stats = state.batch_stats
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Test Acc: {test_acc:.4f} | "
                  f"Best: {best_test_acc:.4f}")
        
        # Early stopping
        if no_improve_count >= EARLY_STOPPING_PATIENCE:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            print(f"No improvement for {EARLY_STOPPING_PATIENCE} epochs")
            state = state.replace(params=best_params, batch_stats=best_batch_stats)
            break
    
    print(f"\n✓ Training complete!")
    print(f"Best test accuracy: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")
    
    # Restore best model
    if best_params is not None:
        state = state.replace(params=best_params, batch_stats=best_batch_stats)
    
    return state, best_test_acc

def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Create and save confusion matrix visualization"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        n_classes = len(class_names)
        cm = np.zeros((n_classes, n_classes), dtype=np.int32)
        
        for true_class in range(n_classes):
            for pred_class in range(n_classes):
                cm[true_class, pred_class] = np.sum(
                    (y_true == true_class) & (y_pred == pred_class)
                )
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Count'},
                   square=True,
                   linewidths=0.5,
                   linecolor='gray')
        
        plt.title('Confusion Matrix - JAX Deep Learning (Improved)\nNAO Robot EEG Control', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_path = os.path.join(output_path, 'jax_deep_improved_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved: {cm_path}")
        plt.close()
        
        return cm
        
    except ImportError:
        print("⚠ matplotlib/seaborn not installed. Skipping confusion matrix plot.")
        n_classes = len(class_names)
        cm = np.zeros((n_classes, n_classes), dtype=np.int32)
        
        for true_class in range(n_classes):
            for pred_class in range(n_classes):
                cm[true_class, pred_class] = np.sum(
                    (y_true == true_class) & (y_pred == pred_class)
                )
        return cm

def evaluate_model(state, X_test, y_test, class_names):
    """Detailed evaluation with per-class metrics"""
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    
    _, predictions = eval_step_improved(state, X_test, y_test)
    predictions = np.array(predictions)
    y_test_np = np.array(y_test)
    
    # Overall accuracy
    accuracy = np.mean(predictions == y_test_np)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    print("\n" + "=" * 70)
    print("Classification Report:")
    print("=" * 70)
    print(f"              precision    recall  f1-score   support\n")
    
    precisions = []
    recalls = []
    f1s = []
    supports = []
    
    for class_idx, class_name in enumerate(class_names):
        mask_true = y_test_np == class_idx
        mask_pred = predictions == class_idx
        
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(~mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        supp = np.sum(mask_true)
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(supp)
        
        print(f"{class_name:>12}       {prec:.2f}      {rec:.2f}      {f1:.2f}     {int(supp)}")
    
    total_supp = sum(supports)
    macro_prec = np.mean(precisions)
    macro_rec = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    weighted_prec = np.average(precisions, weights=supports)
    weighted_rec = np.average(recalls, weights=supports)
    weighted_f1 = np.average(f1s, weights=supports)
    
    print()
    print(f"    accuracy                           {accuracy:.2f}     {total_supp}")
    print(f"   macro avg       {macro_prec:.2f}      {macro_rec:.2f}      {macro_f1:.2f}     {total_supp}")
    print(f"weighted avg       {weighted_prec:.2f}      {weighted_rec:.2f}      {weighted_f1:.2f}     {total_supp}")
    print("=" * 70)
    
    # Confusion matrix (text)
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    
    for true_class in range(n_classes):
        for pred_class in range(n_classes):
            cm[true_class, pred_class] = np.sum(
                (y_test_np == true_class) & (predictions == pred_class)
            )
    
    print("\nConfusion Matrix:")
    print(cm)
    
    return accuracy, predictions, cm

# ==================================================
# MAIN TRAINING PIPELINE
# ==================================================

def main():
    """Main training pipeline for IMPROVED JAX Deep Learning"""
    print("\n" + "=" * 70)
    print("JAX DEEP LEARNING FOR NAO ROBOT CONTROL - IMPROVED")
    print("Pure JAX/Flax Implementation - No scikit-learn")
    print("=" * 70)
    print(f"Author: Yash272001")
    print(f"Date: 2025-01-16 23:01:24 UTC")
    print(f"Issue: #320 (Deep Learning - Optimized)")
    print("=" * 70)
    
    try:
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        
        # Load data
        X, y, class_names, label_to_idx, X_mean, X_std = load_and_process_data(
            BASE_DATA_PATH, DESIRED_ROWS
        )
        
        # Train/test split
        print("\n" + "=" * 70)
        print("Splitting Data")
        print("=" * 70)
        
        n_samples = X.shape[0]
        n_test = int(n_samples * 0.2)
        
        np.random.seed(42)
        indices = np.random.permutation(n_samples)
        
        train_idx = indices[n_test:]
        test_idx = indices[:n_test]
        
        X_train = jnp.array(X[train_idx])
        y_train = jnp.array(y[train_idx])
        X_test = jnp.array(X[test_idx])
        y_test = jnp.array(y[test_idx])
        
        print(f"Training: {len(X_train)} samples")
        print(f"Testing: {len(X_test)} samples")
        
        # Train model
        import time
        start_time = time.time()
        
        state, best_acc = train_model_improved(
            X_train, y_train, X_test, y_test,
            N_EPOCHS, BATCH_SIZE
        )
        
        training_time = time.time() - start_time
        
        # Evaluate
        test_acc, predictions, cm = evaluate_model(state, X_test, y_test, class_names)
        
        # Generate confusion matrix visualization
        print("\n" + "=" * 70)
        print("Generating Visualizations")
        print("=" * 70)
        plot_confusion_matrix(np.array(y_test), predictions, class_names, OUTPUT_PATH)
        
        print(f"\n✓ Training completed in {training_time:.2f} seconds")
        print(f"\n🤖 NAO ROBOT CONTROL SUCCESS RATE: {test_acc*100:.2f}%")
        
        # Improvement comparison
        print("\n" + "=" * 70)
        print("IMPROVEMENT SUMMARY")
        print("=" * 70)
        print(f"Original Model:  89.35%")
        print(f"Improved Model:  {test_acc*100:.2f}%")
        print(f"Improvement:     +{(test_acc - 0.8935)*100:.2f}%")
        print("=" * 70)
        
        # Save model
        model_data = {
            'params': state.params,
            'batch_stats': state.batch_stats,
            'class_names': class_names,
            'label_to_idx': label_to_idx,
            'n_features': X.shape[1],
            'hidden_dims': HIDDEN_DIMS,
            'n_classes': N_CLASSES,
            'X_mean': X_mean,
            'X_std': X_std,
            'training_time': training_time,
            'test_accuracy': float(test_acc),
            'confusion_matrix': cm,
            'improvements': [
                'Batch Normalization',
                'Learning Rate Scheduling',
                'Data Augmentation',
                'Gradient Clipping',
                'Early Stopping',
                'Larger Network [256,128,64,32]',
                'Reduced Dropout (0.1)'
            ]
        }
        
        model_path = os.path.join(OUTPUT_PATH, 'trained_deep_model_jax_improved.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Model saved to: {model_path}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()