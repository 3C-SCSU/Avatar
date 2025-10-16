import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import pandas as pd
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# ==================================================
# CONFIGURATION
# ==================================================
BASE_DATA_PATH = r"C:\Users\yaskk\JAX Random Forest NAO Control\Professor Data\data\data" "Change according to your data path"
OUTPUT_PATH = os.path.join(BASE_DATA_PATH, 'output')

# Hyperparameters
N_TREES = 100
MAX_DEPTH = 20
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 2
BOOTSTRAP_RATIO = 0.8
DESIRED_ROWS = 160

# ==================================================
# DATA PROCESSING
# ==================================================

def normalize_class_label(label):
    """Normalize class labels to standard categories"""
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
    for i, sample in enumerate(all_samples):
        if len(sample) < max_features:
            padded = np.zeros(max_features, dtype=np.float32)
            padded[:len(sample)] = sample
            X_list.append(padded)
        else:
            X_list.append(sample[:max_features])
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(all_labels)
    
    # Encode labels
    class_names = sorted(np.unique(y))
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    y_encoded = np.array([label_to_idx[label] for label in y], dtype=np.int32)
    
    print(f"\n✓ Dataset: X={X.shape}, y={y_encoded.shape}")
    print(f"Classes: {class_names}")
    
    return X, y_encoded, class_names, label_to_idx

# ==================================================
# JAX DECISION TREE IMPLEMENTATION
# ==================================================

class JAXDecisionTree:
    """Decision Tree with Gini Impurity - Pure JAX/NumPy"""
    
    def __init__(self, max_depth=20, min_samples_split=5, min_samples_leaf=2, parent_class=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.parent_class = parent_class
        self.tree = None
        
    def gini_impurity(self, y):
        """Calculate Gini impurity for split criterion"""
        n = len(y)
        if n == 0:
            return 0.0
        
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / n
        return 1.0 - np.sum(probs ** 2)
    
    def get_majority_class(self, y):
        """Get majority class with fallback for edge cases"""
        if len(y) == 0:
            return self.parent_class if self.parent_class is not None else 0
        
        values, counts = np.unique(y, return_counts=True)
        if len(values) == 0:
            return self.parent_class if self.parent_class is not None else 0
        
        return int(values[np.argmax(counts)])
    
    def find_best_split(self, X, y):
        """Find best feature and threshold using Gini impurity"""
        n_samples, n_features = X.shape
        
        if n_samples < self.min_samples_split:
            return None, None, -1
        
        # Random feature selection (sqrt of total features)
        n_features_to_try = max(1, int(np.sqrt(n_features)))
        feature_indices = np.random.choice(n_features, 
                                          min(n_features_to_try, n_features), 
                                          replace=False)
        
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        parent_gini = self.gini_impurity(y)
        
        for feature_idx in feature_indices[:50]:
            feature_values = X[:, feature_idx]
            
            try:
                thresholds = np.percentile(feature_values, [10, 25, 40, 50, 60, 75, 90])
            except:
                continue
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                
                left_gini = self.gini_impurity(y[left_mask])
                right_gini = self.gini_impurity(y[right_mask])
                
                weighted_gini = (n_left/n_samples) * left_gini + (n_right/n_samples) * right_gini
                gain = parent_gini - weighted_gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature_idx)
                    best_threshold = float(threshold)
        
        return best_feature, best_threshold, best_gain
    
    def build_tree(self, X, y, depth=0):
        """Recursively build decision tree"""
        n_samples, n_features = X.shape
        current_majority = self.get_majority_class(y)
        
        # Handle empty nodes
        if n_samples == 0:
            return {'is_leaf': True, 'value': self.parent_class if self.parent_class is not None else 0, 'samples': 0}
        
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1):
            return {'is_leaf': True, 'value': current_majority, 'samples': n_samples}
        
        best_feature, best_threshold, best_gain = self.find_best_split(X, y)
        
        if best_feature is None or best_gain <= 0:
            return {'is_leaf': True, 'value': current_majority, 'samples': n_samples}
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Build subtrees
        old_parent = self.parent_class
        self.parent_class = current_majority
        
        left_tree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        self.parent_class = old_parent
        
        return {
            'is_leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree,
            'samples': n_samples
        }
    
    def fit(self, X, y):
        """Fit the decision tree"""
        self.parent_class = self.get_majority_class(y)
        self.tree = self.build_tree(X, y)
        return self
    
    def predict_sample(self, x, node):
        """Predict single sample by traversing tree"""
        if node['is_leaf']:
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self.predict_sample(x, node['left'])
        else:
            return self.predict_sample(x, node['right'])
    
    def predict(self, X):
        """Predict multiple samples"""
        return np.array([self.predict_sample(X[i], self.tree) for i in range(len(X))])

# ==================================================
# JAX RANDOM FOREST IMPLEMENTATION
# ==================================================

class JAXRandomForest:
    """Random Forest Classifier - Pure JAX/NumPy Implementation"""
    
    def __init__(self, n_trees=100, max_depth=20, min_samples_split=5,
                 min_samples_leaf=2, bootstrap_ratio=0.8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap_ratio = bootstrap_ratio
        self.trees = []
        self.n_classes_ = None
    
    def fit(self, X, y, verbose=True):
        """Fit the random forest with bootstrap sampling"""
        self.n_classes_ = len(np.unique(y))
        n_samples = X.shape[0]
        n_bootstrap = int(n_samples * self.bootstrap_ratio)
        
        if verbose:
            print(f"\n{'='*70}")
            print("Training JAX Random Forest")
            print(f"{'='*70}")
            print(f"Trees: {self.n_trees}")
            print(f"Max depth: {self.max_depth}")
            print(f"Bootstrap samples per tree: {n_bootstrap}")
            print(f"Features: {X.shape[1]}")
            print(f"Classes: {self.n_classes_}")
        
        self.trees = []
        
        for i in range(self.n_trees):
            if verbose and (i + 1) % 10 == 0:
                print(f"Building tree {i + 1}/{self.n_trees}...")
            
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_bootstrap, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Build tree
            tree = JAXDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            
            # Memory management
            if (i + 1) % 10 == 0:
                gc.collect()
        
        if verbose:
            print(f"✓ Random Forest training complete!")
        
        return self
    
    def predict(self, X):
        """Predict using majority voting across all trees"""
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.n_trees))
        
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)
        
        # Majority voting
        final_preds = []
        for i in range(n_samples):
            votes = predictions[i]
            values, counts = np.unique(votes, return_counts=True)
            final_preds.append(int(values[np.argmax(counts)]))
        
        return np.array(final_preds)
    
    def score(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return float(np.mean(predictions == y))

# ==================================================
# MAIN TRAINING PIPELINE
# ==================================================

def main():
    """Main training pipeline for NAO robot EEG control"""
    print("\n" + "=" * 70)
    print("JAX RANDOM FOREST FOR NAO ROBOT CONTROL")
    print("Pure JAX/NumPy Implementation - No scikit-learn")
    print("=" * 70)
    print(f"Author: Yash272001")
    print(f"Date: 2025-10-16")
    print(f"Issue: #320")
    print("=" * 70)
    
    try:
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        
        # Step 1: Load data
        X, y, class_names, label_to_idx = load_and_process_data(BASE_DATA_PATH, DESIRED_ROWS)
        
        # Step 2: Train/test split
        print("\n" + "=" * 70)
        print("Splitting Data")
        print("=" * 70)
        
        n_samples = X.shape[0]
        n_test = int(n_samples * 0.2)
        
        np.random.seed(42)
        indices = np.random.permutation(n_samples)
        
        train_idx = indices[n_test:]
        test_idx = indices[:n_test]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        print(f"Training: {len(X_train)} samples")
        print(f"Testing: {len(X_test)} samples")
        
        # Step 3: Train model
        import time
        start_time = time.time()
        
        rf_model = JAXRandomForest(
            n_trees=N_TREES,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            bootstrap_ratio=BOOTSTRAP_RATIO
        )
        
        rf_model.fit(X_train, y_train, verbose=True)
        
        training_time = time.time() - start_time
        
        # Step 4: Evaluate
        print("\n" + "=" * 70)
        print("Evaluation Results")
        print("=" * 70)
        
        train_acc = rf_model.score(X_train, y_train)
        test_acc = rf_model.score(X_test, y_test)
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"\n🤖 NAO ROBOT CONTROL SUCCESS RATE: {test_acc*100:.2f}%")
        
        # Step 5: Save model
        import pickle
        
        model_data = {
            'model': rf_model,
            'class_names': class_names,
            'label_to_idx': label_to_idx,
            'n_features': X.shape[1],
            'training_time': training_time,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc
        }
        
        model_path = os.path.join(OUTPUT_PATH, 'trained_classifier_jax.pkl')
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