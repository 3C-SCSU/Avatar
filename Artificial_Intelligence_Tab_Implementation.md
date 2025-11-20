# Artificial Intelligence Tab Implementation 

 

## Documentation 

 

**Author:** Zeyini Mohammad & Youssef Elkhouly  

**Date:** November 2025 

**Project:** Avatar - BCI Application with Machine Learning Integration  

**Ticket:** Add Tab: Artificial Intelligence #451 

 

## Executive Summary 

 

Implementation of an Artificial Intelligence tab that consolidates all A.I./M.L. functionality. Supports training and deployment of Random Forest and Deep Learning models using PyTorch, TensorFlow, and JAX frameworks. Training parameters dynamically appear based on model selection. Result Log displays precision metrics per class. 

 

**Created:** 

- `ArtificialIntelligence.qml` - Main AI/ML tab component 

 

**Modified:** 

- `main.qml` - Added tab to StackLayout 

- `GUI5.py` - Added AI/ML methods and signals 

 

## 1. Implementation Details 

 

### Step 1: UI Components 

 

**Model Selection Buttons:** 

- Random Forest (150x50px, green #6eb109) 

- Deep Learning (150x50px, green #6eb109) 

- Yellow text when selected, white when not 

 

**Framework Selection Buttons:** 

- PyTorch, TensorFlow, JAX (150x50px each) 

- Found in codebase: `prediction-random-forest/JAX` and `prediction-deep-learning/JAX` 

 

**Train/Deploy Toggle:** 

- Train button (active: #6eb109, inactive: #64778d) 

- Deploy button (active: #6eb109, inactive: #64778d) 

- Deploy mode is default 

 

### Step 2: Training Parameters 

 

**Deep Learning Parameters** (8 parameters): 

1. Learning rate (e.g., 0.001) 

2. Batch size (e.g., 32) 

3. Epoch # (e.g., 100) 

4. Optimizer (e.g., Adam, SGD) 

5. Activation fn (e.g., ReLU, Tanh) 

6. Drop out rate (e.g., 0.3) 

7. L1/L2 choice (e.g., L1, L2, None) 

8. Momentum (e.g., 0.9) 

 

**Random Forest Parameters** (9 parameters): 

1. n_estimators (e.g., 100) 

2. max_depth (e.g., 20) 

3. min_samples_split (e.g., 50) 

4. min_samples_leaf (e.g., 50) 

5. max_features (e.g., sqrt, log2, number) 

6. bootstrap (true/false) 

7. criterion (e.g., gini, entropy) 

8. random_state (e.g., 42) 

9. n_jobs (e.g., -1) 

 

**Visibility Logic:** 

```qml 

visible: isTrainMode && (isRandomForestSelected || isDeepLearningSelected) 

``` 

 

### Step 3: Result Log Table 

 

**Structure:** 

- Headers: "Class" and "Precision" 

- Rows: Backward, Forward, Left, Right, Land, Takeoff 

- Content-based sizing (auto-fits all classes) 

- Alternating row colors (#64778d and #5a6d7d) 

 

**Default Precision Values:** 

- Backward: 0.98 

- Forward: 1.00 

- Left: 0.97 

- Right: 0.98 

- Land: 0.96 

- Takeoff: 0.98 

 

### Step 4: Backend Integration (GUI5.py) 

 

**New Signals:** 

```python 

trainingStatusUpdated = Signal(str) 

deploymentStatusUpdated = Signal(str) 

trainingLogUpdated = Signal(str) 

inferenceOutputUpdated = Signal(str) 

precisionMetricsUpdated = Signal(dict) 

``` 

 

**New Properties:** 

```python 

self.deployed_model = None 

self.deployed_model_path = None 

self.deployed_model_type = None # 'pytorch', 'tensorflow', 'pickle' 

self.current_model = "Random Forest" 

self.current_framework = "PyTorch" 

self.precision_metrics = { 

"Backward": "0.98", "Forward": "1.00", "Left": "0.97", 

"Right": "0.98", "Land": "0.96", "Takeoff": "0.98" 

} 

``` 

 

**New Methods:** 

 

1. **`@Slot(str) - selectModel(model_name)`** 

- Stores selected model type 

- Emits log message 

 

2. **`@Slot(str) - selectFramework(framework_name)`** 

- Stores selected framework 

- Emits log message 

 

3. **`@Slot(str) - deployModel(model_path)`** 

- Validates and loads model (.pt, .pth, .h5, .keras, .pkl) 

- Detects model type by extension 

- Handles `file:///` prefix from QML dialogs 

- Supports PyTorch, TensorFlow/Keras, pickle formats 

 

4. **`@Slot(str, str, str, str, str, str) - startTraining(...)`** 

- Collects training parameters 

- Determines model type and framework 

- Calls appropriate training function 

- Emits training logs 

 

5. **`@Slot(result=dict) - getPrecisionMetrics()`** 

- Returns precision metrics dictionary 

- Used by QML to initialize Result Log 

 

6. **`_train_model_with_synthetic_data(...)`** 

- Loads and processes data 

- Normalizes features, encodes labels 

- Splits into train/validation sets 

- Calls framework-specific training 

 

7. **`_train_random_forest_sklearn(...)`** 

- Trains using sklearn 

- Calculates precision metrics per class 

- Saves model, updates precision_metrics 

 

8. **`_train_deep_learning_pytorch(...)`** 

- Trains using PyTorch 

- Supports MLP and CNN architectures 

- Implements training loop with validation 

 

**Training Flow:** 

``` 

QML Input → startTraining() 

↓ 

_train_model_with_synthetic_data() 

↓ 

Data Loading & Preprocessing 

↓ 

Framework Selection 

↓ 

[_train_random_forest_sklearn() | _train_deep_learning_pytorch()] 

↓ 

Training & Evaluation 

↓ 

Precision Calculation → Model Saving 

↓ 

Signals → QML Updates 

``` 

## 2. Data Flow 

 

**Model Selection:** 

``` 

User Click → QML onClicked 

↓ 

Property Update 

↓ 

backend.selectModel(modelName) 

↓ 

Python Slot → Log Update 

``` 

 

**Training:** 

``` 

User Input → Training Parameters 

↓ 

Start Training Button 

↓ 

backend.startTraining(...) 

↓ 

Python: startTraining() → _train_model_with_synthetic_data() 

↓ 

Data Loading → Framework Selection → Training 

↓ 

Precision Calculation → Model Saving 

↓ 

Signals → QML UI Updates 

``` 

 

**Deployment:** 

``` 

User Input → Model File Path 

↓ 

Deploy Model Button 

↓ 

backend.deployModel(modelPath) 

↓ 

File Validation → Model Type Detection 

↓ 

Model Loading → Signal → QML Update 

``` 

 

## 3. Testing 

 

### Functional Tests 

- Model selection updates parameters 

- Framework selection stored correctly 

- Training receives correct parameters 

- Deployment loads models correctly (test .pkl, .pt, .h5 files) 

- Precision metrics update after training 

 

### Integration Tests 

- Signals properly connected 

- QML → Python parameter passing 

- Python → QML signal emission 

- Precision metrics flow 

 

## 4. Commit History 

 

1. **7943a23** - Added PyTorch/TensorFlow framework buttons and dynamic training parameters 

2. **50cefca** - Updated parameters to match codebase (added max_features, criterion, n_jobs) 

3. **61bb299** - Added Framework title and JAX button 

4. **2279c64** - Reorganized layout (parameters right, buttons left) + Result Log table 

5. **8daaf2b** - Updated code style for consistency with codebase 

 

``` 

**Technology Stack:** Qt Quick Controls 6.4, QML 6.5, Python (PySide6), PyTorch, TensorFlow, JAX, scikit-learn 

 
