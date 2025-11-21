import sys
import os
import subprocess
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject, Signal, Slot, QProcess, QUrl
from pdf2image import convert_from_path
from djitellopy import Tello
import random
import re
import pandas as pd
import time
import io
import urllib.parse
import contextlib
from collections import defaultdict
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
# from GUI5_ManualDroneControl.cameraview.camera_controller import CameraController
from NAO6.nao_connection import send_command

# Temporary stub for CameraController
from PySide6.QtCore import QObject, Signal, Slot

class CameraController(QObject):
    frameReady = Signal(object)
    
    def __init__(self):
        super().__init__()
        self.tello = None
        self.is_streaming = False
    
    def set_tello_instance(self, tello):
        """Set the Tello drone instance"""
        self.tello = tello
    
    @Slot()
    def startStream(self):
        """Start the camera stream"""
        self.is_streaming = True
        print("Camera stream started")
    
    @Slot()
    def stopStream(self):
        """Stop the camera stream"""
        self.is_streaming = False
        print("Camera stream stopped")
# from Developers.hofCharts import main as hofCharts, ticketsByDev_text NA

from Developers import devCharts

								

# Import BCI connection for brainwave prediction
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'random-forest-prediction')))
    from client.brainflow1 import bciConnection, DataMode
    BCI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: BCI connection not available: {e}")
    BCI_AVAILABLE = False

# Add the parent directory to the Python path for file-shuffler
sys.path.append(str(Path(__file__).resolve().parent / "file-shuffler"))
sys.path.append(str(Path(__file__).resolve().parent / "file-unify-labels"))
sys.path.append(str(Path(__file__).resolve().parent / "file-remove8channel"))
import unifyTXT
import run_file_shuffler
import remove8channel


class TabController(QObject):
    def __init__(self):
        super().__init__()
        self.nao_process = None


class BrainwavesBackend(QObject):
    # Define signals to update QML components
    flightLogUpdated = Signal(list)
    predictionsTableUpdated = Signal(list)
    imagesReady = Signal(list)
    logMessage = Signal(str)
    naoStarted = Signal()
    naoEnded = Signal()
    # AI/ML signals for Training and Deployment Methods 
    trainingStatusUpdated = Signal(str)  # Training status messages
    deploymentStatusUpdated = Signal(str)  # Deployment status messages
    trainingLogUpdated = Signal(str)  # Training accuracy/loss logs
    inferenceOutputUpdated = Signal(str)  # Inference/prediction output
    precisionMetricsUpdated = Signal(dict)  # Precision metrics for each class

    @Slot()
    def startNaoManual(self):
        print("Nao6 Manual session started")
        self.naoStarted.emit("Nao6 Manual session started")

    @Slot()
    def stopNaoManual(self):
        print("Nao6 Manual session ended")
        self.naoEnded.emit("Nao6 Manual session ended")

    @Slot(str, str)
    def connectNao(self, ip="192.168.23.53", port="9559"):
        """Connect to NAO robot with specified IP and Port"""
        try:
            # Log the connection attempt
            self.flight_log.insert(0, f"Attempting to connect to NAO at {ip}:{port}...")
            self.flightLogUpdated.emit(self.flight_log)

            # Send connect command
            if send_command("connect"):
                self.flight_log.insert(0, f"Nao connected successfully at {ip}:{port}.")
            else:
                self.flight_log.insert(0, f"Nao failed to connect at {ip}:{port}.")
        except Exception as e:
            self.flight_log.insert(0, f"Error connecting to NAO: {str(e)}")

        self.flightLogUpdated.emit(self.flight_log)
    
    @Slot()
    def nao_sit_down(self):
        if send_command("sit_down"):
            self.flight_log.insert(0, "Sitting down.")
        else:
            self.flight_log.insert(0, "Nao failed to sit.")
        self.flightLogUpdated.emit(self.flight_log)
    
    @Slot()
    def nao_stand_up(self):
        if send_command("stand_up"):
            self.flight_log.insert(0, "Standing Up.")
        else:
            self.fligt_log.insert(0, "Nao failed to stand up.")
        self.flightLogUpdated.emit(self.flight_log)


            

    @Slot(result=str)
    def getDevList(self):
        exclude = {
            "3C Cloud Computing Club <114175379+3C-SCSU@users.noreply.github.com>",
        }

        proc = subprocess.run(
            ["git", "shortlog", "-sne", "--all"],
            capture_output=True, text=True, encoding="utf-8", errors="ignore"
        )

        if proc.returncode != 0:
            return "No developers found."

        lines = proc.stdout.strip().splitlines()
        filtered_lines = []

        for line in lines:
            # Match the author portion
            match = re.match(r"^\s*\d+\s+(?P<author>.+)$", line)
            if match:
                author = match.group("author").strip()
                if author not in exclude:
                    filtered_lines.append(line)

        return "\n".join(filtered_lines) if filtered_lines else "No developers found."

    @Slot(result=str)
    def getTicketsByDev(self) -> str:

            exclude = {
                "3C Cloud Computing Club <114175379+3C-SCSU@users.noreply.github.com>"
            }

            pretty = "%x1e%an <%ae>%x1f%s%x1f%b"
            try:
                proc = subprocess.run(
                    ["git", "log", "--all", f"--pretty=format:{pretty}"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    check=True
                )
            except subprocess.CalledProcessError:
                return "No tickets found."

            raw = proc.stdout or ""
            commits = raw.split("\x1e")

            jira_re = re.compile(r'\b([A-Za-z]{2,}-\d+)\b')
            hash_re = re.compile(r'(?<![A-Za-z0-9])#\d+\b')
            author_to_ticketset = defaultdict(set)

            for entry in commits:
                entry = entry.strip()
                if not entry:
                    continue
                parts = entry.split("\x1f", 2)
                author = parts[0].strip()
                subject = parts[1] if len(parts) > 1 else ""
                body = parts[2] if len(parts) > 2 else ""
                msg = (subject + "\n" + body).strip()

                found = set()
                for m in jira_re.findall(msg):
                    found.add(m.upper())
                for m in hash_re.findall(msg):
                    found.add(m)

                if found:
                    author_to_ticketset[author].update(found)

            if not author_to_ticketset:
                return "No tickets found."

            # Sort authors by ticket count descending, then by name
            lines = []
            for author, tickets in sorted(author_to_ticketset.items(), key=lambda kv: (-len(kv[1]), kv[0].lower())):
                if author in exclude:
                    continue  # skip excluded authors entirely
                lines.append(f"{author}: {', '.join(sorted(tickets))}")


            return "\n".join(lines)

    @Slot()
    def devChart(self):
        print("hofChart() SLOT CALLED")
        try:
            devCharts.main()
            print("hofCharts.main() COMPLETED")
        except Exception as e:
            print(f"hofCharts.main() ERROR: {e}")


    def __init__(self):
        super().__init__()
        self.flight_log = []  # List to store flight log entries
        self.predictions_log = []  # List to store prediction records
        self.current_prediction_label = ""
        self.current_model = "Random Forest"  # Default model
        self.current_framework = "PyTorch"  # Default framework
        self.deployed_model = None  # Loaded model instance for deployment
        self.deployed_model_path = None  # Path to deployed model
        self.deployed_model_type = None  # 'pytorch', 'tensorflow', 'pickle'
        # Precision metrics for each class (from codebase: Backward, Forward, Left, Right, Land, Takeoff)
        self.precision_metrics = {
            "Backward": "0.98",
            "Forward": "1.00",
            "Left": "0.97",
            "Right": "0.98",
            "Land": "0.96",
            "Takeoff": "0.98"
        }
        self.image_paths = []  # Store converted image paths
        self.plots_dir = os.path.abspath("plotscode/plots")  # Base plots directory
        self.current_dataset = "refresh"  # Default dataset to display
        try:
            self.tello = Tello()
        except Exception as e:
            print(f"Warning: Failed to initialize Tello drone: {e}")
            self.logMessage.emit(f"Warning: Failed to initialize Tello drone: {e}")
        
        # Initialize camera controller with tello instance
        self.camera_controller = CameraController()
        if hasattr(self, 'tello'):
            self.camera_controller.set_tello_instance(self.tello)

        # Initialize BCI connection for brainwave prediction
        if BCI_AVAILABLE:
            try:
                self.bcicon = bciConnection.get_instance()
                print("BCI connection initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize BCI connection: {e}")
                self.bcicon = None
        else:
            self.bcicon = None

    @Slot(str)
    def selectModel(self, model_name):
        """ Select the machine learning model """
        self.logMessage.emit(f"Model selected: {model_name}")
        self.current_model = model_name
        self.flight_log.insert(0, f"Selected Model: {model_name}")
        self.flightLogUpdated.emit(self.flight_log)

    @Slot(str)
    def selectFramework(self, framework_name):
        """ Select the machine learning framework """
        self.logMessage.emit(f"Framework selected: {framework_name}")
        self.current_framework = framework_name
        self.flight_log.insert(0, f"Selected Framework: {framework_name}")
        self.flightLogUpdated.emit(self.flight_log)

    # ========== AI/ML Training and Deployment Methods ==========
    
    @Slot(str, str, str, str, str, str)
    def startTraining(self, learning_rate, epochs, batch_size, architecture, data_path, model_type=""):
        """
        Start training a machine learning model with specified parameters
        
        Args:
            learning_rate: Learning rate for training (e.g., "0.001")
            epochs: Number of training epochs (e.g., "100")
            batch_size: Batch size for training (e.g., "32")
            architecture: Model architecture type (e.g., "CNN", "LSTM", "Transformer")
            data_path: Path to training data directory
            model_type: Type of model (e.g., "Random Forest", "Deep Learning")
        """
        try:
            self.trainingStatusUpdated.emit("Starting training process...")
            self.logMessage.emit(f"Training started: Architecture={architecture}, LR={learning_rate}, Epochs={epochs}")
            
            # Store training parameters
            self.training_params = {
                "learning_rate": float(learning_rate) if learning_rate else 0.001,
                "epochs": int(epochs) if epochs else 100,
                "batch_size": int(batch_size) if batch_size else 32,
                "architecture": architecture,
                "data_path": data_path,
                "model_type": model_type or self.current_model,
                "framework": self.current_framework
            }
            
            # Start training using synthetic dataset
            self.trainingStatusUpdated.emit(f"Loading dataset from: {data_path}")
            self.trainingLogUpdated.emit(f"Architecture: {architecture}")
            self.trainingLogUpdated.emit(f"Learning Rate: {self.training_params['learning_rate']}")
            self.trainingLogUpdated.emit(f"Epochs: {self.training_params['epochs']}")
            self.trainingLogUpdated.emit(f"Batch Size: {self.training_params['batch_size']}")
            
            # Use existing data loading pattern (similar to load_and_process_data)
            # This integrates with existing training routines
            self._train_model_with_synthetic_data(data_path, self.training_params, architecture)
            
            self.trainingStatusUpdated.emit("Training completed successfully")
            self.flight_log.insert(0, f"Training completed: {architecture} model")
            self.flightLogUpdated.emit(self.flight_log)
            
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            self.trainingStatusUpdated.emit(error_msg)
            self.logMessage.emit(error_msg)
            print(f"Error in startTraining: {e}")
    
    @Slot(str)
    def deployModel(self, model_path):
        """
        Deploy a trained model for inference
        
        Args:
            model_path: Path to the trained model file (.pkl, .h5, .pt, .pth, etc.)
        """
        try:
            self.deploymentStatusUpdated.emit("Loading model for deployment...")
            self.logMessage.emit(f"Deploying model from: {model_path}")
            
            if not model_path or not model_path.strip():
                self.deploymentStatusUpdated.emit("Error: No model file specified")
                return
            
            # Clean up file:/// prefix if present (from QML file dialogs)
            if model_path.startswith("file:///"):
                model_path = model_path[7:]
            
            model_path = Path(model_path).resolve()
            
            if not model_path.exists():
                self.deploymentStatusUpdated.emit(f"Error: Model file not found: {model_path}")
                return
            
            # Determine model type and load accordingly
            ext = model_path.suffix.lower()
            
            if ext in ['.pt', '.pth']:
                # Load PyTorch model
                try:
                    self.deployed_model = torch.load(str(model_path), map_location='cpu', weights_only=False)
                    self.deployed_model_type = 'pytorch'
                    self.deploymentStatusUpdated.emit(f"PyTorch model loaded: {model_path.name}")
                except Exception as e:
                    raise Exception(f"Failed to load PyTorch model: {e}")
            elif ext in ['.h5', '.keras']:
                # Load TensorFlow/Keras model
                try:
                    import tensorflow as tf
                    self.deployed_model = tf.keras.models.load_model(str(model_path))
                    self.deployed_model_type = 'tensorflow'
                    self.deploymentStatusUpdated.emit(f"TensorFlow model loaded: {model_path.name}")
                except ImportError:
                    raise Exception("TensorFlow not installed")
                except Exception as e:
                    raise Exception(f"Failed to load TensorFlow model: {e}")
            elif ext == '.pkl':
                # Load pickle-serialized model (sklearn, etc.)
                try:
                    import pickle
                    with open(model_path, 'rb') as f:
                        self.deployed_model = pickle.load(f)
                    self.deployed_model_type = 'pickle'
                    self.deploymentStatusUpdated.emit(f"Pickle model loaded: {model_path.name}")
                except Exception as e:
                    raise Exception(f"Failed to load pickle model: {e}")
            else:
                self.deploymentStatusUpdated.emit(f"Error: Unsupported model format: {ext}")
                return
            
            self.deployed_model_path = str(model_path)
            self.deploymentStatusUpdated.emit("Model deployed successfully")
            self.logMessage.emit(f"Model deployed: {model_path}")
            self.flight_log.insert(0, f"Model deployed: {model_path.name}")
            self.flightLogUpdated.emit(self.flight_log)
            
        except Exception as e:
            error_msg = f"Deployment error: {str(e)}"
            self.deploymentStatusUpdated.emit(error_msg)
            self.logMessage.emit(error_msg)
            print(f"Error in deployModel: {e}")
    
    def _train_model_with_synthetic_data(self, data_path, training_params, architecture):
        """
        Train a model using synthetic dataset by integrating with existing training routines
        
        This method loads data using existing patterns and calls appropriate training
        functions based on framework and model type.
        """
        try:
            import numpy as np
            import torch
            
            # Normalize class labels (from existing codebase pattern)
            def normalize_class_label(label):
                """Normalize class labels to standard format"""
                label_lower = label.lower()
                if 'backward' in label_lower:
                    return 'backward'
                elif 'forward' in label_lower:
                    return 'forward'
                elif 'landing' in label_lower or 'land' in label_lower:
                    return 'landing'
                elif 'left' in label_lower:
                    return 'left'
                elif 'right' in label_lower:
                    return 'right'
                elif 'take' in label_lower or 'takeoff' in label_lower:
                    return 'takeoff'
                return label
            
            # Load and process data (adapted from existing load_and_process_data pattern)
            self.trainingStatusUpdated.emit("Loading and processing data...")
            self.trainingLogUpdated.emit("Scanning dataset directory...")
            
            data_path = Path(data_path) if data_path else Path(".")
            all_samples = []
            all_labels = []
            successful_files = 0
            desired_rows = 160  # Default from existing codebase
            
            # Walk through data directory and load CSV files
            if not data_path.exists():
                self.trainingStatusUpdated.emit(f"Error: Data path does not exist: {data_path}")
                return
            
            for root, dirs, files in os.walk(data_path):
                if 'output' in root.lower():
                    continue
                csv_files = [f for f in files if f.endswith(('.csv', '.txt'))]
                if csv_files:
                    class_label_raw = os.path.basename(root)
                    if class_label_raw.startswith(('group', 'individual', 'Test', 'output')):
                        continue
                    class_label = normalize_class_label(class_label_raw)
                    
                    for csv_file in csv_files[:100]:  # Limit files per class for performance
                        try:
                            file_path = os.path.join(root, csv_file)
                            # Try different separators (from existing pattern)
                            df = None
                            for sep in [',', '\t', r'\s+']:
                                try:
                                    df = pd.read_csv(file_path, sep=sep, header=None, on_bad_lines='skip')
                                    if not df.empty and df.shape[1] > 1:
                                        break
                                except:
                                    continue
                            
                            if df is None or df.empty:
                                continue
                            
                            # Extract numeric columns only
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) == 0:
                                continue
                            df = df[numeric_cols]
                            
                            # Pad or truncate to desired_rows (from existing pattern)
                            if len(df) < desired_rows:
                                padding = pd.DataFrame(0, index=range(desired_rows - len(df)), 
                                                     columns=df.columns)
                                df = pd.concat([df, padding], ignore_index=True)
                            elif len(df) > desired_rows:
                                df = df.iloc[:desired_rows]
                            
                            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
                            
                            # Add samples (one per row or aggregated)
                            for idx, row in df.iterrows():
                                all_samples.append(row.values.astype(np.float32))
                                all_labels.append(class_label)
                            
                            successful_files += 1
                            if successful_files % 50 == 0:
                                self.trainingLogUpdated.emit(f"Processed {successful_files} files...")
                        except Exception as e:
                            continue
            
            if len(all_samples) == 0:
                self.trainingStatusUpdated.emit("Error: No valid data samples found")
                return
            
            # Prepare data arrays
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
            
            # Encode labels
            class_names = sorted(np.unique(y))
            label_to_idx = {label: idx for idx, label in enumerate(class_names)}
            y_encoded = np.array([label_to_idx[label] for label in y], dtype=np.int32)
            
            self.trainingLogUpdated.emit(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
            self.trainingLogUpdated.emit(f"Classes: {', '.join(class_names)}")
            
            # Split data (train/val/test - from existing pattern)
            n_samples = X.shape[0]
            n_test = int(n_samples * 0.2)
            n_val = int(n_samples * 0.1)
            
            np.random.seed(42)
            indices = np.random.permutation(n_samples)
            train_idx = indices[:n_samples - n_test - n_val]
            val_idx = indices[n_samples - n_test - n_val:n_samples - n_test]
            
            X_train, y_train = X[train_idx], y_encoded[train_idx]
            X_val, y_val = X[val_idx], y_encoded[val_idx]
            
            self.trainingLogUpdated.emit(f"Train: {len(X_train)}, Val: {len(X_val)}")
            
            # Train based on framework and model type
            framework = training_params.get('framework', 'PyTorch').lower()
            model_type = training_params.get('model_type', 'Deep Learning').lower()
            
            self.trainingStatusUpdated.emit(f"Training {architecture} model ({framework})...")
            
            if 'random' in model_type or 'forest' in model_type:
                # Random Forest training
                if framework in ['pytorch', 'torch']:
                    self._train_random_forest_pytorch(X_train, y_train, X_val, y_val, training_params, class_names)
                elif framework == 'tensorflow':
                    self._train_random_forest_tensorflow(X_train, y_train, X_val, y_val, training_params, class_names)
                else:
                    # Fallback to sklearn Random Forest
                    self._train_random_forest_sklearn(X_train, y_train, X_val, y_val, training_params, class_names)
            else:
                # Deep Learning training
                if framework in ['pytorch', 'torch']:
                    self._train_deep_learning_pytorch(X_train, y_train, X_val, y_val, training_params, architecture, class_names)
                elif framework == 'tensorflow':
                    self._train_deep_learning_tensorflow(X_train, y_train, X_val, y_val, training_params, architecture, class_names)
                else:
                    # Try PyTorch as default
                    self._train_deep_learning_pytorch(X_train, y_train, X_val, y_val, training_params, architecture, class_names)
            
            self.trainingStatusUpdated.emit("Training completed successfully")
            self.trainingLogUpdated.emit("Model saved and ready for deployment")
            
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            self.trainingStatusUpdated.emit(error_msg)
            self.trainingLogUpdated.emit(f"Error details: {str(e)}")
            print(f"Error in _train_model_with_synthetic_data: {e}")
            import traceback
            traceback.print_exc()
    
    def _train_random_forest_sklearn(self, X_train, y_train, X_val, y_val, params, class_names):
        """Train Random Forest using sklearn (fallback)"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            
            self.trainingLogUpdated.emit("Training Random Forest (sklearn)...")
            
            n_estimators = 100
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Evaluate
            val_pred = model.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            
            self.trainingLogUpdated.emit(f"Validation Accuracy: {val_acc*100:.2f}%")
            
            # Calculate precision metrics for each class
            from sklearn.metrics import precision_score, classification_report
            try:
                # Get precision for each class
                precision_per_class = precision_score(y_val, val_pred, average=None, zero_division=0)
                
                # Update precision metrics dictionary
                for i, class_name in enumerate(class_names):
                    # Normalize class name (capitalize first letter, handle 'land' vs 'landing')
                    normalized_name = class_name.capitalize()
                    if normalized_name.lower() in ['land', 'landing']:
                        normalized_name = "Land"
                    elif normalized_name.lower() in ['take', 'takeoff']:
                        normalized_name = "Takeoff"
                    
                    # Map to standard class names used in codebase
                    class_mapping = {
                        'backward': 'Backward',
                        'forward': 'Forward',
                        'left': 'Left',
                        'right': 'Right',
                        'land': 'Land',
                        'landing': 'Land',
                        'takeoff': 'Takeoff',
                        'take': 'Takeoff'
                    }
                    
                    normalized_key = class_mapping.get(normalized_name.lower(), normalized_name)
                    
                    if i < len(precision_per_class):
                        self.precision_metrics[normalized_key] = f"{precision_per_class[i]:.2f}"
                
                # Emit updated precision metrics
                self.precisionMetricsUpdated.emit(self.precision_metrics)
                self.trainingLogUpdated.emit("Precision metrics calculated and updated")
            except Exception as e:
                self.trainingLogUpdated.emit(f"Note: Could not calculate precision: {e}")
            
            # Save model
            import pickle
            model_path = Path("Models") / f"rf_sklearn_{int(time.time())}.pkl"
            model_path.parent.mkdir(exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            self.trainingLogUpdated.emit(f"Model saved to: {model_path}")
            
        except Exception as e:
            self.trainingLogUpdated.emit(f"Error training Random Forest: {e}")
    
    def _train_random_forest_pytorch(self, X_train, y_train, X_val, y_val, params, class_names):
        """Train Random Forest using PyTorch (if available)"""
        try:
            import torch
            self.trainingLogUpdated.emit("PyTorch Random Forest - using sklearn fallback")
            self._train_random_forest_sklearn(X_train, y_train, X_val, y_val, params, class_names)
        except Exception as e:
            self.trainingLogUpdated.emit(f"Error: {e}")
    
    def _train_random_forest_tensorflow(self, X_train, y_train, X_val, y_val, params, class_names):
        """Train Random Forest using TensorFlow (if available)"""
        try:
            self.trainingLogUpdated.emit("TensorFlow Random Forest - using sklearn fallback")
            self._train_random_forest_sklearn(X_train, y_train, X_val, y_val, params, class_names)
        except Exception as e:
            self.trainingLogUpdated.emit(f"Error: {e}")
    
    def _train_deep_learning_pytorch(self, X_train, y_train, X_val, y_val, params, architecture, class_names):
        """Train Deep Learning model using PyTorch"""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import TensorDataset, DataLoader
            
            self.trainingLogUpdated.emit(f"Training {architecture} model (PyTorch)...")
            
            n_classes = len(class_names)
            learning_rate = params.get('learning_rate', 0.001)
            epochs = params.get('epochs', 100)
            batch_size = params.get('batch_size', 32)
            
            # Convert to PyTorch tensors
            X_train_t = torch.FloatTensor(X_train)
            y_train_t = torch.LongTensor(y_train)
            X_val_t = torch.FloatTensor(X_val)
            y_val_t = torch.LongTensor(y_val)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Define model architecture
            input_dim = X_train.shape[1]
            hidden_dims = [128, 64, 32] if architecture.lower() != 'lstm' else [64, 32]
            
            if architecture.lower() == 'lstm':
                # LSTM requires custom forward, use MLP for now
                # TODO: Implement proper LSTM model class
                layers = []
                prev_dim = input_dim
                for dim in hidden_dims:
                    layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(0.3)])
                    prev_dim = dim
                layers.append(nn.Linear(prev_dim, n_classes))
                model = nn.Sequential(*layers)
            elif architecture.lower() == 'cnn':
                # Simple CNN for 1D signals
                model = nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Flatten(),
                    nn.Linear(32 * (input_dim // 2), hidden_dims[0]),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dims[0], n_classes)
                )
            else:
                # MLP (default)
                layers = []
                prev_dim = input_dim
                for dim in hidden_dims:
                    layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(0.3)])
                    prev_dim = dim
                layers.append(nn.Linear(prev_dim, n_classes))
                model = nn.Sequential(*layers)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Training loop
            best_val_acc = 0.0
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                correct = 0
                total = 0
                
                for X_batch, y_batch in train_loader:
                    if architecture.lower() == 'cnn':
                        X_batch = X_batch.unsqueeze(1)  # Add channel dimension for CNN
                    
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
                
                train_acc = correct / total
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_input = X_val_t.unsqueeze(1) if architecture.lower() == 'cnn' else X_val_t
                    val_outputs = model(val_input)
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_acc = (val_predicted == y_val_t).float().mean().item()
                
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    self.trainingLogUpdated.emit(
                        f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%"
                    )
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
            
            # Save best model
            model_path = Path("Models") / f"dl_pytorch_{architecture.lower()}_{int(time.time())}.pt"
            model_path.parent.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': best_model_state,
                'architecture': architecture,
                'input_dim': input_dim,
                'n_classes': n_classes,
                'class_names': class_names
            }, model_path)
            
            self.trainingLogUpdated.emit(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
            self.trainingLogUpdated.emit(f"Model saved to: {model_path}")
            
        except Exception as e:
            self.trainingLogUpdated.emit(f"Error training PyTorch model: {e}")
            import traceback
            traceback.print_exc()
    
    def _train_deep_learning_tensorflow(self, X_train, y_train, X_val, y_val, params, architecture, class_names):
        """Train Deep Learning model using TensorFlow"""
        try:
            import tensorflow as tf
            
            self.trainingLogUpdated.emit(f"Training {architecture} model (TensorFlow)...")
            self.trainingLogUpdated.emit("TensorFlow training - PyTorch recommended for now")
            # Fallback to PyTorch
            self._train_deep_learning_pytorch(X_train, y_train, X_val, y_val, params, architecture, class_names)
            
        except Exception as e:
            self.trainingLogUpdated.emit(f"Error: {e}")
    
    @Slot(result=str)
    def runInference(self):
        """
        Run inference using the deployed model
        Returns prediction result as string
        """
        try:
            if self.deployed_model is None:
                # Fallback to BCI connection if available
                if hasattr(self, 'bcicon') and self.bcicon:
                    prediction_response = self.bcicon.bciConnectionController()
                    if prediction_response:
                        prediction = prediction_response.get('prediction_label', 'unknown')
                        self.inferenceOutputUpdated.emit(f"Prediction: {prediction}")
                        return prediction
                self.inferenceOutputUpdated.emit("Error: No model deployed")
                return "No model deployed"
            
            # Use deployed model for inference
            if self.deployed_model_type == 'pytorch':
                # For PyTorch models, use BCI connection if available, otherwise use model directly
                if hasattr(self, 'bcicon') and self.bcicon:
                    prediction_response = self.bcicon.bciConnectionController()
                    if prediction_response:
                        prediction = prediction_response.get('prediction_label', 'unknown')
                        self.inferenceOutputUpdated.emit(f"Prediction: {prediction}")
                        return prediction
                # TODO: Direct model inference can be added here
                return "Model loaded but inference needs implementation"
            elif self.deployed_model_type == 'tensorflow':
                # TensorFlow inference
                if hasattr(self, 'bcicon') and self.bcicon:
                    prediction_response = self.bcicon.bciConnectionController()
                    if prediction_response:
                        prediction = prediction_response.get('prediction_label', 'unknown')
                        self.inferenceOutputUpdated.emit(f"Prediction: {prediction}")
                        return prediction
                return "Model loaded but inference needs implementation"
            else:
                # For pickle models (sklearn), use BCI connection
                if hasattr(self, 'bcicon') and self.bcicon:
                    prediction_response = self.bcicon.bciConnectionController()
                    if prediction_response:
                        prediction = prediction_response.get('prediction_label', 'unknown')
                        self.inferenceOutputUpdated.emit(f"Prediction: {prediction}")
                        return prediction
                return "Model loaded but inference needs implementation"
                
        except Exception as e:
            error_msg = f"Inference error: {str(e)}"
            self.inferenceOutputUpdated.emit(error_msg)
            print(f"Error in runInference: {e}")
            return error_msg
    
    @Slot(result=dict)
    def getPrecisionMetrics(self):
        """
        Get precision metrics for each class (called from QML)
        
        Returns:
            dict: Dictionary with class names as keys and precision values as strings
        """
        return self.precision_metrics
    
    @Slot(str)
    def selectAIModelType(self, model_type):
        """
        Select the AI/ML model type (called from QML)
        
        Args:
            model_type: Type of model (e.g., "CNN", "LSTM", "Transformer", "Random Forest")
        """
        self.current_model = model_type
        self.logMessage.emit(f"AI Model type selected: {model_type}")
        self.flight_log.insert(0, f"AI Model type: {model_type}")
        self.flightLogUpdated.emit(self.flight_log)
# ========== lastl line for AI/ML Training and Deployment Methods ==========
    @Slot()
    def readMyMind(self):
        """ Runs the selected model and processes the brainwave data. """
        if self.current_model == "Random Forest":
            if self.current_framework == "PyTorch":
                prediction = self.run_random_forest_pytorch()
            else:
                prediction = self.run_random_forest_tensorflow()
        else:  # Deep Learning
            if self.current_framework == "PyTorch":
                prediction = self.run_deep_learning_pytorch()
            else:
                prediction = self.run_deep_learning_tensorflow()
        self.logMessage.emit(f"Prediction received: {prediction}")
        # Set current prediction
        self.current_prediction_label = prediction

        # Log the prediction
        self.predictions_log.append({
            "count": str(len(self.predictions_log) + 1),
            "server": "Brainwave AI",
            "label": prediction
        })
        self.predictionsTableUpdated.emit(self.predictions_log)

        # Update Flight Log
        self.flight_log.insert(0, f"Executed: {prediction} (Model: {self.current_model}, Framework: {self.current_framework})")
        self.flightLogUpdated.emit(self.flight_log)

    def run_random_forest_pytorch(self):
        """ Random Forest model processing with PyTorch backend """
        print("Running Random Forest Model with PyTorch...")
        try:
            # Use the BCI connection to get real brainwave data
            if hasattr(self, 'bcicon') and self.bcicon:
                prediction_response = self.bcicon.bciConnectionController()
                if prediction_response:
                    return prediction_response.get('prediction_label', 'forward')
        except Exception as e:
            print(f"Error with PyTorch Random Forest: {e}")

        # Fallback to simulation with PyTorch-specific labels
        time.sleep(1)
        return random.choice(["forward", "backward", "left", "right", "takeoff", "land"])

    def run_random_forest_tensorflow(self):
        """ Random Forest model processing with TensorFlow backend """
        print("Running Random Forest Model with TensorFlow...")
        try:
            # Use the BCI connection to get real brainwave data
            if hasattr(self, 'bcicon') and self.bcicon:
                prediction_response = self.bcicon.bciConnectionController()
                if prediction_response:
                    return prediction_response.get('prediction_label', 'forward')
        except Exception as e:
            print(f"Error with TensorFlow Random Forest: {e}")

        # Fallback to simulation with TensorFlow-specific labels
        time.sleep(1)
        return random.choice(["forward", "backward", "left", "right", "takeoff", "land"])

    def run_deep_learning_pytorch(self):
        """ Deep Learning model processing with PyTorch backend """
        print("Running Deep Learning Model with PyTorch...")
        try:
            # Simulate PyTorch deep learning model processing
            # In a real implementation, this would load and run a PyTorch CNN model
            time.sleep(2)  # Simulate longer processing time for deep learning
            return random.choice(["forward", "backward", "left", "right", "takeoff", "land", "up", "down"])
        except Exception as e:
            print(f"Error with PyTorch Deep Learning: {e}")
            return "forward"

    def run_deep_learning_tensorflow(self):
        """ Deep Learning model processing with TensorFlow backend """
        print("Running Deep Learning Model with TensorFlow...")
        try:
            # Simulate TensorFlow deep learning model processing
            # In a real implementation, this would load and run a TensorFlow CNN model
            time.sleep(2)  # Simulate longer processing time for deep learning
            return random.choice(["forward", "backward", "left", "right", "takeoff", "land", "up", "down"])
        except Exception as e:
            print(f"Error with TensorFlow Deep Learning: {e}")
            return "forward"

    @Slot(str)
    def notWhatIWasThinking(self, manual_action):
        # Handle manual action input
        self.predictions_log.append({
            "count": "manual",
            "server": "manual",
            "label": manual_action
        })
        self.predictionsTableUpdated.emit(self.predictions_log)

        # Also update flight log
        self.flight_log.insert(0, f"Manual Action: {manual_action}")
        self.flightLogUpdated.emit(self.flight_log)
        self.logMessage.emit(f"Manual input: {manual_action}") 

    @Slot()
    def executeAction(self):
        # Execute the current prediction
        if self.current_prediction_label:
            self.flight_log.insert(0, f"Executed: {self.current_prediction_label}")
            self.flightLogUpdated.emit(self.flight_log)
            self.logMessage.emit(f"Executed action: {self.current_prediction_label}")

    @Slot()
    def connectDrone(self):
        # Mock function to simulate drone connection
        self.flight_log.insert(0, "Drone connected.")
        self.flightLogUpdated.emit(self.flight_log)
        self.logMessage.emit("Drone connected.")

    @Slot()
    def keepDroneAlive(self):
        # Mock function to simulate sending keep-alive signal
        self.flight_log.insert(0, "Keep alive signal sent.")
        self.flightLogUpdated.emit(self.flight_log)

    @Slot(str)
    def getDroneAction(self, action):
        try:
            if action == 'connect':
                self.tello.connect()
                self.logMessage.emit("Connected to Tello Drone")
            elif action == 'up':
                self.tello.move_up(30)
                self.logMessage.emit("Moving up")
            elif action == 'down':
                self.tello.move_down(30)
                self.logMessage.emit("Moving down")
            elif action == 'forward':
                self.tello.move_forward(30)
                self.logMessage.emit("Moving forward")
            elif action == 'backward':
                self.tello.move_back(30)
                self.logMessage.emit("Moving backward")
            elif action == 'left':
                self.tello.move_left(30)
                self.logMessage.emit("Moving left")
            elif action == 'right':
                self.tello.move_right(30)
                self.logMessage.emit("Moving right")
            elif action == 'turn_left':
                self.tello.rotate_counter_clockwise(45)
                self.logMessage.emit("Rotating left")
            elif action == 'turn_right':
                self.tello.rotate_clockwise(45)
                self.logMessage.emit("Rotating right")
            elif action == 'takeoff':
                self.tello.takeoff()
                self.logMessage.emit("Taking off")
            elif action == 'land':
                self.tello.land()
                self.logMessage.emit("Landing")
            elif action == 'go_home':
                self.go_home()
            elif action == 'stream':
                if hasattr(self, 'camera_controller'):
                    self.camera_controller.start_camera_stream()
                    self.logMessage.emit("Starting camera stream")
                else:
                    self.logMessage.emit("Camera controller not available")
            else:
                self.logMessage.emit("Unknown action")
        except Exception as e:
            self.logMessage.emit(f"Error during {action}: {e}")

    # Method for returning to home (an approximation)
    def go_home(self):
        # Assuming the home action means moving backward and upwards
        self.tello.move_back(50)  # Move back to home point (adjust distance as needed)
        self.tello.move_up(50)  # Move up to avoid obstacles
        self.logMessage.emit("Returning to home")

    @Slot()
    def check_plots_exist(self):
        """
        Check if all necessary plot PDFs exist in both Rollback and Refresh directories.
        If not, run controller.py to generate them.
        """
        print("\n=== CHECKING IF PLOTS EXIST ===")

        # Create plots base directory if it doesn't exist
        plots_base_dir = Path(self.plots_dir)
        if not plots_base_dir.exists():
            print(f"Creating plots base directory: {plots_base_dir}")
            plots_base_dir.mkdir(parents=True, exist_ok=True)

        # List of datasets to check
        datasets = ["rollback", "refresh"]

        # List of PDF files that should exist for each dataset
        pdf_files = [
            "takeoff_plots.pdf", "forward_plots.pdf", "right_plots.pdf",
            "land_plots.pdf", "backward_plots.pdf", "left_plots.pdf"
        ]

        # Check if all directories and PDFs exist
        missing_pdfs = False
        for dataset in datasets:
            dataset_dir = plots_base_dir / dataset
            if not dataset_dir.exists():
                print(f"Creating dataset directory: {dataset_dir}")
                dataset_dir.mkdir(parents=True, exist_ok=True)
                missing_pdfs = True
                continue

            print(f"Checking PDFs in {dataset_dir}...")
            for pdf_file in pdf_files:
                pdf_path = dataset_dir / pdf_file
                if not pdf_path.exists():
                    print(f"Missing file: {pdf_path}")
                    missing_pdfs = True
                    break

        # If any PDFs are missing, run the controller.py script
        if missing_pdfs:
            print("Some plot files are missing. Running controller.py to generate them...")

            # Get the path to controller.py in the plotscode directory
            controller_path = Path(self.plots_dir).parent / "controller.py"  # plotscode/controller.py
            print(f"Controller path: {controller_path}")
            print(f"Controller exists: {controller_path.exists()}")

            if controller_path.exists():
                try:
                    # Change to the plotscode directory before running the script
                    original_dir = os.getcwd()
                    os.chdir(controller_path.parent)

                    # Run the controller.py script to generate plots for both datasets
                    print(f"Executing: {sys.executable} {controller_path}")
                    result = subprocess.run(
                        [sys.executable, str(controller_path)],
                        check=True,
                        capture_output=True,
                        text=True
                    )

                    # Go back to the original directory
                    os.chdir(original_dir)

                    # Print output for debugging
                    print(f"Output: {result.stdout}")
                    if result.stderr:
                        print(f"Errors: {result.stderr}")

                    print("Successfully generated plot files.")
                    return True
                except subprocess.CalledProcessError as e:
                    print(f"Error running controller.py: {e}")
                    if hasattr(e, 'stderr'):
                        print(f"Error output: {e.stderr}")
                    return False
                except Exception as e:
                    print(f"Unexpected error: {str(e)}")
                    return False
            else:
                print(f"Controller script not found: {controller_path}")
                return False

        return True  # All files exist

    @Slot(str)
    def setDataset(self, dataset_name):
        """
        Set the current dataset to display (refresh or rollback).
        :param dataset_name: Name of the dataset ('refresh' or 'rollback')
        """
        if dataset_name.lower() in ["refresh", "rollback"]:
            self.current_dataset = dataset_name.lower()
            print(f"Switched to {self.current_dataset} dataset")
            # Update the displayed images
            self.convert_pdfs_to_images()
        else:
            print(f"Invalid dataset name: {dataset_name}")

    @Slot()
    def convert_pdfs_to_images(self):
        """
        Convert PDF files from the current dataset to images and send to QML.
        """
        print(f"\n=== STARTING CONVERT PDFS TO IMAGES FOR {self.current_dataset.upper()} ===")

        # First check if all plot PDFs exist, and generate them if needed
        success = self.check_plots_exist()
        print(f"Result of check_plots_exist: {success}")

        # Current dataset directory
        dataset_dir = Path(self.plots_dir) / self.current_dataset

        # Convert PDF files to images and send image paths + graph names to QML.
        self.image_paths = []
        graph_titles = ["Takeoff", "Forward", "Right",
                        "Landing", "Backward", "Left"]

        # Load files in the correct order
        pdf_files = [
            "takeoff_plots.pdf", "forward_plots.pdf", "right_plots.pdf",
            "land_plots.pdf", "backward_plots.pdf", "left_plots.pdf"
        ]

        for i, pdf_file in enumerate(pdf_files):
            pdf_path = dataset_dir / pdf_file
            if not pdf_path.exists():
                print(f"Missing file: {pdf_path}")  # Debugging: Check missing PDFs
                continue  # Skip if file does not exist

            images = convert_from_path(str(pdf_path), dpi=150)  # Convert PDF to image
            image_path = dataset_dir / f"{pdf_file.replace('.pdf', '.png')}"
            images[0].save(str(image_path), "PNG")  # Save first page as an image

            # Debugging: Print the generated image path
            print(f"Generated image: {image_path}")

            self.image_paths.append({
                "graphTitle": graph_titles[i],
                "imagePath": QUrl.fromLocalFile(str(image_path)).toString()
            })

        # Debugging: Print final list of image paths
        print("Final Image Paths Sent to QML:", self.image_paths)

        self.imagesReady.emit(self.image_paths)  # Send data to QML

    @Slot()
    def launch_file_shuffler_gui(self):
        # Launch the file shuffler GUI program
        file_shuffler_path = Path(__file__).resolve().parent / "file-shuffler/file-shuffler-gui.py"
        subprocess.Popen(["python", str(file_shuffler_path)])

    @Slot(str, result=str)
    def run_file_shuffler_program(self, path):
        # Need to parse the path as the FolderDialog appends file:// in front of the selection
        path = path.replace("file://", "")
        if path.startswith("/C:"):
            path = 'C' + path[2:]

        response = run_file_shuffler.main(path)
        return response

        # Adding Synthetic Data and Live Data Logic (Row 327 to 355) as part of Ticket 186

    @Slot(str, result=str)
    def unify_thoughts(self, base_dir):
        """
        Called from QML when the user picks a directory.
        """
        # strip file:/// if necessary
        path = base_dir.replace("file://", "")

        if base_dir.startswith("file:///"):
            base_dir = urllib.parse.unquote(base_dir.replace("file://", ""))
            if os.name == 'nt' and base_dir.startswith("/"):
                base_dir = base_dir[1:]
        print("Unify Thoughts on directory:", base_dir)
        output = io.StringIO()

        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                unifyTXT.move_any_txt_files(base_dir)
                print("Unify complete.")

        except Exception as e:
            print("Error during unify:", e)

        return output.getvalue()

    @Slot(str, result=str)
    def remove_8_channel(self, base_dir):
        """
        Called from QML when the user picks a directory to remove 8 channel data.
        """
        # Decode URL path
        if base_dir.startswith("file:///"):
            base_dir = urllib.parse.unquote(base_dir.replace("file://", ""))
            if os.name == 'nt' and base_dir.startswith("/"):
                base_dir = base_dir[1:]
        print("Removing 8 Channel data form:", base_dir)
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                remove8channel.file_remover(base_dir)
                print("8 Channel Data Removal complete.")
        except Exception as e:
            print("Error during cleanup: ", e)

    @Slot(str)
    def setDataMode(self, mode):
        """
        Set data mode to either synthetic or live based on radio button selection.
        """
        if mode == "synthetic":
            self.init_synthetic_board()
            self.logMessage.emit("Switched to Synthetic Data Mode")
        elif mode == "live":
            self.init_live_board()
            self.logMessage.emit("Switched to Live Data Mode")
        else:
            print(f"Unknown data mode: {mode}")

    def init_synthetic_board(self):
        """ Initialize BrainFlow with synthetic board for testing """
        params = BrainFlowInputParams()
        self.board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
        print("\nSynthetic board initialized.")

    def init_live_board(self):
        """ Initialize BrainFlow with a real headset """
        params = BrainFlowInputParams()
        params.serial_port = "/dev/cu.usbserial-D200PMA1"  # Update if different on your system
        self.board = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)
        print("\nLive headset board initialized.")




if __name__ == "__main__":
    os.environ["QT_QUICK_CONTROLS_STYLE"] = "Fusion"
    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()

    # Create our controllers
    tab_controller = TabController()
    backend = BrainwavesBackend()
    
    engine.rootContext().setContextProperty("tabController", tab_controller)
    engine.rootContext().setContextProperty("backend", backend)
    engine.rootContext().setContextProperty("imageModel", [])  # Initialize empty model
    engine.rootContext().setContextProperty("fileShufflerGui", backend)  # For file shuffler
    engine.rootContext().setContextProperty("cameraController", backend.camera_controller)
    engine.rootContext().setContextProperty("fileShufflerGui", backend)  # For file shuffler

    # Load QML
    qml_file = Path(__file__).resolve().parent / "main.qml"
    engine.load(str(qml_file))
    
    # Check if QML loaded successfully
    if not engine.rootObjects():
        print("ERROR: Failed to load QML file - no root objects created!")
        print("This usually means there's a QML syntax error or missing QML component.")
        sys.exit(-1)

    # Convert PDFs after engine load
    try:
        backend.convert_pdfs_to_images()
    except Exception as e:
        print(f"Warning: Error converting PDFs: {str(e)}")

    # Ensure image model updates correctly
    backend.imagesReady.connect(lambda images: engine.rootContext().setContextProperty("imageModel", images))

    sys.exit(app.exec())




