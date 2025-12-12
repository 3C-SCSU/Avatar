import sys
import os
import subprocess
from pathlib import Path
from PySide6.QtWidgets import QApplication,QFileDialog, QMessageBox
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject, Signal, Slot, Property, QProcess, QUrl, QTimer
from pdf2image import convert_from_path
from djitellopy import Tello
import random
import threading
import re
import pandas as pd
import torch
import time
from collections import defaultdict
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from predictions_local.brainflowprocessor import BrainFlowDataProcessor
from predictions_local.deeplearningpytorchpredictor import DeeplearningPytorchPredictor
from cameraview.camera_controller import CameraController
from NAO6.nao_connection import send_command
import asyncio
import copy
import queue
# from Developers.hofCharts import main as hofCharts, ticketsByDev_text NA

from developers_api import DevelopersAPI

								
from NA06_Manual_Control import ManualNaoController
from NA06_Manual_Control.camera_view import DroneCameraController

# Import BCI connection for brainwave prediction
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'random-forest-prediction')))
    from client.brainflow1 import bciConnection, DataMode
    BCI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: BCI connection not available: {e}")
    BCI_AVAILABLE = False


from cloud_api import CloudAPI

from shuffler_api import ShufflerAPI

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
    enqueueMoveRequested = Signal(str)

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



    def __init__(self):
        super().__init__()
        self.action_log = [] # List used to store the actions performed by the drone
        self.flight_log = []  # List to store flight log entries
        self.predictions_log = []  # List to store prediction records
        self.current_prediction_label = ""
        self.current_data_mode = "synthetic"
        self.current_model = "Random Forest"  # Default model
        self.current_framework = "PyTorch"  # Default framework
        self.image_paths = []  # Store converted image paths
        self.plots_dir = os.path.abspath("plotscode/plots")  # Base plots directory
        self.current_dataset = "refresh"  # Default dataset to display
        self.connected = False
        self.drone_lock = threading.RLock()  # <-- reentrant lock avoids deadlock

        # Timer to send periodic hover signals ()
        self.hover_timer = QTimer()
        self.hover_timer.timeout.connect(self.hover_loop)

        self.is_flying = False


        #Movement clumping
        self.step_cm = 30                     # each "move" command = 30 cm
        self.clump_dir = None                 # current direction being clumped
        self.clump_count = 0                  # how many commands in this batch
        self.clump_window_ms = 1000           # 1 second clump window

        self.clump_timer = QTimer()
        self.clump_timer.setSingleShot(True)
        self.clump_timer.timeout.connect(self._flush_clumped_move)
        self.enqueueMoveRequested.connect(self._enqueue_move)

        # optional override for movement distance (used by clumper)
        self._movement_distance_override = None

        #Tello command queue 
        self.cmd_queue = queue.Queue()
        self._drone_worker = threading.Thread(
            target=self._drone_loop,
            daemon=True,
        )
        self._drone_worker.start()

        try:
            self.tello = Tello(retry_count=1)
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

    # Command queue helpers
    def _queue_action(self, action: str, dist: int | None = None):
        """
        Put a command into the Tello queue.
        dist is an optional distance override (cm) for movement commands.
        """
        self.cmd_queue.put((action, dist))

    def _drone_loop(self):
        """
        Single worker thread that owns the Tello.
        It pulls commands from the queue and executes them in FIFO order.
        """
        while True:
            action, dist = self.cmd_queue.get()
            try:
                # Set distance override if provided
                old = self._movement_distance_override
                if dist is not None:
                    self._movement_distance_override = dist
                try:
                    # All low-level execution is here
                    self.getDroneAction(action)
                finally:
                    self._movement_distance_override = old
            except Exception as e:
                error_msg = f"Worker error during {action}: {e}"
                print(error_msg)
                self.logMessage.emit(error_msg)
            finally:
                self.cmd_queue.task_done()

    @Slot()
    def takeoff(self):
        self.tello.takeoff()
        self.connected = True
        self.is_flying = True
        self.hover_timer.start(200)
        self.logMessage.emit("Hovering")
    
    @Slot()
    def hover(self):
        self.tello.send_rc_control(0, 0, 0, 0)
        self.logMessage.emit("Hovering")

    def hover_loop(self):
        if self.is_flying:
            pass
            #self.tello.send_rc_control(0, 0, 0, 0)

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

    @Slot()
    def readMyMind(self):
        """ Runs the selected model and processes the brainwave data. """
        if self.current_model == "Random Forest":
            if self.current_framework == "PyTorch":
                prediction = self.run_random_forest_pytorch()
            else:
                prediction = self.run_random_forest_tensorflow()
        elif self.current_model == "GaussianNB":
            if self.current_framework == "PyTorch":
                prediction = self.run_gaussiannb_pytorch()
            else:
                prediction = self.run_gaussiannb_tensorflow()
        else:  # Deep Learning
            if self.current_framework == "PyTorch":
                prediction = self.run_deep_learning_pytorch()
                self.logMessage.emit("Getting prediction from pytorch using the deep learning model.")
            else:
                prediction = self.run_deep_learning_tensorflow()
        self.logMessage.emit(f"Prediction received: {prediction}")
        # Set current prediction
        self.current_prediction_label = prediction

        #auto mode is desired
        self.doDroneTAction(prediction)

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
        self.get_brainwave_data()
        try:
            model = DeeplearningPytorchPredictor()
            pred_label = model(self.brainwave_data)
            return pred_label
        except Exception as e:
            print(f"Error with PyTorch Deep Learning: {e}")
            return "Error"
    
    def get_brainwave_data(self):
        if self.current_data_mode == 'synthetic':
            self.brainwave_processor = BrainFlowDataProcessor(board_id=BoardIds.SYNTHETIC_BOARD.value)
            self.brainwave_data = self.brainwave_processor.get_tensor()
            print("synethetic data retrieved")
            return self.brainwave_data
        else:
            self.brainwave_processor = BrainFlowDataProcessor(board_id=BoardIds.CYTON_DAISY_BOARD.value)
            self.brainwave_data = self.brainwave_processor.get_tensor()
            print("live Cyton Daisy data retrieved")
            return self.brainwave_data

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
    def run_gaussiannb_pytorch(self):
        """ GaussianNB model processing with PyTorch backend """
        print("Running GaussianNB Model with PyTorch...")
        try:
            # Import the GaussianNB model
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), 'prediction-gaussiannb', 'pytorch'))
            from gaussiannb_model import GaussianNB
            
            # Try to load trained model
            model_path = os.path.join(os.path.dirname(__file__), 'prediction-gaussiannb', 'pytorch', 'gaussiannb_trained.pth')
            
            if os.path.exists(model_path):
                # Load the trained model
                checkpoint = torch.load(model_path)
                model = GaussianNB(
                    num_features=checkpoint['num_features'],
                    num_classes=checkpoint['num_classes']
                )
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Get prediction from BCI connection
                if hasattr(self, 'bcicon'):
                    prediction_response = self.bcicon.bciConnectionController()
                    if prediction_response:
                        return prediction_response.get('prediction_label', 'forward')
                
                # Fallback
                return random.choice(["forward", "backward", "left", "right", "takeoff", "land"])
            else:
                print(f"GaussianNB model not found at {model_path}. Using simulation.")
                return random.choice(["forward", "backward", "left", "right", "takeoff", "land"])
                
        except Exception as e:
            print(f"Error with PyTorch GaussianNB: {e}")
            return random.choice(["forward", "backward", "left", "right", "takeoff", "land"])
    
    def run_gaussiannb_tensorflow(self):
        """ GaussianNB model processing with TensorFlow backend """
        print("Running GaussianNB Model with TensorFlow...")
        try:
            # Note: GaussianNB with TensorFlow is not implemented yet
            # For now, fallback to simulation
            print("TensorFlow backend for GaussianNB not implemented. Using simulation.")
            time.sleep(1)
            return random.choice(["forward", "backward", "left", "right", "takeoff", "land"])
        except Exception as e:
            print(f"Error with TensorFlow GaussianNB: {e}")
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

            # send it to the drone
            self.doDroneTAction(self.current_prediction_label)
            
    @Slot()
    def connectDrone(self):
        self.doDroneTAction('connect')
        self.flight_log.insert(0, "Drone connected.")
        self.flightLogUpdated.emit(self.flight_log)
        self.logMessage.emit("Drone connected.")

    @Slot(str)
    def keepDroneAlive(self,text):
        # Mock function to simulate sending keep-alive signal
        self.flight_log.insert(0, "Keep alive signal sent.")
        self.flightLogUpdated.emit(self.flight_log)

        #remove null at end and make all lowercase
        text = text.strip().lower()

        #exicut cmd
        self.doDroneTAction(text)

        
        

    @Slot(str)
    def doDroneTAction(self, action):
        if action in ('up', 'down', 'forward', 'backward', 'left', 'right'):
            # Clumped in main thread; clumper will enqueue final chunks.
            self.enqueueMoveRequested.emit(action)
        else:
            # Non-movement actions go straight into the queue.
            self._queue_action(action)



    def _enqueue_move(self, direction: str):
        with self.drone_lock:
            if self.clump_dir is None:
                # start new batch
                self.clump_dir = direction
                self.clump_count = 1
            elif self.clump_dir == direction:
                # same direction → just increase count
                self.clump_count += 1
            else:
                # different direction → flush existing batch immediately
                d = self.clump_dir
                c = self.clump_count
                self.clump_dir = direction
                self.clump_count = 1
                # flush previous batch (this only enqueues work)
                self._execute_clumped_move(d, c)

            # (re)start 1s timer (Qt main thread)
            self.clump_timer.stop()
            self.clump_timer.start(self.clump_window_ms)


    def _flush_clumped_move(self):
        with self.drone_lock:
            if self.clump_dir is None or self.clump_count == 0:
                return
            direction = self.clump_dir
            count = self.clump_count
            self.clump_dir = None
            self.clump_count = 0

        # Just enqueue the appropriate chunks; execution is in worker thread
        self._execute_clumped_move(direction, count)


    def _split_distance(self, total, max_step=500):
        """
        Split a total distance into chunks of at most max_step.
        Example: 1100 -> [500, 500, 100]
        """
        chunks = []
        remaining = total
        while remaining > 0:
            step = min(remaining, max_step)
            chunks.append(step)
            remaining -= step
        return chunks

    def _execute_clumped_move(self, direction: str, count: int):
        total_distance = count * self.step_cm
        chunks = self._split_distance(total_distance, max_step=500)

        for dist in chunks:
            self._queue_action(direction, dist)



    @Slot(str)
    def getDroneAction(self, action):
        with self.drone_lock:
            try:
                if action == 'connect':
                    self.tello.connect(wait_for_state=False)
                    battery = self.tello.get_battery()
                    self.connected = True
                    self.logMessage.emit(f"Connected to Tello Drone (Battery: {battery}%)")
                    self.flight_log.insert(0, f"Drone connected (Battery: {battery}%)")
                    self.flightLogUpdated.emit(self.flight_log)
                    return
                elif not self.connected:
                    self.logMessage.emit("Drone not connected. Please connect first.")
                    self.flight_log.insert(0, "Command failed: Drone not connected")
                    self.flightLogUpdated.emit(self.flight_log)
                    return

                def record_action(name, value=None):
                    self.action_log.append((name, value))

                # movement actions: distance is normally 30cm,
                # but can be overridden by the clumper
                if action == 'up':
                    dist = self._movement_distance_override or 30
                    self.tello.move_up(dist)
                    record_action('up', dist)
                    self.logMessage.emit("Moving up")
                    self.flight_log.insert(0, f"Moving up {dist}cm")

                elif action == 'down':
                    dist = self._movement_distance_override or 30
                    self.tello.move_down(dist)
                    record_action('down', dist)
                    self.logMessage.emit("Moving down")
                    self.flight_log.insert(0, f"Moving down {dist}cm")

                elif action == 'forward':
                    dist = self._movement_distance_override or 30
                    self.tello.move_forward(dist)
                    record_action('forward', dist)
                    self.logMessage.emit("Moving forward")
                    self.flight_log.insert(0, f"Moving forward {dist}cm")

                elif action == 'backward':
                    dist = self._movement_distance_override or 30
                    self.tello.move_back(dist)
                    record_action('backward', dist)
                    self.logMessage.emit("Moving backward")
                    self.flight_log.insert(0, f"Moving backward {dist}cm")

                elif action == 'left':
                    dist = self._movement_distance_override or 30
                    self.tello.move_left(dist)
                    record_action('left', dist)
                    self.logMessage.emit("Moving left")
                    self.flight_log.insert(0, f"Moving left {dist}cm")

                elif action == 'right':
                    dist = self._movement_distance_override or 30
                    self.tello.move_right(dist)
                    record_action('right', dist)
                    self.logMessage.emit("Moving right")
                    self.flight_log.insert(0, f"Moving right {dist}cm")

					
                elif action == 'turn_left':
                    self.tello.rotate_counter_clockwise(45)
                    record_action('turn_left', 45)
                    self.logMessage.emit("Rotating left")
                    self.flight_log.insert(0, "Rotating left 45°")
					
                elif action == 'turn_right':
                    self.tello.rotate_clockwise(45)
                    record_action('turn_right', 45)
                    self.logMessage.emit("Rotating right")
                    self.flight_log.insert(0, "Rotating right 45°")
					
                elif action == 'flip_forward':
                    self.tello.flip_forward()
                    record_action('flip_forward')
                    self.logMessage.emit("Flipping forward")
                    self.flight_log.insert(0, "Flipping forward")
					
                elif action == 'flip_back':
                    self.tello.flip_back()
                    record_action('flip_back')
                    self.logMessage.emit("Flipping backward")
                    self.flight_log.insert(0, "Flipping backward")
					
                elif action == 'flip_left':
                    self.tello.flip_left()
                    record_action('flip_left')
                    self.logMessage.emit("Flipping left")
                    self.flight_log.insert(0, "Flipping left")
					
                elif action == 'flip_right':
                    self.tello.flip_right()
                    record_action('flip_right')
                    self.logMessage.emit("Flipping right")
                    self.flight_log.insert(0, "Flipping right")

                elif action == 'takeoff':
                    self.tello.takeoff()
                    self.logMessage.emit("Taking off")
                    self.flight_log.insert(0, "Taking off")
                    self.action_log.clear()
				
                elif action == 'land':
                    self.tello.land()
                    self.logMessage.emit("Landing")
                    self.flight_log.insert(0, "Landing")
                    self.action_log.clear()
					
                elif action == 'go_home':
                    self.go_home()
                    self.logMessage.emit("Going home")
                    self.flight_log.insert(0, "Going home")
					
                elif action == 'stream':
                    if hasattr(self, 'camera_controller'):
                        self.camera_controller.start_camera_stream()
                        self.logMessage.emit("Starting camera stream")
                        self.flight_log.insert(0, "Starting camera stream")
						
                    else:
                        self.logMessage.emit("Camera controller not available")
                        self.flight_log.insert(0, "Camera controller not available")
						
                else:
                    self.logMessage.emit("Unknown action")
                    self.flight_log.insert(0, "Unknown action")

                self.flightLogUpdated.emit(self.flight_log)

            except Exception as e:
                error_msg = f"Error during {action}: {str(e)}"
                self.logMessage.emit(error_msg)
                self.flight_log.insert(0, error_msg)
                self.flightLogUpdated.emit(self.flight_log)
                # If critical error, mark as disconnected
                if "Tello" in str(e) or "timeout" in str(e).lower():
                    self.connected = False

    def go_home(self):
        try:
            self.logMessage.emit("Returning to home by reversing actions...")
            self.flight_log.insert(0, "Returning to home by reversing actions")

            # Take a snapshot so we don't race with new actions
            history = list(self.action_log)  # list of (name, value)

            current_dir = None
            current_total = 0

            def flush_segment():
                nonlocal current_dir, current_total
                if current_dir is None or current_total <= 0:
                    return
                # Use the same 500cm safety limit
                for dist in self._split_distance(current_total, max_step=500):
                    # Queue a single clumped command (direction + distance)
                    self._queue_action(current_dir, dist)
                current_dir = None
                current_total = 0

            # Walk the history in reverse order
            for action, value in reversed(history):
                opp = None

                # Map movement actions to their opposites
                if action == "up":
                    opp = "down"
                elif action == "down":
                    opp = "up"
                elif action == "forward":
                    opp = "backward"
                elif action == "backward":
                    opp = "forward"
                elif action == "left":
                    opp = "right"
                elif action == "right":
                    opp = "left"

                if opp is not None:
                    # Use recorded distance if available, otherwise default step size
                    dist = value if (value is not None) else self.step_cm

                    # If same direction as current segment, accumulate; otherwise flush + start new
                    if current_dir == opp:
                        current_total += dist
                    else:
                        flush_segment()
                        current_dir = opp
                        current_total = dist
                else:
                    # For non-movement actions, flush the current segment and enqueue them individually
                    flush_segment()
                    if action == "turn_left":
                        self._queue_action("turn_right")
                    elif action == "turn_right":
                        self._queue_action("turn_left")
                    elif action == "flip_forward":
                        self._queue_action("flip_back")
                    elif action == "flip_back":
                        self._queue_action("flip_forward")
                    elif action == "flip_left":
                        self._queue_action("flip_right")
                    elif action == "flip_right":
                        self._queue_action("flip_left")
                    # ignore 'takeoff', 'land', 'go_home' themselves, etc.

            # Flush any remaining movement in the last segment
            flush_segment()

            # Finally, land when we get "home"
            self._queue_action("land")

            self.flight_log.insert(0, "Go-home path queued")
            self.logMessage.emit("Go-home path queued.")
            self.flightLogUpdated.emit(self.flight_log)

            # Reset action log for the next flight
            self.action_log.clear()

        except Exception as e:
            error_msg = f"Error during go_home: {str(e)}"
            self.logMessage.emit(error_msg)
            self.flight_log.insert(0, error_msg)
            self.flightLogUpdated.emit(self.flight_log)


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

    @Slot(str)
    def setDataMode(self, mode):
        """
        Set data mode to either synthetic or live based on radio button selection.
        """
        
        self.current_data_mode = mode

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
    print("TabController created")
    manual_nao_controller = ManualNaoController()
    drone_camera_controller = DroneCameraController()

    # Initialize backend before loading QML
    # Queue holds just directions now: e.g. "forward", "left", etc.
    cloud_api = CloudAPI()
    backend = BrainwavesBackend()
    developers = DevelopersAPI()
    shuffler_api = ShufflerAPI()
    engine.rootContext().setContextProperty("tabController", tab_controller)
    engine.rootContext().setContextProperty("backend", backend)
    engine.rootContext().setContextProperty("developersBackend", developers)
    engine.rootContext().setContextProperty("cloudAPI", cloud_api)
    engine.rootContext().setContextProperty("imageModel", [])  # Initialize empty model
    engine.rootContext().setContextProperty("cameraController", backend.camera_controller)
    print("Controllers exposed to QML")
    engine.rootContext().setContextProperty("fileShufflerGui", shuffler_api)  # For file shuffler
    engine.rootContext().setContextProperty("manualNaoController", manual_nao_controller)
    engine.rootContext().setContextProperty("droneCameraController", drone_camera_controller)

    # Load QML
    qml_file = Path(__file__).resolve().parent / "main.qml"

    engine.load(str(qml_file))

    # Start of change : Added Cloud Computing (Transfer Data) functionality 

    if engine.rootObjects():
        cloud_api.set_root_object(engine.rootObjects()[0])
        cloud_api.connect_buttons()
    else:
        print("Error: QML not loaded properly.")

    # End of change : Added Cloud Computing (Transfer Data) functionality 


    # Convert PDFs after engine load
    try:
        backend.convert_pdfs_to_images()
    except Exception as e:
        print(f"Error converting PDFs: {str(e)}")

    # Ensure image model updates correctly
    backend.imagesReady.connect(lambda images: engine.rootContext().setContextProperty("imageModel", images))

    sys.exit(app.exec())



