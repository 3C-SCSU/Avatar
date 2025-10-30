# This Python file uses the following encoding: utf-8
import sys
import os
from pathlib import Path
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from djitellopy import TelloException

# Add parent directory to path to import BrainwavesBackend from GUI5.py
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from GUI5 import BrainwavesBackend


def normalize_bci_label(label):
    """
    Normalize BCI prediction labels to drone action commands.
    Converts various label formats to standardized lowercase action strings.
    
    Args:
        label (str): The raw BCI prediction label
        
    Returns:
        str: Normalized drone action command in lowercase
    """
    if not label:
        return ""
    
    # Convert to lowercase for standardization
    label_lower = label.lower().strip()
    
    # Map label variations to drone actions
    label_mapping = {
        "move forward": "forward",
        "move backward": "backward",
        "move left": "left",
        "move right": "right",
        "move up": "up",
        "move down": "down",
        "take off": "takeoff",
        "landing": "land",
        # Direct mappings (already correct)
        "forward": "forward",
        "backward": "backward",
        "left": "left",
        "right": "right",
        "up": "up",
        "down": "down",
        "takeoff": "takeoff",
        "land": "land",
        "turn_left": "turn_left",
        "turn_right": "turn_right",
    }
    
    normalized = label_mapping.get(label_lower, "")
    if not normalized:
        print(f"Warning: Unknown BCI label '{label}' - ignoring")
    return normalized



class BrainwaveReadingBackend(BrainwavesBackend):
    """
    Extended BrainwavesBackend specifically for the BrainwaveReading module.
    Overrides key methods to integrate BCI label normalization and Tello execution.
    """
    
    def __init__(self, mock_mode=True, bci_connection=None):
        """
        Initialize the BrainwaveReading backend.
        
        Args:
            mock_mode (bool): If True, uses mock predictions for testing.
                            If False, requires actual BCI connection.
            bci_connection: BCI connection object for real brainwave reading.
                          Required when mock_mode=False.
        """
        super().__init__()
        self.mock_mode = mock_mode
        self.bci_connection = bci_connection
        self.mock_predictions = ["forward", "backward", "left", "right", "takeoff", "land"]
        self.mock_index = 0
        
        if not mock_mode and bci_connection is None:
            print("Warning: mock_mode=False but no BCI connection provided. Falling back to mock mode.")
            self.mock_mode = True
    
    @Slot()
    def readMyMind(self):
        """
        Read brainwave data and generate prediction.
        Uses mock data if mock_mode=True, otherwise calls actual BCI system.
        """
        if self.mock_mode:
            # Cycle through mock predictions for testing
            self.current_prediction_label = self.mock_predictions[self.mock_index]
            self.mock_index = (self.mock_index + 1) % len(self.mock_predictions)
            server_name = "Mock Server (Testing)"
        else:
            # TODO: Implement actual BCI prediction
            # Example integration:
            # try:
            #     prediction_response = self.bci_connection.use_brainflow()
            #     self.current_prediction_label = prediction_response["prediction_label"]
            #     server_name = "BCI Server"
            # except Exception as e:
            #     self.logMessage.emit(f"BCI prediction failed: {e}")
            #     return
            raise NotImplementedError(
                "Real BCI prediction not yet implemented. "
                "Set mock_mode=True or provide BCI connection implementation."
            )
        
        # Normalize the label for drone commands
        normalized_label = normalize_bci_label(self.current_prediction_label)
        
        # Update the predictions log
        self.predictions_log.append(
            {
                "count": str(len(self.predictions_log) + 1),
                "server": server_name,
                "label": self.current_prediction_label,
            }
        )
        self.predictionsTableUpdated.emit(self.predictions_log)
        
        # Log to flight log
        mode_indicator = "[MOCK] " if self.mock_mode else ""
        self.flight_log.insert(0, f"{mode_indicator}BCI Prediction: {self.current_prediction_label} → {normalized_label}")
        self.flightLogUpdated.emit(self.flight_log)

    @Slot(str)
    def notWhatIWasThinking(self, manual_action):
        """
        Handle manual action input when BCI prediction is incorrect.
        Normalizes the manual input and executes it on the drone.
        """
        if not manual_action or manual_action.strip() == "":
            self.logMessage.emit("No manual action provided")
            return
            
        # Normalize the manual action
        normalized_action = normalize_bci_label(manual_action)

        if not normalized_action:
            self.logMessage.emit(f"Unknown action '{manual_action}' - no drone command executed")
            return
        
        # Add to predictions log
        self.predictions_log.append(
            {
                "count": "manual",
                "server": "manual",
                "label": manual_action
            }
        )
        self.predictionsTableUpdated.emit(self.predictions_log)
        
        # Execute the manual action on drone
        self.getDroneAction(normalized_action)
        
        # Log the manual override
        self.flight_log.insert(0, f"Manual override: {manual_action} → {normalized_action}")
        self.flightLogUpdated.emit(self.flight_log)

    @Slot()
    def executeAction(self):
        """
        Execute the current BCI prediction on the Tello drone.
        Normalizes the label and sends it to getDroneAction().
        """
        print(f"DEBUG: executeAction() called, current_prediction_label='{self.current_prediction_label}'")
        
        if not self.current_prediction_label:
            msg = "No prediction to execute - click 'Read my mind...' first"
            print(f"DEBUG: {msg}")
            self.logMessage.emit(msg)
            self.flight_log.insert(0, msg)
            self.flightLogUpdated.emit(self.flight_log)
            return
        
        # Normalize the BCI label to drone action format
        normalized_label = normalize_bci_label(self.current_prediction_label)
        print(f"DEBUG: Normalized '{self.current_prediction_label}' → '{normalized_label}'")

        if not normalized_label:
            msg = f"Cannot execute - unknown action '{self.current_prediction_label}'"
            print(f"DEBUG: {msg}")
            self.logMessage.emit(msg)
            self.flight_log.insert(0, msg)
            self.flightLogUpdated.emit(self.flight_log)
            return

        print(f"DEBUG: Calling getDroneAction('{normalized_label}')")
        # Execute on the drone
        self.getDroneAction(normalized_label)
        
        # Update flight log
        log_msg = f"Executed: {self.current_prediction_label} → {normalized_label}"
        print(f"DEBUG: {log_msg}")
        self.flight_log.insert(0, log_msg)
        self.flightLogUpdated.emit(self.flight_log)
        self.logMessage.emit(f"Executed action: {normalized_label}")

    @Slot()
    def connectDrone(self):
        """
        Connect to the Tello drone using the actual getDroneAction method.
        """
        self.getDroneAction('connect')
        self.flight_log.insert(0, "Connecting to drone...")
        self.flightLogUpdated.emit(self.flight_log)

    @Slot()
    def keepDroneAlive(self):
        """
        Send keep-alive signal to maintain Tello connection.
        Queries battery status to keep the connection active.
        """
        if not self.is_connected:
            self.logMessage.emit("Drone not connected. Cannot send keep-alive.")
            self.flight_log.insert(0, "Keep-alive failed: Not connected")
            self.flightLogUpdated.emit(self.flight_log)
            return
            
        try:
            # Query battery to keep connection alive
            battery = self.tello.query_battery()
            self.logMessage.emit(f"Keep-alive sent. Battery: {battery}%")
            self.flight_log.insert(0, f"Keep-alive: Battery {battery}%")
            self.flightLogUpdated.emit(self.flight_log)
        except (AttributeError, ConnectionError, TimeoutError, TelloException) as e:
            self.logMessage.emit(f"Keep-alive error: {e}")
            self.flight_log.insert(0, f"Keep-alive error: {e}")
            self.flightLogUpdated.emit(self.flight_log)

if __name__ == "__main__":
    # Set the Quick Controls style to "Fusion"
    os.environ["QT_QUICK_CONTROLS_STYLE"] = "Fusion"

    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    # Determine if we should use mock mode or real BCI
    # Check for --mock or --real command line argument
    mock_mode = True  # Default to mock mode for safety
    if "--real" in sys.argv:
        mock_mode = False
        print("Starting in REAL BCI mode")
        # TODO: Initialize actual BCI connection here
        # bci_conn = bciConnection(...)
        # backend = BrainwaveReadingBackend(mock_mode=False, bci_connection=bci_conn)
    elif "--mock" in sys.argv or len(sys.argv) == 1:
        print("Starting in MOCK mode (use --real for actual BCI)")
    
    # Create the backend with full Tello integration
    backend = BrainwaveReadingBackend(mock_mode=mock_mode)
    engine.rootContext().setContextProperty("backend", backend)

    # Load the QML file
    qml_file = Path(__file__).resolve().parent / "GUI5_BrainwaveReading.qml"

    # Check if the QML file exists
    if not qml_file.exists():
        print(f"Error: QML file not found at {qml_file}")
        sys.exit(-1)

    # Load the QML file
    engine.load(str(qml_file))

    # Check if the QML engine loaded successfully
    if not engine.rootObjects():
        print("Error: Failed to load QML file")
        sys.exit(-1)

    sys.exit(app.exec())
