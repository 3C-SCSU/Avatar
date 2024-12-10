# This Python file uses the following encoding: utf-8
import sys
import os
from pathlib import Path
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject, Signal, Slot


class BrainwavesBackend(QObject):
    """
    Backend class that acts as a bridge between QML and Python logic.
    Handles logic for brainwave simulation, drone actions, and communication with QML.
    """
    
    # Define signals to communicate data with QML components
    flightLogUpdated = Signal(list)  # Signal to send flight log updates
    predictionsTableUpdated = Signal(list)  # Signal to send predictions table updates

    def __init__(self):
        super().__init__()
        
        # Initialize flight log and prediction log lists
        self.flight_log = []  # List to store flight log entries
        self.predictions_log = []  # List to store prediction records
        self.current_prediction_label = ""

    @Slot()
    def readMyMind(self):
        """
        Simulate a brainwave reading by generating mock predictions.
        This is called when a user clicks the 'Read My Mind' button in QML.
        """
        self.current_prediction_label = "Move Forward"  # Mocked prediction logic
        # Add new prediction data to the predictions log
        self.predictions_log.append({
            "count": len(self.predictions_log) + 1,
            "server": "Prediction Server",
            "label": self.current_prediction_label
        })
        # Notify QML of the updated predictions
        self.predictionsTableUpdated.emit(self.predictions_log)

    @Slot(str)
    def notWhatIWasThinking(self, manual_action):
        """
        Handle manual input from the user. Simulate a user-driven prediction entry.
        :param manual_action: Text string inputted by the user for manual action.
        """
        # Log the manual input into the predictions log
        self.predictions_log.append({
            "count": len(self.predictions_log) + 1,
            "server": "manual",
            "label": manual_action
        })
        # Notify QML of the updated predictions table
        self.predictionsTableUpdated.emit(self.predictions_log)

    @Slot()
    def executeAction(self):
        """
        Simulate executing a drone action based on the current prediction.
        Logs the action into the flight log.
        """
        if self.current_prediction_label:
            # Log the action into the flight log
            self.flight_log.insert(0, f"Executed: {self.current_prediction_label}")
            # Notify QML of the updated flight log
            self.flightLogUpdated.emit(self.flight_log)

    @Slot()
    def connectDrone(self):
        """
        Simulate a connection to a drone.
        Logs a 'Drone Connected' message into the flight log.
        """
        self.flight_log.insert(0, "Drone connected.")
        self.flightLogUpdated.emit(self.flight_log)

    @Slot()
    def keepDroneAlive(self):
        """
        Simulate sending a keep-alive signal to the drone.
        Logs a 'Keep alive' message into the flight log.
        """
        self.flight_log.insert(0, "Keep alive signal sent.")
        self.flightLogUpdated.emit(self.flight_log)


if __name__ == "__main__":
    # Set the Quick Controls style to "Fusion"
    os.environ["QT_QUICK_CONTROLS_STYLE"] = "Fusion"

    # Start the QML application
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    # Determine the QML file's path
    qml_file = Path(__file__).resolve().parent / "GUI5_BrainwaveReading.qml"

    # Ensure the QML file exists
    if not qml_file.exists():
        print(f"Error: {qml_file} does not exist.")
        sys.exit(-1)

    # Load the QML file
    engine.load(str(qml_file))

    # Ensure the QML was successfully loaded
    if not engine.rootObjects():
        print("Error: QML application engine failed to load.")
        sys.exit(-1)

    # Create and set the backend for communication with QML
    backend = BrainwavesBackend()
    engine.rootContext().setContextProperty("backend", backend)

    # Execute the main application loop
    sys.exit(app.exec())
