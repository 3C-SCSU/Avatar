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

    # Signals to communicate updates with QML components
    flightLogUpdated = Signal(list)  # Updates flight log data in QML
    predictionsTableUpdated = Signal(list)  # Updates predictions table in QML

    def __init__(self):
        super().__init__()

        # Initialize internal data structures
        self.flight_log = []  # List to store flight log entries
        self.predictions_log = []  # List to store prediction records
        self.current_prediction_label = ""

    @Slot()
    def readMyMind(self):
        """
        Simulates brainwave reading and generates a mock prediction.
        Triggered when the 'Read My Mind' button is clicked in QML.
        """
        self.current_prediction_label = "Move Forward"  # Example mock prediction
        self.predictions_log.append({
            "count": len(self.predictions_log) + 1,
            "server": "Prediction Server",
            "label": self.current_prediction_label
        })

        # Notify QML about the updated predictions log
        self.predictionsTableUpdated.emit(self.predictions_log)

    @Slot(str)
    def notWhatIWasThinking(self, manual_action):
        """
        Handles manual input for user-defined predictions.
        :param manual_action: Text input provided by the user.
        """
        self.predictions_log.append({
            "count": len(self.predictions_log) + 1,
            "server": "manual",
            "label": manual_action
        })

        # Notify QML about the updated predictions table
        self.predictionsTableUpdated.emit(self.predictions_log)

    @Slot()
    def executeAction(self):
        """
        Executes a drone action based on the current prediction label.
        Logs the executed action in the flight log.
        """
        if self.current_prediction_label:
            self.flight_log.insert(0, f"Executed: {self.current_prediction_label}")
            self.flightLogUpdated.emit(self.flight_log)

    @Slot()
    def connectDrone(self):
        """
        Simulates connecting to a drone.
        Logs a connection message in the flight log.
        """
        self.flight_log.insert(0, "Drone connected.")
        self.flightLogUpdated.emit(self.flight_log)

    @Slot()
    def keepDroneAlive(self):
        """
        Simulates sending a keep-alive signal to the drone.
        Logs the action in the flight log.
        """
        self.flight_log.insert(0, "Keep alive signal sent.")
        self.flightLogUpdated.emit(self.flight_log)


if __name__ == "__main__":
    # Set the Quick Controls style to "Fusion" for a consistent look
    os.environ["QT_QUICK_CONTROLS_STYLE"] = "Fusion"

    # Initialize the QML application
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    # Determine the path to the QML file
    qml_file = Path(__file__).resolve().parent / "GUI5_BrainwaveReading.qml"

    # Validate the QML file's existence
    if not qml_file.exists():
        print(f"Error: {qml_file} does not exist.")
        sys.exit(-1)

    # Load the QML file into the engine
    engine.load(str(qml_file))

    # Ensure the QML was loaded successfully
    if not engine.rootObjects():
        print("Error: Failed to load QML application engine.")
        sys.exit(-1)

    # Initialize the backend and expose it to QML
    backend = BrainwavesBackend()
    engine.rootContext().setContextProperty("backend", backend)

    # Start the main application loop
    sys.exit(app.exec())
