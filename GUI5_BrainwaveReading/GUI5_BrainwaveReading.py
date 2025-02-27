# This Python file uses the following encoding: utf-8
import sys
import os
from pathlib import Path
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject, Signal, Slot

class BrainwavesBackend(QObject):
    # Define signals to update QML components
    flightLogUpdated = Signal(list)
    predictionsTableUpdated = Signal(list)

    def __init__(self):
        super().__init__()
        self.flight_log = []  # List to store flight log entries
        self.predictions_log = []  # List to store prediction records
        self.current_prediction_label = ""

    @Slot()
    def readMyMind(self):
        # Mock function to simulate brainwave reading
        self.current_prediction_label = "Move Forward"
        # Update the predictions log
        self.predictions_log.append({
            "count": "1",
            "server": "Prediction Server",
            "label": self.current_prediction_label
        })
        self.predictionsTableUpdated.emit(self.predictions_log)

    @Slot(str)
    def notWhatIWasThinking(self, manual_action):
        # Handle manual action input
        self.predictions_log.append({
            "count": "manual",
            "server": "manual",
            "label": manual_action
        })
        self.predictionsTableUpdated.emit(self.predictions_log)

    @Slot()
    def executeAction(self):
        # Execute the current prediction
        if self.current_prediction_label:
            self.flight_log.insert(0, f"Executed: {self.current_prediction_label}")
            self.flightLogUpdated.emit(self.flight_log)

    @Slot()
    def connectDrone(self):
        # Mock function to simulate drone connection
        self.flight_log.insert(0, "Drone connected.")
        self.flightLogUpdated.emit(self.flight_log)

    @Slot()
    def keepDroneAlive(self):
        # Mock function to simulate sending keep-alive signal
        self.flight_log.insert(0, "Keep alive signal sent.")
        self.flightLogUpdated.emit(self.flight_log)

if __name__ == "__main__":
    # Set the Quick Controls style to "Fusion"
    os.environ["QT_QUICK_CONTROLS_STYLE"] = "Fusion"

    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    # Load the QML file
    qml_file = Path(__file__).resolve().parent / "GUI5_BrainwaveReading.qml"

    # Check if the QML file exists
    if not qml_file.exists():
        sys.exit(-1)

    # Load the QML file
    engine.load(str(qml_file))

    # Check if the QML engine loaded successfully
    if not engine.rootObjects():
        sys.exit(-1)

    # Create and set the backend context
    backend = BrainwavesBackend()
    engine.rootContext().setContextProperty("backend", backend)

    sys.exit(app.exec())
