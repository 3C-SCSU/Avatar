# This Python file uses the following encoding: utf-8
import sys
import os
from pathlib import Path
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject, Signal, Slot, QThread, Property

import time
import random

VALIDPREDICTIONS = set(["takeoff", "right", "left", "landing", "forward", "backward"])

class Prediction:
    def __init__(self, count: str, server: str, label: str):
        self.count = count
        self.server = server
        self.label = label


class LiveData(QThread):
    data = Signal(Prediction)

    def __init__(self):
        super().__init__()
        self.running = True
        self.thoughts = [
            "takeoff", "right", "left", "landing",
            "forward", "backward"
        ]
        self.count = 1

    def run(self):
        print("Connecting to BCI headset...")
        time.sleep(2)
        print("BCI headset connected. Streaming data...")

        while self.running:
            thought = random.choice(self.thoughts)
            prediction = Prediction(str(self.count), "Prediction Server", thought)
            self.count += 1
            self.data.emit(prediction)
            time.sleep(2)

        print("BCI headset disconnected. Data stream stopped.")

    def stop(self):
        self.running = False
        self.wait()


class BrainwavesBackend(QObject):
    # Define signals to update QML components
    flightLogUpdated = Signal(list)
    predictionsTableUpdated = Signal(list)
    currentPredictionChanged = Signal()  # Signal to notify QML

    def __init__(self):
        super().__init__()
        self.flight_log = []  # List to store flight log entries
        self.predictions_log = []  # List to store prediction records
        self._current_prediction_label = "Not what I was thinking"  # Private attribute for the property
        self.liveData = LiveData()
        self.liveData.data.connect(self.handleAction)

    @Property(str, notify=currentPredictionChanged)
    def current_prediction_label(self):
        return self._current_prediction_label

    @current_prediction_label.setter
    def current_prediction_label(self, value):
        if self._current_prediction_label != value:
            self._current_prediction_label = value
            self.currentPredictionChanged.emit()  # Emit the signal

    @Slot(Prediction)
    def handleAction(self, prediction):
        self.current_prediction_label = prediction.label  # Use the property setter
        self.predictions_log.append({
            "count": prediction.count,
            "server": prediction.server,
            "label": self._current_prediction_label
        })
        self.predictionsTableUpdated.emit(self.predictions_log)

        if self.current_prediction_label in VALIDPREDICTIONS:
            self.executeAction()

    @Slot()
    def readMyMind(self):
        if not self.liveData.isRunning():
            self.liveData.start()
        else:
            self.liveData.stop()

    @Slot(str)
    def notWhatIWasThinking(self, manual_action):
        # Handle manual action input
        prediction = Prediction("manual", "manual", manual_action)
        self.handleAction(prediction)

    @Slot()
    def executeAction(self):
        # Execute the current prediction
        if self.current_prediction_label:
            action_message = f"Executed: {self.current_prediction_label}"
            self.flight_log.insert(0, action_message)
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
    qml_file = Path(__file__).resolve().parent / "main.qml"

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
