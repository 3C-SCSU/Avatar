import sys
from pathlib import Path  # Import the Path module
from PySide6.QtCore import QObject, Signal, Slot, Property
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

class BrainwaveBackend(QObject):
    # Signals to send data to QML
    predictionUpdated = Signal(str, int)

    def __init__(self):
        super().__init__()
        self._prediction_label = ""
        self._prediction_count = 0

    @Slot()
    def readMind(self):
        """Simulate reading brainwave prediction and updating QML."""
        # Simulate a prediction response
        self._prediction_label = "forward"
        self._prediction_count += 1
        self.predictionUpdated.emit(self._prediction_label, self._prediction_count)

    @Slot(str)
    def executePrediction(self, label):
        """Handle execution of prediction."""
        print(f"Executed action: {label}")

    @Slot(str)
    def manualCommand(self, command):
        """Handle manual command input."""
        print(f"Manual command received: {command}")

    @Slot()
    def keepDroneAlive(self):
        """Simulate keeping the drone connection alive."""
        print("Keep Drone Alive signal sent!")

    @Property(str)
    def predictionLabel(self):
        return self._prediction_label

    @Property(int)
    def predictionCount(self):
        return self._prediction_count

if __name__ == "__main__":
    app = QGuiApplication(sys.argv)

    # QML engine
    engine = QQmlApplicationEngine()

    # Instantiate the backend
    backend = BrainwaveBackend()

    # Expose the backend to QML
    engine.rootContext().setContextProperty("backend", backend)

    # Load QML file
    qml_file = Path(__file__).resolve().parent / "main.qml"
    engine.load(str(qml_file))

    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec())
