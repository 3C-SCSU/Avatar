"""
Camera Controller for Tello Drone
"""

from PySide6.QtCore import QObject, Signal, Slot

class DroneCameraController(QObject):
    logMessage = Signal(str)  # Signal to emit log messages to QML

    def __init__(self):
        super().__init__()

    @Slot()
    def start_camera_stream(self):
        self.logMessage.emit("Start camera clicked")

    @Slot()
    def stop_camera_stream(self):
        self.logMessage.emit("Stop camera clicked")

    @Slot()
    def capture_photo(self):
        self.logMessage.emit("Capture photo clicked")