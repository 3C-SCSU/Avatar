"""
Camera Controller for Tello Drone
Handles camera streaming and video capture functionality
"""
from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal
import base64
import threading
import time


class DroneCameraController(QObject):
    # emit base64-encoded image data URIs or simple strings QML can use as Image.source
    videoFrame = pyqtSignal(str)      # e.g. "data:image/jpeg;base64,...."
    logMessage = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._thread = None

    @pyqtSlot()
    def start(self):
        if self._running:
            return
        self._running = True
        self.logMessage.emit("DroneCameraController: start")
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    @pyqtSlot()
    def stop(self):
        self._running = False
        self.logMessage.emit("DroneCameraController: stop")

    def _capture_loop(self):
        # Replace this stub with real frame capture (from tello/opencv/etc.)
        while self._running:
            try:
                # Example: create a tiny placeholder JPEG (or get frames and encode)
                # Here we emit a timestamp string for debugging; replace with actual base64 image data
                payload = f"placeholder-frame-{int(time.time())}"
                self.videoFrame.emit(payload)
                time.sleep(0.2)
            except Exception as e:
                self.logMessage.emit(f"DroneCameraController error: {e}")
                self._running = False