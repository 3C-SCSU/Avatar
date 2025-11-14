"""
Camera Controller for Tello Drone
"""

import cv2, base64, time, threading
from PySide6.QtCore import QObject, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap
from djitellopy import Tello
import numpy as np

class DroneCameraController(QObject):
    frameReady = Signal(str)  # Send base64 encoded frame to QML
    streamStatusChanged = Signal(bool)  # Notify QML about stream status
    logMessage = Signal(str)  # Signal to emit log messages to QML

    def __init__(self, tello_instance=None):
        super().__init__()
        self.tello = tello_instance
        self.is_streaming = False
        self.frame_reader = None
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.process_frame)

    def set_tello_instance(self, tello_instance):
        """Set the Tello drone instance"""
        self.tello = tello_instance

    @Slot()
    def start_camera_stream(self):
        """Start the camera stream from Tello drone"""
        if self.tello is None:
            self.logMessage.emit("Error: Tello drone not connected")
            return

        if self.is_streaming:
            self.logMessage.emit("Camera stream already running")
            return

        try:
            # Start Tello video stream
            self.tello.streamon()
            self.is_streaming = True
            self.streamStatusChanged.emit(True)

            # Get frame reader
            self.frame_reader = self.tello.get_frame_read()

            # Start frame processing timer
            self.frame_timer.start(33)  # ~30 FPS

            self.logMessage.emit("Camera stream started")

        except Exception as e:
            self.logMessage.emit(f"Error starting camera stream: {e}")
            self.is_streaming = False
            self.streamStatusChanged.emit(False)

    @Slot()
    def stop_camera_stream(self):
        """Stop the camera stream"""
        if not self.is_streaming:
            return

        try:
            self.frame_reader = None
            self.frame_timer.stop()

            if self.tello:
                self.tello.streamoff()

            self.is_streaming = False
            self.streamStatusChanged.emit(False)
            self.logMessage.emit("Camera stream stopped")

        except Exception as e:
            self.logMessage.emit(f"Error stopping camera stream: {e}")

    @Slot()
    def capture_photo(self):
        """Capture a single photo from the stream"""
        if not self.is_streaming or not self.tello:
            self.logMessage.emit("Camera stream not active")
            return

        try:
            frame = self.frame_reader.frame
            if frame is not None:
                timestamp = int(time.time())
                filename = f"drone_photo_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                self.logMessage.emit(f"Photo captured: {filename}")
            else:
                self.logMessage.emit("No frame available for capture")

        except Exception as e:
            self.logMessage.emit(f"Error capturing photo: {e}")

    def process_frame(self):
        """Process and emit video frames"""
        if not self.is_streaming or not self.frame_reader:
            return

        frame = self.frame_reader.frame
        if frame is None: return

        try:
            # Get frame from Tello
            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            self.frameReady.emit(f"data:image/jpeg;base64,{jpg_as_text}")

        except Exception as e:
            self.logMessage.emit(f"Frame error: {e}")