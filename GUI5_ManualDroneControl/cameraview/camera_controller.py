"""
Camera Controller for Tello Drone
Handles camera streaming and video capture functionality
"""

import os
import cv2
import time
from PySide6.QtCore import QObject, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap
from djitellopy import Tello
import numpy as np


class CameraController(QObject):
    """Python backend that handles camera streaming for Tello Drone."""

    # Signals for QML communication
    frameReady = Signal(str)              # Path to image file for QML
    streamStatusChanged = Signal(bool)    # Stream on/off status
    logMessage = Signal(str)              # Log messages to QML console

    def __init__(self, tello_instance=None):
        super().__init__()
        self.tello = tello_instance
        self.is_streaming = False
        self.current_frame = None

        # QTimer used to periodically process and emit frames
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.process_frame)

    # -------------------------------------------------------------
    # Connection setup
    # -------------------------------------------------------------
    def set_tello_instance(self, tello_instance):
        """Attach the Tello drone instance (if available)."""
        self.tello = tello_instance

    # -------------------------------------------------------------
    # Stream control
    # -------------------------------------------------------------
    @Slot()
    def start_camera_stream(self):
        """Start the camera stream from Tello drone."""
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

            # Start frame processing timer (~30 FPS)
            self.frame_timer.start(33)
            self.logMessage.emit("Camera stream started")

        except Exception as e:
            self.logMessage.emit(f"Error starting camera stream: {e}")
            self.is_streaming = False
            self.streamStatusChanged.emit(False)

    @Slot()
    def stop_camera_stream(self):
        """Stop the camera stream."""
        if not self.is_streaming:
            return

        try:
            self.frame_timer.stop()

            if self.tello:
                self.tello.streamoff()

            self.is_streaming = False
            self.streamStatusChanged.emit(False)
            self.logMessage.emit("Camera stream stopped")

        except Exception as e:
            self.logMessage.emit(f"Error stopping camera stream: {e}")

    # -------------------------------------------------------------
    # Frame processing
    # -------------------------------------------------------------
    def process_frame(self):
        """Process and emit video frames."""
        if not self.is_streaming or not self.tello:
            return

        try:
            frame = self.tello.get_frame_read().frame
            if frame is None:
                return

            # Convert frame for saving and QML display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Use Windows-safe path instead of /tmp
            temp_path = os.path.join(os.getcwd(), "temp_frame.jpg")
            cv2.imwrite(temp_path, frame_rgb)

            # Emit the path for QML
            file_url = f"file:///{temp_path.replace('\\', '/')}"
            self.frameReady.emit(file_url)

        except Exception as e:
            self.logMessage.emit(f"Error processing frame: {e}")

    # -------------------------------------------------------------
    # Photo capture
    # -------------------------------------------------------------
    @Slot()
    def capture_photo(self):
        """Capture a single photo from the stream."""
        if not self.is_streaming or not self.tello:
            self.logMessage.emit("Camera stream not active")
            return

        try:
            frame = self.tello.get_frame_read().frame
            if frame is not None:
                timestamp = int(time.time())
                filename = f"drone_photo_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                self.logMessage.emit(f"Photo captured: {filename}")
            else:
                self.logMessage.emit("No frame available for capture")

        except Exception as e:
            self.logMessage.emit(f"Error capturing photo: {e}")

