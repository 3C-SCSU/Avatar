"""
Camera Controller for Tello Drone
Handles camera streaming and video capture functionality
"""
import cv2
import threading
import time
from PySide6.QtCore import QObject, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap
from djitellopy import Tello
import numpy as np


class CameraController(QObject):
    # Signals for QML communication
    frameReady = Signal(str)  # Send base64 encoded frame to QML
    streamStatusChanged = Signal(bool)  # Stream on/off status
    logMessage = Signal(str)  # Log messages
    
    def __init__(self, tello_instance=None):
        super().__init__()
        self.tello = tello_instance
        self.is_streaming = False
        self.stream_thread = None
        self.current_frame = None
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
            self.frame_timer.stop()
            
            if self.tello:
                self.tello.streamoff()
                
            self.is_streaming = False
            self.streamStatusChanged.emit(False)
            self.logMessage.emit("Camera stream stopped")
            
        except Exception as e:
            self.logMessage.emit(f"Error stopping camera stream: {e}")
    
    def process_frame(self):
        """Process and emit video frames"""
        if not self.is_streaming or not self.tello:
            return
            
        try:
            # Get frame from Tello
            frame = self.tello.get_frame_read().frame
            
            if frame is not None:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to QImage
                height, width, channel = frame_rgb.shape
                bytes_per_line = 3 * width
                qt_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                
                # Convert to base64 for QML
                pixmap = QPixmap.fromImage(qt_image)
                byte_array = pixmap.toImage().bits().asstring(pixmap.toImage().sizeInBytes())
                
                # For QML, we'll use a simpler approach - save frame as temp file
                temp_path = "/tmp/drone_frame.jpg"
                cv2.imwrite(temp_path, frame)
                
                # Emit the file path
                self.frameReady.emit(f"file://{temp_path}")
                
        except Exception as e:
            self.logMessage.emit(f"Error processing frame: {e}")
    
    @Slot()
    def capture_photo(self):
        """Capture a single photo from the stream"""
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