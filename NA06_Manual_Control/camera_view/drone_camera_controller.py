"""
Camera Controller for Tello Drone
Handles camera streaming and video capture functionality
"""

from PySide6.QtCore import QObject

class DroneCameraController(QObject):
    def __init__(self):
        super().__init__()
