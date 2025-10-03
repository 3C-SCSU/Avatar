"""
Camera Controller for Tello Drone
Handles camera streaming and video capture functionality
"""
from PySide6.QtCore import QObject


class CameraController(QObject):
    
    def __init__(self, tello_instance=None):
        super().__init__()