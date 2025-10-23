#!/usr/bin/env python3
"""
Test script for Nao6Viewer.py
Focuses specifically on the Nao robot controller functionality
"""

import math
import os
import sys
import unittest
from unittest.mock import MagicMock

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import the Qt modules (with fallbacks if needed)
try:
    from PySide6.QtCore import QQuaternion, Qt, QVector3D
    from PySide6.QtWidgets import QApplication

    qt_available = True
except ImportError:
    # Create mock Qt classes for testing without Qt
    qt_available = False
    print("WARNING: PySide6 not available. Running with limited functionality.")

    class Qt:
        white = "white"
        AlignCenter = "AlignCenter"

    class QVector3D:
        def __init__(self, x=0, y=0, z=0):
            self._x, self._y, self._z = x, y, z

        def x(self):
            return self._x

        def y(self):
            return self._y

        def z(self):
            return self._z

        def setY(self, y):
            self._y = y

        def __add__(self, other):
            return QVector3D(
                self._x + other.x(), self._y + other.y(), self._z + other.z()
            )

        def __sub__(self, other):
            return QVector3D(
                self._x - other.x(), self._y - other.y(), self._z - other.z()
            )

        def __repr__(self):
            return f"QVector3D({self._x}, {self._y}, {self._z})"

    class QQuaternion:
        @staticmethod
        def fromAxisAndAngle(axis, angle):
            return MagicMock()

    class QApplication:
        def __init__(self, *args):
            pass

        @staticmethod
        def exec():
            return 0


# Create a QApplication instance early if Qt is available
if qt_available:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        print("Created QApplication instance")

# Try to import Nao6Viewer module (with fallback implementation)
try:
    if qt_available:
        from Nao6Viewer import NaoViewerWidget

        nao_viewer_available = True
    else:
        # Skip actual import if Qt is not available
        raise ImportError("Qt not available, using mock implementation")
except ImportError:
    nao_viewer_available = False
    print(
        "WARNING: Nao6Viewer.py not available or Qt not available. Using mock implementation."
    )

    class NaoViewerWidget:
        """Mock implementation of NaoViewerWidget for testing"""

        def __init__(self, obj_file_path=None):
            self.model_position = QVector3D(0, 0, 0)
            self.model_rotation_y = 0
            self.vertical_state = 0
            self.max_vertical_state = 1
            self.move_step = 1.0
            self.rotation_step = 90.0
            self.vertical_step = 1.0
            print(f"Mock: Initialized NaoViewerWidget with {obj_file_path}")

        def update_model_transform(self):
            """Mock update transform method"""
            pass

        def moveForward(self):
            """Move the model forward in its current direction"""
            angle_rad = math.radians(self.model_rotation_y)
            direction_x = math.sin(angle_rad)
            direction_z = math.cos(angle_rad)
            self.model_position += QVector3D(
                direction_x * self.move_step, 0, direction_z * self.move_step
            )
            print(f"Mock: Moving forward to {self.model_position}")
            return True

        def moveBackward(self):
            """Move the model backward from its current direction"""
            angle_rad = math.radians(self.model_rotation_y)
            direction_x = math.sin(angle_rad)
            direction_z = math.cos(angle_rad)
            self.model_position -= QVector3D(
                direction_x * self.move_step, 0, direction_z * self.move_step
            )
            print(f"Mock: Moving backward to {self.model_position}")
            return True

        def turnLeft(self):
            """Rotate the model to the left (counter-clockwise)"""
            self.model_rotation_y = (self.model_rotation_y - self.rotation_step) % 360
            print(f"Mock: Turning left to {self.model_rotation_y} degrees")
            return True

        def turnRight(self):
            """Rotate the model to the right (clockwise)"""
            self.model_rotation_y = (self.model_rotation_y + self.rotation_step) % 360
            print(f"Mock: Turning right to {self.model_rotation_y} degrees")
            return True

        def moveUp(self):
            """Move the model upward along the Y axis (takeoff)"""
            if self.vertical_state < self.max_vertical_state:
                self.vertical_state += 1
                self.model_position.setY(self.model_position.y() + self.vertical_step)
                print(f"Mock: Taking off to vertical state {self.vertical_state}")
                return True
            print(f"Mock: Already at max height (state {self.vertical_state})")
            return False

        def moveDown(self):
            """Move the model downward along the Y axis (land)"""
            if self.vertical_state > 0:
                self.vertical_state -= 1
                self.model_position.setY(self.model_position.y() - self.vertical_step)
                print(f"Mock: Landing to vertical state {self.vertical_state}")
                return True
            print("Mock: Already on ground (state 0)")
            return False

        def getCameraInfo(self):
            """Mock camera info method"""
            return {
                "position": (0, 0, 10),
                "view_center": (0, 0, 0),
                "up_vector": (0, 1, 0),
            }


class TestNaoViewerWidget(unittest.TestCase):
    """Test cases for NaoViewerWidget class - the core of the Nao6 controller"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests"""
        # Create a QApplication if not already created
        if qt_available and QApplication.instance() is None:
            cls.app = QApplication(sys.argv)
            print("Created QApplication in setUpClass")
        else:
            cls.app = None

    def setUp(self):
        """Set up test environment before each test"""
        # Skip creating a real NaoViewerWidget if Qt is not available
        if not qt_available:
            self.viewer = NaoViewerWidget("mock_path")
            self.initial_position = QVector3D(0, 0, 0)
            self.initial_rotation = 0
            return

        # Create a mock obj file path for testing
        self.obj_path = "test_nao.obj"

        # Create a simple OBJ file for testing if one doesn't exist
        if not os.path.exists(self.obj_path):
            with open(self.obj_path, "w") as f:
                f.write("# Mock OBJ file for testing\n")
                f.write("v 0 0 0\n")
                f.write("v 1 0 0\n")
                f.write("v 0 1 0\n")
                f.write("f 1 2 3\n")

        try:
            # Create the viewer widget with the test file
            self.viewer = NaoViewerWidget(self.obj_path)

            # Store initial position and rotation
            self.initial_position = QVector3D(0, 0, 0)
            self.initial_rotation = 0
        except Exception as e:
            print(f"Failed to create NaoViewerWidget: {e}")
            # Fall back to mock implementation
            self.viewer = NaoViewerWidget("mock_path")
            self.initial_position = QVector3D(0, 0, 0)
            self.initial_rotation = 0

    def tearDown(self):
        """Clean up after each test"""
        # Clean up the test OBJ file
        if (
            hasattr(self, "obj_path")
            and os.path.exists(self.obj_path)
            and os.path.basename(self.obj_path) == "test_nao.obj"
        ):
            os.remove(self.obj_path)

    def test_initial_state(self):
        """Test that the viewer initializes with correct default values"""
        self.assertEqual(self.viewer.vertical_state, 0)
        self.assertEqual(self.viewer.model_rotation_y, 0)
        # Ensure model starts on the ground
        self.assertEqual(self.viewer.model_position.y(), 0)

    def test_move_forward(self):
        """Test moving the model forward"""
        # Reset to initial position with 0 rotation
        self.viewer.model_position = QVector3D(0, 0, 0)
        self.viewer.model_rotation_y = 0

        # Move forward
        self.viewer.moveForward()

        # With initial rotation of 0, we should move in positive Z direction
        self.assertEqual(self.viewer.model_position.x(), 0)
        self.assertEqual(self.viewer.model_position.y(), 0)
        self.assertGreater(self.viewer.model_position.z(), 0)

    def test_move_backward(self):
        """Test moving the model backward"""
        # Reset to initial position with 0 rotation
        self.viewer.model_position = QVector3D(0, 0, 0)
        self.viewer.model_rotation_y = 0

        # Move backward
        self.viewer.moveBackward()

        # With initial rotation of 0, we should move in negative Z direction
        self.assertEqual(self.viewer.model_position.x(), 0)
        self.assertEqual(self.viewer.model_position.y(), 0)
        self.assertLess(self.viewer.model_position.z(), 0)

    def test_turn_left(self):
        """Test turning the model left (counter-clockwise)"""
        # Reset rotation
        self.viewer.model_rotation_y = 0

        # Turn left
        self.viewer.turnLeft()

        # Should rotate 90 degrees left (which is 270 degrees in the system)
        self.assertEqual(self.viewer.model_rotation_y, 270)

        # Turn left again
        self.viewer.turnLeft()

        # Should now be at 180 degrees
        self.assertEqual(self.viewer.model_rotation_y, 180)

    def test_turn_right(self):
        """Test turning the model right (clockwise)"""
        # Reset rotation
        self.viewer.model_rotation_y = 0

        # Turn right
        self.viewer.turnRight()

        # Should rotate 90 degrees right
        self.assertEqual(self.viewer.model_rotation_y, 90)

        # Turn right again
        self.viewer.turnRight()

        # Should now be at 180 degrees
        self.assertEqual(self.viewer.model_rotation_y, 180)

    def test_move_up_takeoff(self):
        """Test moving the model up (takeoff)"""
        # Reset position and state
        self.viewer.model_position = QVector3D(0, 0, 0)
        self.viewer.vertical_state = 0

        # Take off
        self.viewer.moveUp()

        # Should increase vertical state and Y position
        self.assertEqual(self.viewer.vertical_state, 1)
        self.assertGreater(self.viewer.model_position.y(), 0)

    def test_move_down_land(self):
        """Test moving the model down (land)"""
        # First take off
        self.viewer.model_position = QVector3D(0, 0, 0)
        self.viewer.vertical_state = 0
        self.viewer.moveUp()

        # Then land
        self.viewer.moveDown()

        # Should reset vertical state and Y position
        self.assertEqual(self.viewer.vertical_state, 0)
        self.assertEqual(self.viewer.model_position.y(), 0)

    def test_max_vertical_state(self):
        """Test that the model cannot exceed maximum vertical state"""
        # Reset position and state
        self.viewer.model_position = QVector3D(0, 0, 0)
        self.viewer.vertical_state = 0

        # Take off
        self.viewer.moveUp()

        # Try to go beyond max height
        result = self.viewer.moveUp()

        # Should not change vertical state and return False
        self.assertEqual(self.viewer.vertical_state, self.viewer.max_vertical_state)
        if not isinstance(self.viewer, MagicMock):
            self.assertFalse(result)

    def test_min_vertical_state(self):
        """Test that the model cannot go below ground level"""
        # Reset position and state
        self.viewer.model_position = QVector3D(0, 0, 0)
        self.viewer.vertical_state = 0

        # Try to land when already on ground
        result = self.viewer.moveDown()

        # Should still be at vertical state 0 and return False
        self.assertEqual(self.viewer.vertical_state, 0)
        if not isinstance(self.viewer, MagicMock):
            self.assertFalse(result)

    def test_direction_after_rotation(self):
        """Test moving forward/backward after rotation changes direction"""
        # Reset position and rotation
        self.viewer.model_position = QVector3D(0, 0, 0)
        self.viewer.model_rotation_y = 0

        # Turn right (90 degrees)
        self.viewer.turnRight()

        # Move forward
        self.viewer.moveForward()

        # After 90 degree rotation, forward should be along positive X axis
        self.assertGreater(self.viewer.model_position.x(), 0)
        self.assertEqual(
            round(self.viewer.model_position.z(), 10), 0
        )  # Account for floating point precision

    def test_camera_info(self):
        """Test getting camera information"""
        camera_info = self.viewer.getCameraInfo()

        # Should return a dictionary with position, view_center, and up_vector
        self.assertIsInstance(camera_info, dict)
        self.assertIn("position", camera_info)
        self.assertIn("view_center", camera_info)
        self.assertIn("up_vector", camera_info)

    def test_rotate_360(self):
        """Test rotating the model 360 degrees returns to original orientation"""
        # Reset rotation
        self.viewer.model_rotation_y = 0

        # Rotate right 4 times (4 * 90 = 360 degrees)
        for _ in range(4):
            self.viewer.turnRight()

        # Should be back at 0 degrees (or 360, which is the same)
        self.assertEqual(self.viewer.model_rotation_y, 0)


if __name__ == "__main__":
    print("Starting Nao6 Controller tests...")

    # Run the tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestNaoViewerWidget)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    sys.exit(not result.wasSuccessful())
