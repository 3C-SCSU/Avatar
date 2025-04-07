import sys
import os
import math
import datetime
from PySide6.QtCore import QUrl, Qt, QSize
from PySide6.QtGui import QColor, QVector3D, QPixmap, QQuaternion, QMatrix4x4
from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QFrame, QGridLayout, QTextEdit
)
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DExtras import Qt3DExtras
from PySide6.Qt3DRender import Qt3DRender

class NaoViewerWidget(QWidget):
    def __init__(self, obj_file_path="Nao/nao.obj", parent=None):
        super().__init__(parent)

        # Set up layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Create the 3D window container
        self.view_container = Qt3DExtras.Qt3DWindow()
        self.view_container.defaultFrameGraph().setClearColor(QColor(50, 50, 50))

        # Create a container widget to hold the 3D view
        self.container = QWidget.createWindowContainer(self.view_container, self)
        self.layout.addWidget(self.container)

        # Create the root entity
        self.root_entity = Qt3DCore.QEntity()

        # Set up the camera
        self.camera = self.view_container.camera()
        self.camera.setPosition(QVector3D(0, 0, 10.0))
        self.camera.setViewCenter(QVector3D(0, 0, 0))

        # Create camera controller
        self.camera_controller = Qt3DExtras.QOrbitCameraController(self.root_entity)
        self.camera_controller.setCamera(self.camera)
        self.camera_controller.setLinearSpeed(50.0)
        self.camera_controller.setLookSpeed(180.0)

        # Add a light
        self.light_entity = Qt3DCore.QEntity(self.root_entity)
        self.light = Qt3DRender.QPointLight(self.light_entity)
        self.light.setColor(QColor(Qt.white))
        self.light.setIntensity(1.0)

        self.light_transform = Qt3DCore.QTransform(self.light_entity)
        self.light_transform.setTranslation(QVector3D(10.0, 10.0, 10.0))
        self.light_entity.addComponent(self.light)
        self.light_entity.addComponent(self.light_transform)

        # Model movement parameters
        self.move_step = 1.0  # Distance to move forward/backward
        self.rotation_step = 90.0  # Degrees to rotate
        self.vertical_step = 1.0  # Distance to move up/down

        # Keep track of model's current position and rotation
        self.model_position = QVector3D(0, 0, 0)
        self.model_rotation_y = 0  # Current rotation around Y axis in degrees

        # Add vertical state counter
        self.vertical_state = 0  # 0 means on ground, 1 means in air (after takeoff)
        self.max_vertical_state = 1  # Maximum allowed vertical state

        # Load the OBJ file
        self.load_obj(obj_file_path)

        # Set the root entity
        self.view_container.setRootEntity(self.root_entity)

    def load_obj(self, obj_file_path):
        # Create mesh entity
        self.mesh_entity = Qt3DCore.QEntity(self.root_entity)

        # Create a mesh component from the OBJ file
        self.mesh = Qt3DRender.QMesh()

        # Convert the file path to a QUrl
        if os.path.isabs(obj_file_path):
            self.mesh.setSource(QUrl.fromLocalFile(obj_file_path))
        else:
            # If relative path, convert to absolute
            abs_path = os.path.abspath(obj_file_path)
            self.mesh.setSource(QUrl.fromLocalFile(abs_path))

        # Create a material
        self.material = Qt3DExtras.QPhongMaterial(self.mesh_entity)
        self.material.setDiffuse(QColor(Qt.blue))

        # Create a transform for positioning and scaling
        self.mesh_transform = Qt3DCore.QTransform()
        self.mesh_transform.setScale(1.0)

        # Initialize position
        self.mesh_transform.setTranslation(self.model_position)

        # Add components to the mesh entity
        self.mesh_entity.addComponent(self.mesh)
        self.mesh_entity.addComponent(self.material)
        self.mesh_entity.addComponent(self.mesh_transform)

        print(f"Loaded OBJ file: {obj_file_path}")

    def update_model_transform(self):
        """Update the model's transform to reflect current position and rotation"""
        # Create rotation quaternion around Y axis
        rotation = QQuaternion.fromAxisAndAngle(QVector3D(0, 1, 0), self.model_rotation_y)

        # Set rotation and translation
        self.mesh_transform.setRotation(rotation)
        self.mesh_transform.setTranslation(self.model_position)

        print(f"Model updated - Position: ({self.model_position.x():.2f}, {self.model_position.y():.2f}, {self.model_position.z():.2f}), Rotation: {self.model_rotation_y}°, Vertical State: {self.vertical_state}")

    def moveForward(self):
        """Move the model forward in its current direction"""
        # Calculate direction vector based on current rotation
        angle_rad = math.radians(self.model_rotation_y)
        direction_x = math.sin(angle_rad)
        direction_z = math.cos(angle_rad)

        # Calculate new position - CORRECTED: Using positive direction for forward
        self.model_position += QVector3D(direction_x * self.move_step, 0, direction_z * self.move_step)

        # Update the model's transform
        self.update_model_transform()
        print(f"Moving forward along direction vector: ({direction_x:.2f}, 0, {direction_z:.2f})")

    def moveBackward(self):
        """Move the model backward from its current direction"""
        # Calculate direction vector based on current rotation
        angle_rad = math.radians(self.model_rotation_y)
        direction_x = math.sin(angle_rad)
        direction_z = math.cos(angle_rad)

        # Calculate new position - CORRECTED: Using negative direction for backward
        self.model_position -= QVector3D(direction_x * self.move_step, 0, direction_z * self.move_step)

        # Update the model's transform
        self.update_model_transform()
        print(f"Moving backward along direction vector: ({-direction_x:.2f}, 0, {-direction_z:.2f})")

    def turnLeft(self):
        """Rotate the model to the left (counter-clockwise)"""
        # CORRECTED: Decrease the Y rotation angle for left turn
        self.model_rotation_y = (self.model_rotation_y - self.rotation_step) % 360

        # Update the model's transform
        self.update_model_transform()
        print(f"Turning left to {self.model_rotation_y}°")

    def turnRight(self):
        """Rotate the model to the right (clockwise)"""
        # CORRECTED: Increase the Y rotation angle for right turn
        self.model_rotation_y = (self.model_rotation_y + self.rotation_step) % 360

        # Update the model's transform
        self.update_model_transform()
        print(f"Turning right to {self.model_rotation_y}°")

    def moveUp(self):
        """Move the model upward along the Y axis (takeoff)"""
        # Only allow takeoff if not already at max height
        if self.vertical_state < self.max_vertical_state:
            # Increase Y position
            self.model_position.setY(self.model_position.y() + self.vertical_step)

            # Increment vertical state
            self.vertical_state += 1

            # Update the model's transform
            self.update_model_transform()
            print(f"Taking off! New vertical state: {self.vertical_state}")
        else:
            print(f"Already at maximum takeoff height (state: {self.vertical_state})")

    def moveDown(self):
        """Move the model downward along the Y axis (land)"""
        # Only allow landing if currently in the air
        if self.vertical_state > 0:
            # Decrease Y position
            self.model_position.setY(self.model_position.y() - self.vertical_step)

            # Decrement vertical state
            self.vertical_state -= 1

            # Update the model's transform
            self.update_model_transform()
            print(f"Landing! New vertical state: {self.vertical_state}")
        else:
            print("Cannot land - already on the ground (state: 0)")

    # Method to get current camera information - useful for debugging
    def getCameraInfo(self):
        position = self.camera.position()
        view_center = self.camera.viewCenter()
        up_vector = self.camera.upVector()

        camera_info = {
            "position": (position.x(), position.y(), position.z()),
            "view_center": (view_center.x(), view_center.y(), view_center.z()),
            "up_vector": (up_vector.x(), up_vector.y(), up_vector.z())
        }

        return camera_info


# Main function to run the application
def main():
    app = QApplication(sys.argv)

    # Create the main window
    main_window = QWidget()
    main_window.setWindowTitle("Nao Viewer")
    main_window.resize(800, 600)

    # Set up horizontal layout for vertical split
    main_layout = QHBoxLayout(main_window)
    main_layout.setSpacing(10)
    main_layout.setContentsMargins(10, 10, 10, 10)

    # Left panel (control panel)
    left_panel = QFrame()
    left_panel.setFrameShape(QFrame.StyledPanel)
    left_panel.setMinimumWidth(300)
    left_layout = QVBoxLayout(left_panel)

    # Add header to the left panel
    header = QLabel("Nao Robot Control Panel")
    header.setAlignment(Qt.AlignCenter)
    header.setStyleSheet("font-size: 18px; margin: 10px;")
    left_layout.addWidget(header)

    # Create grid layout for buttons
    grid_layout = QGridLayout()
    grid_layout.setSpacing(10)

    # Create and add the Nao viewer widget first so we can reference it
    nao_viewer = NaoViewerWidget("Nao/nao.obj")

    # Create button references
    button_refs = {}

    # Function to create control buttons
    def create_button(label, image_name, action_method, row, col):
        button_frame = QFrame()
        button_frame.setStyleSheet("background-color: #1c2733; border-radius: 4px;")
        button_frame.setFixedSize(QSize(140, 140))

        button_layout = QVBoxLayout(button_frame)

        # Add image (you'll need to have these image files in your project directory)
        image_label = QLabel()
        # If you have the images, uncomment the next line:
        image_label.setPixmap(QPixmap(f"{image_name}.png").scaled(90, 90, Qt.KeepAspectRatio))
        image_label.setAlignment(Qt.AlignCenter)
        button_layout.addWidget(image_label)

        # Add text label
        text_label = QLabel(label)
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setStyleSheet("color: white; margin-bottom: 10px;")
        button_layout.addWidget(text_label)

        # Add click functionality
        def on_click():
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Check if this is the Land button and if it should be disabled
            if action_method == "moveDown" and nao_viewer.vertical_state == 0:
                text_field.append(f"Cannot land - already on the ground! at {timestamp}")
                return

            text_field.append(f"{label} Button Clicked! at {timestamp}")

            # Call the appropriate method on the nao_viewer
            method = getattr(nao_viewer, action_method, None)
            if method is not None and callable(method):
                method()

                # Update the Land button appearance based on vertical state
                if action_method == "moveUp" or action_method == "moveDown":
                    update_land_button_state()

        button_frame.mousePressEvent = lambda event: on_click()

        grid_layout.addWidget(button_frame, row, col)
        button_refs[action_method] = button_frame
        return button_frame

    # Create all buttons
    backward_btn = create_button("Backward", "controllerImages/back", "moveBackward", 0, 0)
    forward_btn = create_button("Forward", "controllerImages/forward", "moveForward", 0, 1)
    right_btn = create_button("Right", "controllerImages/right", "turnRight", 1, 0)
    left_btn = create_button("Left", "controllerImages/left", "turnLeft", 1, 1)
    takeoff_btn = create_button("Takeoff", "controllerImages/takeoff", "moveUp", 2, 0)
    land_btn = create_button("Land", "controllerImages/land", "moveDown", 2, 1)

    # Add the grid layout to the left panel
    left_layout.addLayout(grid_layout)

    # MOVED: Add text field for logging button clicks with scrolling
    # Place it after the grid layout (buttons)
    text_field = QTextEdit()
    text_field.setReadOnly(True)  # Make it read-only since it's just for logs
    text_field.setText("Control log:")
    text_field.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 4px; color: black;")
    text_field.setMinimumHeight(100)
    left_layout.addWidget(text_field)

    # Function to update land button appearance
    def update_land_button_state():
        if nao_viewer.vertical_state == 0:
            # Disable appearance
            land_btn.setStyleSheet("background-color: #444444; border-radius: 4px;")
        else:
            # Enable appearance
            land_btn.setStyleSheet("background-color: #1c2733; border-radius: 4px;")

    # Initial button state update
    update_land_button_state()

    # Add the grid layout to the left panel
    left_layout.addLayout(grid_layout)
    left_layout.addStretch(1)

    # Right panel (3D viewer)
    right_panel = QFrame()
    right_panel.setFrameShape(QFrame.StyledPanel)

    # Create and set layout for the right panel
    right_layout = QVBoxLayout(right_panel)
    right_layout.setContentsMargins(0, 0, 0, 0)

    # Add the already created nao_viewer widget to the right panel
    right_layout.addWidget(nao_viewer)

    # Add panels to main layout
    main_layout.addWidget(left_panel, 1)  # 1 part for left panel
    main_layout.addWidget(right_panel, 2)  # 2 parts for right panel (makes it wider)

    # Show the main window
    main_window.show()

    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
