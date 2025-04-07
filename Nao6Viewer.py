import sys
import os
import math
import datetime
from PySide6.QtCore import QUrl, Qt, QSize, QPropertyAnimation, QTimer
from PySide6.QtGui import QColor, QVector3D, QPixmap, QQuaternion, QMatrix4x4
from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QFrame, QGridLayout, QTextEdit
)
from PySide6.Qt3DCore import Qt3DCore
from PySide6.Qt3DExtras import Qt3DExtras
from PySide6.Qt3DRender import Qt3DRender

class NaoViewerWidget(QWidget):
    def __init__(self, obj_file_path="Nao/Nao.obj", mtl_file_path="Nao/Nao.obj", parent=None):
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
        self.camera.setPosition(QVector3D(0, 0, 12.0))
        self.camera.setViewCenter(QVector3D(0, 2, 0))

        # Create camera controller
        self.camera_controller = Qt3DExtras.QOrbitCameraController(self.root_entity)
        self.camera_controller.setCamera(self.camera)
        self.camera_controller.setLinearSpeed(50.0)
        self.camera_controller.setLookSpeed(180.0)

        # Add a light
        self.light_entity = Qt3DCore.QEntity(self.root_entity)
        self.light = Qt3DRender.QPointLight(self.light_entity)
        self.light.setColor(QColor(Qt.white))
        self.light.setIntensity(0.6)

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

        self._animation_frame = 0
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._update_animation_frame)

        # Load the robot
        self.load_robot(mtl_file_path)

        # Set the root entity
        self.view_container.setRootEntity(self.root_entity)

    def parse_mtl_file(self, mtl_file_path):
        """
        Parses the MTL file and extracts material properties.
        :param mtl_file_path: Path to the MTL file.
        :return: Dictionary of material properties.
        """
        material_properties = {}
        
        try:
            with open(mtl_file_path, 'r') as mtl_file:
                current_material = None
                for line in mtl_file:
                    line = line.strip()
                    
                    if line.startswith('newmtl '):
                        current_material = line.split(' ')[1]
                        material_properties[current_material] = {}
                    
                    # Parse specular exponent (Ns)
                    elif line.startswith('Ns '):  
                        material_properties[current_material]['Ns'] = float(line.split(' ')[1])

                    # Parse ambient color (Ka)
                    elif line.startswith('Ka '):  
                        color_values = line.split(' ')[1:]
                        material_properties[current_material]['Ka'] = [float(v) for v in color_values]

                    # Parse diffuse color (Kd)
                    elif line.startswith('Kd '):  
                        color_values = line.split(' ')[1:]
                        material_properties[current_material]['Kd'] = [float(v) for v in color_values]
                    
                    # Parse specular color (Ks)
                    elif line.startswith('Ks '):  
                        color_values = line.split(' ')[1:]
                        material_properties[current_material]['Ks'] = [float(v) for v in color_values]
                    
                    # Parse emissive color (Ke)
                    elif line.startswith('Ke '):  
                        color_values = line.split(' ')[1:]
                        material_properties[current_material]['Ke'] = [float(v) for v in color_values]
                    
                    # Parse optical density (Ni)
                    elif line.startswith('Ni '):  
                        material_properties[current_material]['Ni'] = float(line.split(' ')[1])

        except FileNotFoundError:
            print(f"ERROR: MTL file does not exist: {mtl_file_path}")
            return {}

        return material_properties

    def parse_obj_file(self, obj_file_path):
        """
        Parses the OBJ file and extracts vertices, normals, textures, faces, and materials.
        :param obj_file_path: Path to the OBJ file.
        :return: Parsed vertices, normals, textures, faces, and materials.
        """
        vertices = []
        normals = []
        textures = []
        faces = []
        materials = []
        current_material = None
        current_face_index = 0

        # Check if the file exists
        if not os.path.exists(obj_file_path):
            print(f"ERROR: OBJ file does not exist: {obj_file_path}")
            return [], [], [], [], []

        try:
            with open(obj_file_path, 'r') as obj_file:
                for line in obj_file:
                    line = line.strip()
                    if line.startswith('v '):  # Vertex positions
                        parts = line.split()[1:]
                        vertices.append([float(x) for x in parts])
                    elif line.startswith('vn '):  # Vertex normals
                        parts = line.split()[1:]
                        normals.append([float(x) for x in parts])
                    elif line.startswith('vt '):  # Texture coordinates
                        parts = line.split()[1:]
                        textures.append([float(x) for x in parts])
                    elif line.startswith('f '):  # Faces
                        face_data = line.split()[1:]
                        face_vertices = []
                        for face in face_data:
                            indices = face.split('/')
                            vertex_idx = int(indices[0]) - 1  # OBJ indices start at 1
                            texture_idx = int(indices[1]) - 1 if len(indices) > 1 and indices[1] else None
                            normal_idx = int(indices[2]) - 1 if len(indices) > 2 else None
                            face_vertices.append((vertex_idx, texture_idx, normal_idx))
                        faces.append(face_vertices)
                    elif line.startswith('usemtl '):  # Material
                        material_name = line.split()[1]
                        
                        if current_material is not None:
                            materials.append((current_material, current_face_index))
                        
                        current_material = material_name
                        current_face_index = len(faces)

        except FileNotFoundError:
            print(f"ERROR: OBJ file does not exist: {obj_file_path}")
            return [], [], [], [], []

        print(f"Parsed OBJ file: {obj_file_path}")
        print(f"  Vertices: {len(vertices)}")
        print(f"  Normals: {len(normals)}")
        print(f"  Textures: {len(textures)}")
        print(f"  Faces: {len(faces)}")
        print(f"  Materials: {len(materials)}")
        
        return vertices, normals, textures, faces, materials

    def get_first_material_name_from_obj(self, obj_file_path):
        """Extracts the first material name from an .obj file."""
        try:
            with open(obj_file_path, 'r') as obj_file:
                for line in obj_file:
                    # Look for the line containing 'usemtl', which defines the material
                    if line.startswith('usemtl'):
                        # Extract the material name (after the 'usemtl' keyword)
                        material_name = line.split()[1]
                        return material_name
        except FileNotFoundError:
            print(f"Error: The file '{obj_file_path}' was not found.")
        except Exception as e:
            print(f"Error: {e}")
        
        # If no material is found
        return None
    
    def process_materials_for_files(self, file_paths):
        """Processes a list of .obj file paths and returns the first material name for each file."""
        material_names = {}
        
        for path in file_paths:
            material_name = self.get_first_material_name_from_obj(path)
            material_names[path] = material_name
            
        return material_names
    
    def find_corresponding_material(self, material, file_name):
        """Searches a list of material names for a specific material name. Returns Qt3DExtras.QPhongMaterial"""
        
        for nao6_materials in self.material_list:
            material_name_without_decimals = material.split('.')[0]
            if nao6_materials[0] == material_name_without_decimals:
                return(nao6_materials[1])
            
        print(f"No corresponding material found for {file_name}. Using default color {self.material_list[10][0]}")
        return self.material_list[10][1]
    
    def load_robot(self, mtl_file_path):
        # All obj files rendered
        file_paths = [
            "Nao/nao6_turn_right_animation/face_forward/gray/nao6_right_gray0001.obj",
            "Nao/nao6_turn_right_animation/face_forward/orange/nao6_right_orange0001.obj",
            "Nao/nao6_turn_right_animation/face_forward/teal/nao6_right_teal0001.obj",
            "Nao/nao6_turn_right_animation/face_forward/white/nao6_right_white0001.obj",
        ]

        # Create paremt mesh entity
        self.mesh_entity = Qt3DCore.QEntity(self.root_entity)
        self.mesh_transform = Qt3DCore.QTransform()

        # Create mesh entity for each part
        self.gray_entity = Qt3DCore.QEntity(self.mesh_entity)
        self.gray_transform = Qt3DCore.QTransform()
        self.orange_entity = Qt3DCore.QEntity(self.mesh_entity)
        self.orange_transform = Qt3DCore.QTransform()
        self.teal_entity = Qt3DCore.QEntity(self.mesh_entity)
        self.teal_transform = Qt3DCore.QTransform()
        self.white_entity = Qt3DCore.QEntity(self.mesh_entity)
        self.white_transform = Qt3DCore.QTransform()

        # Create a mesh component from the OBJ file
        self.mesh = Qt3DRender.QMesh()
        self.gray_mesh = Qt3DRender.QMesh()
        self.gray_mesh.setSource(QUrl.fromLocalFile("Nao/nao6_turn_right_animation/face_forward/gray/nao6_right_gray0001.obj"))
        self.orange_mesh = Qt3DRender.QMesh()
        self.orange_mesh.setSource(QUrl.fromLocalFile("Nao/nao6_turn_right_animation/face_forward/orange/nao6_right_orange0001.obj"))
        self.teal_mesh = Qt3DRender.QMesh()
        self.teal_mesh.setSource(QUrl.fromLocalFile("Nao/nao6_turn_right_animation/face_forward/teal/nao6_right_teal0001.obj"))
        self.white_mesh = Qt3DRender.QMesh()
        self.white_mesh.setSource(QUrl.fromLocalFile("Nao/nao6_turn_right_animation/face_forward/white/nao6_right_white0001.obj"))

        entity_dict = {
            "gray": self.gray_entity,
            "orange": self.orange_entity,
            "teal": self.teal_entity,
            "white": self.white_entity
        }

        # Load material properties from .mtl file
        material_properties = self.parse_mtl_file(mtl_file_path)

        # Apply different materials based on the .obj file's usemtl section
        # For each part in the model, we'll create a new material and apply it
        self.material_list = []
        
        # Iterate through the parsed materials and create Qt3D materials
        for material_name, material_data in material_properties.items():
            material = Qt3DExtras.QPhongMaterial(self.mesh_entity)

            # Set ambient color (Ka)
            ambient_color = material_data.get('Ka', [0.2, 0.2, 0.2])
            material.setAmbient(QColor(*[int(c * 255) for c in ambient_color]))  # Convert to 0-255 range

            # Set diffuse color (Kd)
            diffuse_color = material_data.get('Kd', [1.0, 1.0, 1.0])
            material.setDiffuse(QColor(*[int(c * 255) for c in diffuse_color]))  # Convert to 0-255 range

            # Set specular color (Ks)
            specular_color = material_data.get('Ks', [0.8, 0.8, 0.8])
            material.setSpecular(QColor(*[int(c * 255) for c in specular_color]))  # Convert to 0-255 range
            
            # Set specular exponent (Ns)
            specular_exponent = material_data.get('Ns', 0.0)
            material.setShininess(specular_exponent / 1000.0)  # Normalized to 0-1 range
            
            # # Print the material properties (Ka, Kd, Ks, Ns)
            # print(f"Material name: {material_name}")
            # print(f"  Ka (Ambient Color): {ambient_color}")
            # print(f"  Kd (Diffuse Color): {diffuse_color}")
            # print(f"  Ks (Specular Color): {specular_color}")
            # print(f"  Ns (Specular Exponent): {specular_exponent}")

            # Add this material to the list
            self.material_list.append((material_name, material))


        # Iterate over file paths and add the corresponding component based on the material name
        for color_name, entity in entity_dict.items():
            # Check if the file path contains the part name
            if color_name == "gray":
                print(f"Material used on {color_name} is {self.material_list[3][0]}")
                entity.addComponent(self.material_list[3][1])
            if color_name == "orange":
                print(f"Material used on {color_name} is {self.material_list[5][0]}")
                entity.addComponent(self.material_list[5][1])
            if color_name == "teal":
                print(f"Material used on {color_name} is {self.material_list[6][0]}")
                entity.addComponent(self.material_list[6][1])
            if color_name == "white":
                print(f"Material used on {color_name} is {self.material_list[8][0]}")
                entity.addComponent(self.material_list[8][1])


        self.mesh_entity.addComponent(self.mesh)
        self.mesh_entity.addComponent(self.mesh_transform)

        # Iterate through each entity in entity_dict
        for entity_name, entity_info in entity_dict.items():
            mesh_loader_attr = f"{entity_name}_mesh"
            transform_attr = f"{entity_name}_transform"
            # Get the mesh loader and transform components using dynamic attribute access
            mesh_loader = getattr(self, mesh_loader_attr)
            transform = getattr(self, transform_attr)
            
            # Add components for the current entity
            entity_info.addComponent(mesh_loader)
            entity_info.addComponent(transform)

        print(f"Loaded robot")

    def _update_animation_frame(self):
        """Timer callback to update animation frames"""
        if not self._animation_frames_white:
            self._animation_timer.stop()
            return

        if self._animation_frame < len(self._animation_frames_white):
            frame_path = self._animation_frames_gray[self._animation_frame]
            mtl_path = frame_path.replace(".obj", ".mtl")
            self._load_obj_frame(frame_path, mtl_path, self.gray_mesh)

            frame_path = self._animation_frames_orange[self._animation_frame]
            mtl_path = frame_path.replace(".obj", ".mtl")
            self._load_obj_frame(frame_path, mtl_path, self.orange_mesh)

            frame_path = self._animation_frames_teal[self._animation_frame]
            mtl_path = frame_path.replace(".obj", ".mtl")
            self._load_obj_frame(frame_path, mtl_path, self.teal_mesh)
            
            frame_path = self._animation_frames_white[self._animation_frame]
            mtl_path = frame_path.replace(".obj", ".mtl")
            self._load_obj_frame(frame_path, mtl_path, self.white_mesh)

            self._animation_frame += 1
        else:
            # Animation completed
            self._animation_timer.stop()
            self._animation_frame = 0
            self._animation_frames = []

    def _load_obj_frame(self, obj_path, mtl_path=None, mesh=None):
        """Load a new OBJ frame to the 3D scene"""
        try:
            # Update the mesh source
            mesh.setSource(QUrl.fromLocalFile(obj_path))

            # Update material if MTL path is provided
            if mtl_path and os.path.exists(mtl_path):
                material_properties = self.parse_mtl_file(mtl_path)
                if material_properties:
                    # Here we could update materials but for simplicity we'll keep the default
                    pass

            return True
        except Exception as e:
            print(f"Error loading OBJ frame: {e}")
            return False

    def _find_animation_frames(self, animation_folder):
        """Find all OBJ files for an animation sequence"""
        frames = []
        try:
            if os.path.isdir(animation_folder):
                # Get all OBJ files in the folder
                for file in sorted(os.listdir(animation_folder)):
                    if file.lower().endswith('.obj'):
                        frames.append(os.path.join(animation_folder, file))
        except Exception as e:
            print(f"Error finding animation frames: {e}")

        return frames

    def _play_obj_animation(self, animation_folder, frame_delay=100):
        """Play an OBJ animation sequence"""
        # Stop any current animation
        if self._animation_timer.isActive():
            self._animation_timer.stop()

        # Find all frames
        self._animation_frames_gray = self._find_animation_frames(animation_folder+"gray/")
        self._animation_frames_orange = self._find_animation_frames(animation_folder+"orange/")
        self._animation_frames_teal = self._find_animation_frames(animation_folder+"teal/")
        self._animation_frames_white = self._find_animation_frames(animation_folder+"white/")

        self._update_animation_frame()

        if not self._animation_frames_white:
            print(f"No animation frames found in {animation_folder}")
            return False

        # Reset animation state
        self._animation_frame = 0

        # Start the animation timer
        self._animation_timer.setInterval(frame_delay)
        self._animation_timer.start()

        return True
    
    def _create_movement_animation(self, property_name, start_value, end_value, duration=1000):
        if self._current_animation:
            self._current_animation.stop()

        animation = QPropertyAnimation(self.objTransform)
        animation.setTargetObject(self.controller)
        animation.setPropertyName(property_name.encode())
        animation.setStartValue(start_value)
        animation.setEndValue(end_value)
        animation.setDuration(duration)

        self._current_animation = animation
        return animation


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

        # Start OBJ animation sequence for turning left
        animation_folder = f"Nao/nao6_turn_left_animation/face_forward/"
        self._play_obj_animation(animation_folder)

        # Update the model's transform
        self.update_model_transform()
        print(f"Turning left to {self.model_rotation_y}°")

    def turnRight(self):
        """Rotate the model to the right (clockwise)"""
        # CORRECTED: Increase the Y rotation angle for right turn
        self.model_rotation_y = (self.model_rotation_y + self.rotation_step) % 360

        # Start OBJ animation sequence for turning right
        animation_folder = f"Nao/nao6_turn_right_animation/face_forward/"
        self._play_obj_animation(animation_folder)

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
    nao_viewer = NaoViewerWidget("Nao/Nao.obj", "Nao/Nao.mtl")

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

    # Add text field for logging button clicks with scrolling
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
