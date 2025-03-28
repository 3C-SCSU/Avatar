# This Python file uses the following encoding: utf-8
import sys
import os
from pathlib import Path
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject, Signal, Slot, QUrl
from pdf2image import convert_from_path  
from djitellopy import Tello

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent / "file-shuffler"))
import run_file_shuffler



class BrainwavesBackend(QObject):
    # Define signals to update QML components
    flightLogUpdated = Signal(list)
    predictionsTableUpdated = Signal(list)
    imagesReady = Signal(list)

    def __init__(self):
        super().__init__()
        self.flight_log = []  # List to store flight log entries
        self.predictions_log = []  # List to store prediction records
        self.current_prediction_label = ""
        self.image_paths = []  # Store converted image paths
        self.plots_dir = os.path.abspath("plotscode/plots") # Change the path if needed
        self.tello = Tello()

    @Slot(str)
    def selectModel(self, model_name):
        """ Select the machine learning model """
        print(f"Model selected: {model_name}")
        self.current_model = model_name
        self.flight_log.insert(0, f"Selected Model: {model_name}")
        self.flightLogUpdated.emit(self.flight_log)

    @Slot()
    def readMyMind(self):
        """ Runs the selected model and processes the brainwave data. """
        if self.current_model == "Random Forest":
            prediction = self.run_random_forest()
        else:
            prediction = self.run_deep_learning()

        # Log the prediction
        self.predictions_log.append({
            "count": str(len(self.predictions_log) + 1),
            "server": "Brainwave AI",
            "label": prediction
        })
        self.predictionsTableUpdated.emit(self.predictions_log)

        # Update Flight Log
        self.flight_log.insert(0, f"Executed: {prediction} (Model: {self.current_model})")
        self.flightLogUpdated.emit(self.flight_log)

    def run_random_forest(self):
        """ Simulated Random Forest model processing """
        print("Running Random Forest Model...")
        time.sleep(1)  # Simulate processing delay
        return random.choice(["Move Forward", "Turn Left", "Turn Right", "Land"])

    def run_deep_learning(self):
        """ Simulated Deep Learning model processing """
        print("Running Deep Learning Model...")
        time.sleep(2)  # Simulate slightly longer DL processing time
        return random.choice(["Move Forward", "Turn Left", "Turn Right", "Land", "Hover"])

    @Slot(str)
    def notWhatIWasThinking(self, manual_action):
        # Handle manual action input
        self.predictions_log.append({
            "count": "manual",
            "server": "manual",
            "label": manual_action
        })
        self.predictionsTableUpdated.emit(self.predictions_log)

    @Slot()
    def executeAction(self):
        # Execute the current prediction
        if self.current_prediction_label:
            self.flight_log.insert(0, f"Executed: {self.current_prediction_label}")
            self.flightLogUpdated.emit(self.flight_log)

    @Slot()
    def connectDrone(self):
        # Mock function to simulate drone connection
        self.flight_log.insert(0, "Drone connected.")
        self.flightLogUpdated.emit(self.flight_log)
        
    @Slot()
    def keepDroneAlive(self):
        # Mock function to simulate sending keep-alive signal
        self.flight_log.insert(0, "Keep alive signal sent.")
        self.flightLogUpdated.emit(self.flight_log)

    @Slot(str)
    def getDroneAction(self, action):
        try:
            if action == 'connect':
                self.tello.connect()
                self.logMessage.emit("Connected to Tello Drone")
            elif action == 'up':
                self.tello.move_up(30)
                self.logMessage.emit("Moving up")
            elif action == 'down':
                self.tello.move_down(30)
                self.logMessage.emit("Moving down")
            elif action == 'forward':
                self.tello.move_forward(30)
                self.logMessage.emit("Moving forward")
            elif action == 'backward':
                self.tello.move_back(30)
                self.logMessage.emit("Moving backward")
            elif action == 'left':
                self.tello.move_left(30)
                self.logMessage.emit("Moving left")
            elif action == 'right':
                self.tello.move_right(30)
                self.logMessage.emit("Moving right")
            elif action == 'turn_left':
                self.tello.rotate_counter_clockwise(45)
                self.logMessage.emit("Rotating left")
            elif action == 'turn_right':
                self.tello.rotate_clockwise(45)
                self.logMessage.emit("Rotating right")
            elif action == 'takeoff':
                self.tello.takeoff()
                self.logMessage.emit("Taking off")
            elif action == 'land':
                self.tello.land()
                self.logMessage.emit("Landing")
            elif action == 'go_home':
                self.go_home()
            else:
                self.logMessage.emit("Unknown action")
        except Exception as e:
            self.logMessage.emit(f"Error during {action}: {e}")

    # Method for returning to home (an approximation)
    def go_home(self):
        # Assuming the home action means moving backward and upwards
        self.tello.move_back(50)  # Move back to home point (adjust distance as needed)
        self.tello.move_up(50)    # Move up to avoid obstacles
        self.logMessage.emit("Returning to home")
    
    @Slot()
    def convert_pdfs_to_images(self):
        # Convert PDF files to images and send image paths + graph names to QML.
        self.image_paths = []
        graph_titles = ["Takeoff Graph", "Forward Graph", "Right Graph",
                        "Landing Graph", "Backward Graph", "Left Graph"]

        # Load files in the correct order
        pdf_files = [
            "takeoff_plots.pdf", "forward_plots.pdf", "right_plots.pdf",
            "land_plots.pdf", "backward_plots.pdf", "left_plots.pdf"
        ]

        for i, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(self.plots_dir, pdf_file)
            if not os.path.exists(pdf_path):
                print(f"Missing file: {pdf_path}")  # Debugging: Check missing PDFs
                continue  # Skip if file does not exist

            images = convert_from_path(pdf_path, dpi=150)  # Convert PDF to image
            image_path = os.path.join(self.plots_dir, f"{pdf_file.replace('.pdf', '.png')}")
            images[0].save(image_path, "PNG")  # Save first page as an image
            
            # Debugging: Print the generated image path
            print(f"Generated image: {image_path}")

            self.image_paths.append({"graphTitle": graph_titles[i], "imagePath": QUrl.fromLocalFile(image_path).toString()})


        # Debugging: Print final list of image paths
        print("Final Image Paths Sent to QML:", self.image_paths)

        self.imagesReady.emit(self.image_paths)  # Send data to QML
    
    @Slot()
    def launch_file_shuffler_gui(self):
        # Launch the file shuffler GUI program
        file_shuffler_path = Path(__file__).resolve().parent / "file-shuffler/file-shuffler-gui.py"
        subprocess.Popen(["python", str(file_shuffler_path)])


    @Slot(str, result=str)
    def run_file_shuffler_program(self, path):
        #Need to parse the path as the FolderDialog appends file:// in front of the selection
        path = path.replace("file://", "")
        response = run_file_shuffler.main(path)
        return response

if __name__ == "__main__":
    os.environ["QT_QUICK_CONTROLS_STYLE"] = "Fusion"
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    # Initialize backend before loading QML
    backend = BrainwavesBackend()
    engine.rootContext().setContextProperty("backend", backend)
    engine.rootContext().setContextProperty("imageModel", [])  # Initialize empty model
    engine.rootContext().setContextProperty("fileShufflerGui", backend) #For file shuffler


    # Load QML
    qml_file = Path(__file__).resolve().parent / "main.qml"
    engine.load(str(qml_file))

    # Convert PDFs after engine load
    backend.convert_pdfs_to_images()

    # Ensure image model updates correctly
    backend.imagesReady.connect(lambda images: engine.rootContext().setContextProperty("imageModel", images))


    sys.exit(app.exec())