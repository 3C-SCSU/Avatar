# This Python file uses the following encoding: utf-8
import sys
import os
from pathlib import Path
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject, Signal, Slot

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

    @Slot()
    def readMyMind(self):
        # Mock function to simulate brainwave reading
        self.current_prediction_label = "Move Forward"
        # Update the predictions log
        self.predictions_log.append({
            "count": "1",
            "server": "Prediction Server",
            "label": self.current_prediction_label
        })
        self.predictionsTableUpdated.emit(self.predictions_log)

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
        
 # Drone control class
    class DroneController(QObject):
        logMessage = Signal(str)  # Signal to emit log messages

        def __init__(self):
            super().__init__()
            self.tello = Tello()
            try:
                self.tello.connect()
                self.logMessage.emit("Connected to Tello Drone")
            except Exception as e:
                self.logMessage.emit(f"Error during initialization: {e}")

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

if __name__ == "__main__":
    os.environ["QT_QUICK_CONTROLS_STYLE"] = "Fusion"
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    # Initialize backend before loading QML
    backend = BrainwavesBackend()
    engine.rootContext().setContextProperty("backend", backend)
    engine.rootContext().setContextProperty("imageModel", [])  # Initialize empty model

    # Load QML
    qml_file = Path(__file__).resolve().parent / "main.qml"
    engine.load(str(qml_file))

    # Convert PDFs after engine load
    backend.convert_pdfs_to_images()

    # Ensure image model updates correctly
    backend.imagesReady.connect(lambda images: engine.rootContext().setContextProperty("imageModel", images))


    sys.exit(app.exec())
