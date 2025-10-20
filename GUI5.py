
import sys
import os
import subprocess
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject, Signal, Slot, QUrl, QTimer

from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
from djitellopy import Tello

import random
import re
import pandas as pd
import time
import io
import urllib.parse
import contextlib
from collections import defaultdict
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# Add the parent directory to the Python path for file-shuffler utilities
from GUI5_ManualDroneControl.cameraview.camera_controller import CameraController
from NAO6.nao_connection import send_command
# from Developers.hofCharts import main as hofCharts, ticketsByDev_text

from Developers import devCharts

								

# Import BCI connection for brainwave prediction
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'random-forest-prediction')))
    from client.brainflow1 import bciConnection, DataMode
    BCI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: BCI connection not available: {e}")
    BCI_AVAILABLE = False

# Add the parent directory to the Python path for file-shuffler
sys.path.append(str(Path(__file__).resolve().parent / "file-shuffler"))
sys.path.append(str(Path(__file__).resolve().parent / "file-unify-labels"))
sys.path.append(str(Path(__file__).resolve().parent / "file-remove8channel"))
import unifyTXT
import run_file_shuffler
import remove8channel


class TabController(QObject):
    def __init__(self):
        super().__init__()
        self.nao_process = None

    @Slot()
    def startNaoViewer(self):
        # Stub to satisfy QML calls
        print("startNaoViewer called (stub)")

    @Slot()
    def stopNaoViewer(self):
        # Stub to satisfy QML calls
        print("stopNaoViewer called (stub)")


class BrainwavesBackend(QObject):
    # Signals to update QML components
    flightLogUpdated = Signal(list)
    predictionsTableUpdated = Signal(list)
    imagesReady = Signal(list)
    logMessage = Signal(str)
    naoStarted = Signal(str)    # carry a message for QML
    naoEnded = Signal(str)

    @Slot()
    def startNaoManual(self):
        print("Nao6 Manual session started")
        self.naoStarted.emit("Nao6 Manual session started")

    @Slot()
    def stopNaoManual(self):
        print("Nao6 Manual session ended")
        self.naoEnded.emit("Nao6 Manual session ended")

    @Slot()
    def connectNao(self):
        # Mock function to simulate NAO connection
        self.flight_log.insert(0, "Nao connected.")
        if send_command("connect"):
        # Mock function to simulate drone connection
            self.flight_log.insert(0, "Nao connected.")
        else:
            self.flight_log.insert(0, "Nao failed to connect.")
        self.flightLogUpdated.emit(self.flight_log)
    
    @Slot()
    def nao_sit_down(self):
        if send_command("sit_down"):
            self.flight_log.insert(0, "Sitting down.")
        else:
            self.flight_log.insert(0, "Nao failed to sit.")
        self.flightLogUpdated.emit(self.flight_log)
    
    @Slot()
    def nao_stand_up(self):
        if send_command("stand_up"):
            self.flight_log.insert(0, "Standing Up.")
        else:
            self.fligt_log.insert(0, "Nao failed to stand up.")
        self.flightLogUpdated.emit(self.flight_log)


            

    @Slot(result=str)
    def getDevList(self):
        exclude = {
            "3C Cloud Computing Club <114175379+3C-SCSU@users.noreply.github.com>",
        }

        proc = subprocess.run(
            ["git", "shortlog", "-sne", "--all"],
            capture_output=True, text=True, encoding="utf-8", errors="ignore"
        )

        if proc.returncode != 0:
            return "No developers found."

        lines = proc.stdout.strip().splitlines()
        filtered_lines = []

        for line in lines:
            # Match the author portion
            match = re.match(r"^\s*\d+\s+(?P<author>.+)$", line)
            if match:
                author = match.group("author").strip()
                if author not in exclude:
                    filtered_lines.append(line)

        return "\n".join(filtered_lines) if filtered_lines else "No developers found."

    @Slot(result=str)
    def getTicketsByDev(self) -> str:

            exclude = {
                "3C Cloud Computing Club <114175379+3C-SCSU@users.noreply.github.com>"
            }

            pretty = "%x1e%an <%ae>%x1f%s%x1f%b"
            try:
                proc = subprocess.run(
                    ["git", "log", "--all", f"--pretty=format:{pretty}"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    check=True
                )
            except subprocess.CalledProcessError:
                return "No tickets found."

            raw = proc.stdout or ""
            commits = raw.split("\x1e")

            jira_re = re.compile(r'\b([A-Za-z]{2,}-\d+)\b')
            hash_re = re.compile(r'(?<![A-Za-z0-9])#\d+\b')
            author_to_ticketset = defaultdict(set)

            for entry in commits:
                entry = entry.strip()
                if not entry:
                    continue
                parts = entry.split("\x1f", 2)
                author = parts[0].strip()
                subject = parts[1] if len(parts) > 1 else ""
                body = parts[2] if len(parts) > 2 else ""
                msg = (subject + "\n" + body).strip()

                found = set()
                for m in jira_re.findall(msg):
                    found.add(m.upper())
                for m in hash_re.findall(msg):
                    found.add(m)

                if found:
                    author_to_ticketset[author].update(found)

            if not author_to_ticketset:
                return "No tickets found."

            # Sort authors by ticket count descending, then by name
            lines = []
            for author, tickets in sorted(author_to_ticketset.items(), key=lambda kv: (-len(kv[1]), kv[0].lower())):
                if author in exclude:
                    continue  # skip excluded authors entirely
                lines.append(f"{author}: {', '.join(sorted(tickets))}")


            return "\n".join(lines)

    @Slot()
    def devChart(self):
        print("hofChart() SLOT CALLED")
        try:
            devCharts.main()
            print("hofCharts.main() COMPLETED")
        except Exception as e:
            print(f"hofCharts.main() ERROR: {e}")


    def __init__(self):
        super().__init__()

        # State
        self.flight_log = []                 # Flight log messages (newest first)
        self.predictions_log = []            # Prediction records
        self.current_prediction_label = ""
        self.image_paths = []                # Converted image paths for QML
        self.plots_dir = os.path.abspath("plotscode/plots")
        self.current_dataset = "refresh"     # default dataset
        self.current_model = "Deep Learning" # default model so readMyMind() is safe

        # Drone connectivity flags
        self.drone_connected = False
        self.simulation = False

        # Initialize Tello with shorter timeouts to avoid UI hangs
        try:
            self.tello = Tello()
            try:
                # reduce timeouts; if constants not present, ignore
                self.tello.RESPONSE_TIMEOUT = 3
                self.tello.TIME_BTW_COMMANDS = 0.1
                self.tello.TAKEOFF_TIMEOUT = 10
            except Exception:
                pass
        except Exception as e:
            warning = f"Warning: Failed to initialize Tello drone: {e}"
            print(warning)
            self.logMessage.emit(warning)

        # Bridge logMessage -> flightLogUpdated (thread-safe via Qt signal)
        self.logMessage.connect(self._append_flight_log)

    # Internal helper to prepend to flight log and notify QML
    @Slot(str)
    def _append_flight_log(self, msg: str):
        self.flight_log.insert(0, msg)
        self.flightLogUpdated.emit(self.flight_log)

    @Slot(str)
    def selectModel(self, model_name):
        """Select the machine learning model."""
        print(f"Model selected: {model_name}")
        self.current_model = model_name
        self.flight_log.insert(0, f"Selected Model: {model_name}")
        self.flightLogUpdated.emit(self.flight_log)

    @Slot()
    def readMyMind(self):
        """Run the selected model and process brainwave data."""
        model = getattr(self, "current_model", None) or "Deep Learning"
        if model == "Random Forest":
            prediction = self.run_random_forest()
        else:
            prediction = self.run_deep_learning()

        self.current_prediction_label = prediction

        # Log the prediction to the table
        self.predictions_log.append({
            "count": str(len(self.predictions_log) + 1),
            "server": "Brainwave AI",
            "label": prediction
        })
        self.predictionsTableUpdated.emit(self.predictions_log)

        # Update flight log
        self.flight_log.insert(0, f"Executed: {prediction} (Model: {model})")
        self.flightLogUpdated.emit(self.flight_log)

    def run_random_forest(self):
        """Simulated Random Forest model processing (no scikit-learn)."""
        print("Running Random Forest Model...")
        time.sleep(1)
        return random.choice(["Move Forward", "Turn Left", "Turn Right", "Land"])

    def run_deep_learning(self):
        """Simulated Deep Learning model processing."""
        print("Running Deep Learning Model...")
        time.sleep(2)
        return random.choice(["Move Forward", "Turn Left", "Turn Right", "Land", "Hover"])

    @Slot(str)
    def notWhatIWasThinking(self, manual_action):
        # Manual correction path
        self.predictions_log.append({
            "count": "manual",
            "server": "manual",
            "label": manual_action
        })
        self.predictionsTableUpdated.emit(self.predictions_log)

        self.flight_log.insert(0, f"Manual Action: {manual_action}")
        self.flightLogUpdated.emit(self.flight_log)

    @Slot()
    def executeAction(self):
        if self.current_prediction_label:
            self.flight_log.insert(0, f"Executed: {self.current_prediction_label}")
            self.flightLogUpdated.emit(self.flight_log)

    @Slot()
    def connectDrone(self):
        self.flight_log.insert(0, "Drone connected.")
        self.flightLogUpdated.emit(self.flight_log)

    @Slot()
    def keepDroneAlive(self):
        self.flight_log.insert(0, "Keep alive signal sent.")
        self.flightLogUpdated.emit(self.flight_log)

    @Slot(str)
    def getDroneAction(self, action):
        """
        Execute a drone action. If not connected (or no response), simulate to keep UI responsive.
        """
        try:
            if action == 'connect':
                # Try real connect
                try:
                    self.tello.connect()
                    self.drone_connected = True
                    self.simulation = False
                    self.logMessage.emit("Connected to Tello Drone")
                except Exception as e:
                    # Fall back to simulation
                    self.drone_connected = False
                    self.simulation = True
                    self.logMessage.emit(f"No response from Tello, switching to SIM mode: {e}")
                return

            # If not connected, simulate
            if not self.drone_connected or self.simulation:
                self.logMessage.emit(f"[SIM] {action}")
                return

            # Real drone commands
            if action == 'up':
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

    def go_home(self):
        # Approximate "return to home"
        self.tello.move_back(50)
        self.tello.move_up(50)
        self.logMessage.emit("Returning to home")

    @Slot()
    def check_plots_exist(self):
        """
        Ensure required plot PDFs exist for both datasets. If missing, try to generate via plotscode/controller.py.
        """
        print("\n=== CHECKING IF PLOTS EXIST ===")
        plots_base_dir = Path(self.plots_dir)
        if not plots_base_dir.exists():
            print(f"Creating plots base directory: {plots_base_dir}")
            plots_base_dir.mkdir(parents=True, exist_ok=True)

        datasets = ["rollback", "refresh"]
        pdf_files = [
            "takeoff_plots.pdf", "forward_plots.pdf", "right_plots.pdf",
            "land_plots.pdf", "backward_plots.pdf", "left_plots.pdf"
        ]
        missing_pdfs = False
        for dataset in datasets:
            dataset_dir = plots_base_dir / dataset
            if not dataset_dir.exists():
                print(f"Creating dataset directory: {dataset_dir}")
                dataset_dir.mkdir(parents=True, exist_ok=True)
                missing_pdfs = True
                continue
            print(f"Checking PDFs in {dataset_dir}...")
            for pdf_file in pdf_files:
                pdf_path = dataset_dir / pdf_file
                if not pdf_path.exists():
                    print(f"Missing file: {pdf_path}")
                    missing_pdfs = True
                    break

        if missing_pdfs:
            print("Some plot files are missing. Running controller.py to generate them...")
            controller_path = Path(self.plots_dir).parent / "controller.py"
            print(f"Controller path: {controller_path}")
            print(f"Controller exists: {controller_path.exists()}")

            if controller_path.exists():
                try:
                    original_dir = os.getcwd()
                    os.chdir(controller_path.parent)
                    print(f"Executing: {sys.executable} {controller_path}")
                    result = subprocess.run(
                        [sys.executable, str(controller_path)],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    os.chdir(original_dir)
                    print(f"Output: {result.stdout}")
                    if result.stderr:
                        print(f"Errors: {result.stderr}")
                    print("Successfully generated plot files.")
                    return True
                except subprocess.CalledProcessError as e:
                    print(f"Error running controller.py: {e}")
                    if hasattr(e, 'stderr'):
                        print(f"Error output: {e.stderr}")
                    return False
                except Exception as e:
                    print(f"Unexpected error: {str(e)}")
                    return False
            else:
                print(f"Controller script not found: {controller_path}")
                return False
        return True

    @Slot(str)
    def setDataset(self, dataset_name):
        """
        Switch displayed dataset ('refresh' or 'rollback') and regenerate images.
        """
        if dataset_name.lower() in ["refresh", "rollback"]:
            self.current_dataset = dataset_name.lower()
            print(f"Switched to {self.current_dataset} dataset")
            self.convert_pdfs_to_images()
        else:
            print(f"Invalid dataset name: {dataset_name}")

    @Slot()
    def convert_pdfs_to_images(self):
        """
        Convert PDFs for the current dataset to PNGs and emit imagesReady for QML.
        """
        print(f"\n=== STARTING CONVERT PDFS TO IMAGES FOR {self.current_dataset.upper()} ===")
        success = self.check_plots_exist()
        print(f"Result of check_plots_exist: {success}")

        dataset_dir = Path(self.plots_dir) / self.current_dataset
        self.image_paths = []
        graph_titles = ["Takeoff", "Forward", "Right",
                        "Landing", "Backward", "Left"]
        pdf_files = [
            "takeoff_plots.pdf", "forward_plots.pdf", "right_plots.pdf",
            "land_plots.pdf", "backward_plots.pdf", "left_plots.pdf"
        ]

        # Allow overriding Poppler path via env var if desired
        poppler_path = os.environ.get("POPPLER_PATH", None)

        for i, pdf_file in enumerate(pdf_files):
            pdf_path = dataset_dir / pdf_file
            if not pdf_path.exists():
                print(f"Missing file: {pdf_path}")
                continue

            # Convert first page to PNG
            if poppler_path:
                images = convert_from_path(str(pdf_path), dpi=150, poppler_path=poppler_path)
            else:
                images = convert_from_path(str(pdf_path), dpi=150)

            image_path = dataset_dir / f"{pdf_file.replace('.pdf', '.png')}"
            images[0].save(str(image_path), "PNG")
            print(f"Generated image: {image_path}")

            self.image_paths.append({
                "graphTitle": graph_titles[i],
                "imagePath": QUrl.fromLocalFile(str(image_path)).toString()
            })

        print("Final Image Paths Sent to QML:", self.image_paths)
        self.imagesReady.emit(self.image_paths)

    @Slot()
    def launch_file_shuffler_gui(self):
        # Launch the file shuffler GUI program
        file_shuffler_path = Path(__file__).resolve().parent / "file-shuffler/file-shuffler-gui.py"
        subprocess.Popen(["python", str(file_shuffler_path)])

    @Slot(str, result=str)
    def run_file_shuffler_program(self, path):
        # Path may be file:// encoded from QML
        path = path.replace("file://", "")
        if path.startswith("/C:"):
            path = 'C' + path[2:]
        response = run_file_shuffler.main(path)
        return response

    @Slot(str, result=str)
    def unify_thoughts(self, base_dir):
        """
        Called from QML when the user picks a directory to unify .txt labels.
        """
        # strip file:/// if necessary
        path = base_dir.replace("file://", "")
        if base_dir.startswith("file:///"):
            base_dir = urllib.parse.unquote(base_dir.replace("file://", ""))
            if os.name == 'nt' and base_dir.startswith("/"):
                base_dir = base_dir[1:]
        print("Unify Thoughts on directory:", base_dir)
        output = io.StringIO()

        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                unifyTXT.move_any_txt_files(base_dir)
                print("Unify complete.")

        except Exception as e:
            print("Error during unify:", e)

        return output.getvalue()

    @Slot(str, result=str)
    def remove_8_channel(self, base_dir):
        """
        Called from QML when the user picks a directory to remove 8-channel data.
        """
        if base_dir.startswith("file:///"):
            base_dir = urllib.parse.unquote(base_dir.replace("file://", ""))
            if os.name == 'nt' and base_dir.startswith("/"):
                base_dir = base_dir[1:]
        print("Removing 8 Channel data from:", base_dir)
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                remove8channel.file_remover(base_dir)
                print("8 Channel Data Removal complete.")
        except Exception as e:
            print("Error during cleanup: ", e)

    @Slot(str)
    def setDataMode(self, mode):
        """
        Switch between synthetic and live BrainFlow data modes.
        """
        if mode == "synthetic":
            self.init_synthetic_board()
            print("Switched to Synthetic Data Mode")
        elif mode == "live":
            self.init_live_board()
            print("Switched to Live Data Mode")
        else:
            print(f"Unknown data mode: {mode}")

    def init_synthetic_board(self):
        params = BrainFlowInputParams()
        self.board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
        print("\nSynthetic board initialized.")

    def init_live_board(self):
        params = BrainFlowInputParams()
        params.serial_port = "/dev/cu.usbserial-D200PMA1"  # Update for your system
        self.board = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)
        print("\nLive headset board initialized.")


if __name__ == "__main__":
    os.environ["QT_QUICK_CONTROLS_STYLE"] = "Fusion"
    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()

    # Controllers
    tab_controller = TabController()
    print("TabController created")

    backend = BrainwavesBackend()
    backend.logMessage.emit("Drone Flight Log connected.")  # sanity message into the Flight Log

    # Expose to QML
    engine.rootContext().setContextProperty("tabController", tab_controller)
    engine.rootContext().setContextProperty("backend", backend)
    engine.rootContext().setContextProperty("imageModel", [])  # initial empty model
    engine.rootContext().setContextProperty("fileShufflerGui", backend)
    print("Controllers exposed to QML")

    # Load QML
    qml_file = Path(__file__).resolve().parent / "main.qml"

    engine.load(str(qml_file))

    # Convert PDFs after engine load, guard Poppler nicely
    try:
        backend.convert_pdfs_to_images()
    except PDFInfoNotInstalledError:
        backend.logMessage.emit("Poppler is not installed. Skipping plot image conversion.")
    except Exception as e:
        print(f"Error converting PDFs: {str(e)}")

    # Update imageModel with a queued singleShot to avoid paint timing warnings
    def _set_image_model(images):
        engine.rootContext().setContextProperty("imageModel", images)
    backend.imagesReady.connect(lambda images: QTimer.singleShot(0, lambda: _set_image_model(images)))

    sys.exit(app.exec())


