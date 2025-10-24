import sys
import os
import subprocess
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject, Signal, Slot, QProcess, QUrl
from pdf2image import convert_from_path
from djitellopy import Tello
import threading
import random
import re
import pandas as pd
import time
import io
import urllib.parse
import contextlib
from collections import defaultdict
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from GUI5_ManualDroneControl.cameraview.camera_controller import CameraController
from NAO6.nao_connection import send_command
from PySide6.QtCore import Property
# from Developers.hofCharts import main as hofCharts, ticketsByDev_text NA

from Developers import devCharts
from dotenv import load_dotenv
from Developers.devCharts import GitHubClient, PRAnalyzer, CacheManager, TicketListManager

								

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


class BrainwavesBackend(QObject):
    # Define signals to update QML components
    flightLogUpdated = Signal(list)
    predictionsTableUpdated = Signal(list)
    imagesReady = Signal(list)
    logMessage = Signal(str)
    naoStarted = Signal()
    naoEnded = Signal()
    isConnectedChanged = Signal()

    @Slot()
    def startNaoManual(self):
        print("Nao6 Manual session started")
        self.naoStarted.emit("Nao6 Manual session started")

    @Slot()
    def stopNaoManual(self):
        print("Nao6 Manual session ended")
        self.naoEnded.emit("Nao6 Manual session ended")

    @Slot(str, str)
    def connectNao(self, ip="192.168.23.53", port="9559"):
        """Connect to NAO robot with specified IP and Port"""
        try:
            # Log the connection attempt
            self.flight_log.insert(0, f"Attempting to connect to NAO at {ip}:{port}...")
            self.flightLogUpdated.emit(self.flight_log)

            # Send connect command
            if send_command("connect"):
                self.flight_log.insert(0, f"Nao connected successfully at {ip}:{port}.")
            else:
                self.flight_log.insert(0, f"Nao failed to connect at {ip}:{port}.")
        except Exception as e:
            self.flight_log.insert(0, f"Error connecting to NAO: {str(e)}")

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
    def getApprovedPRsByAuthor(self) -> str:
        """
        Uses the new OOP classes to return a text list of approved PR counts by author.
        """
        try:
            # Import classes from the refactor

            # Ensure we load the Developers/token.env (so Config.from_env equivalent)
            env_path = Path(__file__).resolve().parent / "Developers" / "token.env"
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)

            repo = os.getenv("GITHUB_REPO")
            token = os.getenv("GITHUB_TOKEN")
            if not repo or not token:
                return "Missing GitHub credentials."

            # Build GH client and analyzer
            gh = GitHubClient(repo=repo, token=token)
            analyzer = PRAnalyzer(gh)

            # Cache file next to Developers/pr_tier_analysis (keeps original location)
            cache_file = Path(__file__).resolve().parent / "Developers" / "approved_prs_cache.json"
            cache = CacheManager(cache_file, ttl_seconds=60 * 60)

            # Try cache first
            cached = cache.load()
            if cached:
                # cached is List[Tuple[author,count]]
                return "\n".join(f"{author}: {count} approved tickets/PR(s)" for author, count in cached)

            # Fetch closed PRs and filter merged
            closed_prs = gh.list_closed_pulls()
            if not isinstance(closed_prs, list):
                return "Error fetching PRs."

            merged_prs = [pr for pr in closed_prs if pr.get("merged_at")]

            # For each merged PR, check reviews for approval via GitHubClient
            approved_prs = []
            for pr in merged_prs:
                try:
                    number = pr.get("number")
                    if number is None:
                        continue
                    if gh.pr_has_approval(number):
                        approved_prs.append(pr)
                except Exception:
                    # conservative: skip PRs we can't verify
                    continue

            author_counts = analyzer.count_approved_prs_by_author(approved_prs)
            if not author_counts:
                # save empty cache for consistency with previous behavior
                cache.save([])
                return "No approved pull requests found."

            # Save cache and return nicely formatted text
            cache.save(author_counts)
            return "\n".join(f"{author}: {count} approved tickets/PR(s)" for author, count in author_counts)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error fetching PR data: {e}"

    @Slot(result=str)
    def refreshDevTickList(self) -> str:
        """
        Run the existing devCharts.py script as a subprocess so we reuse its working code
        (plots, CSV, JSON). Runs in current environment; intended to be called from a worker thread.
        """

        # Resolve the developer script path
        script_path = Path(__file__).resolve().parent / "Developers" / "devCharts.py"
        if not script_path.exists():
            return f"devCharts.py not found at {script_path}"

        # Ensure token.env is loaded for local env (optional — subprocess will load it if its cwd is Developers)
        env_path = script_path.parent / "token.env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)  # keeps os.environ set for subprocess inherit (not strictly required)

        # Build the command: run using the same python executable
        cmd = [sys.executable, str(script_path)]

        try:
            # Run subprocess in the Developers directory so token.env and relative paths work
            # Use explicit UTF-8 decoding and replace invalid bytes (prevents UnicodeDecodeError on Windows)
            proc = subprocess.run(
                cmd,
                cwd=str(script_path.parent),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False
            )

            # be defensive: proc.stdout/stderr can be None in some edge cases
            stdout = (proc.stdout or "").strip()
            stderr = (proc.stderr or "").strip()
            rc = proc.returncode

            if rc == 0:
                msg_lines = []
                if stdout:
                    msg_lines.append("Output:\n" + stdout)
                msg_lines.append("Refresh completed successfully.")
                return "\n\n".join(msg_lines)
            else:
                # On failure include stderr for debugging (already utf-8 decoded with replacement)
                return f"devCharts.py failed (code {rc}).\n\nStdout:\n{stdout}\n\nStderr:\n{stderr}"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error launching devCharts.py: {e}"



    @Slot(result=str)
    def getTicketsByDev(self) -> str:
        """
        Return a human-readable ticket list per developer using the new OOP classes.
        """
        try:

            env_path = Path(__file__).resolve().parent / "Developers" / "token.env"
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)

            repo = os.getenv("GITHUB_REPO")
            token = os.getenv("GITHUB_TOKEN")
            if not repo or not token:
                return "Missing GitHub credentials."

            gh = GitHubClient(repo=repo, token=token)
            analyzer = PRAnalyzer(gh)

            # Ticket list file location (same as the refactor)
            ticket_list_path = Path(__file__).resolve().parent / "Developers" / "authorTicketList.json"
            ticket_mgr = TicketListManager(ticket_list_path, ttl_seconds=60 * 60)

            # Try cached ticket list first
            cached = ticket_mgr.load()
            if cached:
                lines = []
                for entry in cached:
                    name = entry.get("author_name", "Unknown")
                    tickets = entry.get("tickets", [])
                    ticket_str = ", ".join(t.replace("PR#", "#") for t in tickets)
                    lines.append(f"{name}. Tickets: {ticket_str if ticket_str else 'None'}")
                return "\n".join(lines)

            # No cache => fetch closed PRs and build list
            closed_prs = gh.list_closed_pulls()
            if not isinstance(closed_prs, list):
                return "Error fetching PRs."

            merged_prs = [pr for pr in closed_prs if pr.get("merged_at")]
            if not merged_prs:
                return "No merged PRs found."

            author_ticket_list = analyzer.ticket_list_by_author(merged_prs)
            if not author_ticket_list:
                return "No tickets found in PRs."

            out_path = ticket_mgr.save(author_ticket_list)

            # Create nicely formatted lines (filter out the '3C-SCSU' as original)
            filtered = [e for e in author_ticket_list if e.get("author_name") != "3C-SCSU"]
            lines = []
            for entry in sorted(filtered, key=lambda e: e.get("author_name", "").lower()):
                name = entry.get("author_name", "Unknown")
                tickets = entry.get("tickets", [])
                ticket_str = ", ".join(t for t in tickets)
                lines.append(f"{name}. Tickets: {ticket_str if ticket_str else 'None'}")

            if out_path:
                lines.append(f"Saved ticket list to: {out_path}")
            else:
                lines.append("Ticket list file path not available.")

            return "\n".join(lines)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error fetching tickets: {e}"

    '''
    @Slot()
    def devChart(self):
        print("hofChart() SLOT CALLED")
        try:
            devCharts.main()
            print("hofCharts.main() COMPLETED")
        except Exception as e:
            print(f"hofCharts.main() ERROR: {e}")
    '''

    def get_is_connected(self):
        return self.isConnectedChanged

    def set_is_connected(self, value):
        if self.is_connected != value:
            self.is_connected = value
            self.isConnectedChanged.emit()

    is_connected_prop = Property(bool, get_is_connected, set_is_connected, notify=isConnectedChanged)


    def __init__(self):
        super().__init__()
        self.flight_log = []  # List to store flight log entries
        self.predictions_log = []  # List to store prediction records
        self.current_prediction_label = ""
        self.current_model = "Random Forest"  # Default model
        self.current_framework = "PyTorch"  # Default framework
        self.image_paths = []  # Store converted image paths
        self.plots_dir = os.path.abspath("plotscode/plots")  # Base plots directory
        self.current_dataset = "refresh"  # Default dataset to display
        self.is_connected = False
        try:
            self.tello = Tello()
            self.connected = False
        except Exception as e:
            print(f"Warning: Failed to initialize Tello drone: {e}")
            self.logMessage.emit(f"Warning: Failed to initialize Tello drone: {e}")

        # Initialize camera controller with tello instance
        self.camera_controller = CameraController()
        if hasattr(self, 'tello'):
            self.camera_controller.set_tello_instance(self.tello)

        # Initialize BCI connection for brainwave prediction
        if BCI_AVAILABLE:
            try:
                self.bcicon = bciConnection.get_instance()
                print("BCI connection initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize BCI connection: {e}")
                self.bcicon = None
        else:
            self.bcicon = None

    @Slot(str)
    def selectModel(self, model_name):
        """ Select the machine learning model """
        self.logMessage.emit(f"Model selected: {model_name}")
        self.current_model = model_name
        self.flight_log.insert(0, f"Selected Model: {model_name}")
        self.flightLogUpdated.emit(self.flight_log)

    @Slot(str)
    def selectFramework(self, framework_name):
        """ Select the machine learning framework """
        self.logMessage.emit(f"Framework selected: {framework_name}")
        self.current_framework = framework_name
        self.flight_log.insert(0, f"Selected Framework: {framework_name}")
        self.flightLogUpdated.emit(self.flight_log)

    @Slot()
    def readMyMind(self):
        """ Runs the selected model and processes the brainwave data. """
        if self.current_model == "Random Forest":
            if self.current_framework == "PyTorch":
                prediction = self.run_random_forest_pytorch()
            else:
                prediction = self.run_random_forest_tensorflow()
        else:  # Deep Learning
            if self.current_framework == "PyTorch":
                prediction = self.run_deep_learning_pytorch()
            else:
                prediction = self.run_deep_learning_tensorflow()
        self.logMessage.emit(f"Prediction received: {prediction}")
        # Set current prediction
        self.current_prediction_label = prediction

        # Log the prediction
        self.predictions_log.append({
            "count": str(len(self.predictions_log) + 1),
            "server": "Brainwave AI",
            "label": prediction
        })
        self.predictionsTableUpdated.emit(self.predictions_log)

        # Update Flight Log
        self.flight_log.insert(0, f"Executed: {prediction} (Model: {self.current_model}, Framework: {self.current_framework})")
        self.flightLogUpdated.emit(self.flight_log)

    def run_random_forest_pytorch(self):
        """ Random Forest model processing with PyTorch backend """
        print("Running Random Forest Model with PyTorch...")
        try:
            # Use the BCI connection to get real brainwave data
            if hasattr(self, 'bcicon') and self.bcicon:
                prediction_response = self.bcicon.bciConnectionController()
                if prediction_response:
                    return prediction_response.get('prediction_label', 'forward')
        except Exception as e:
            print(f"Error with PyTorch Random Forest: {e}")

        # Fallback to simulation with PyTorch-specific labels
        time.sleep(1)
        return random.choice(["forward", "backward", "left", "right", "takeoff", "land"])

    def run_random_forest_tensorflow(self):
        """ Random Forest model processing with TensorFlow backend """
        print("Running Random Forest Model with TensorFlow...")
        try:
            # Use the BCI connection to get real brainwave data
            if hasattr(self, 'bcicon') and self.bcicon:
                prediction_response = self.bcicon.bciConnectionController()
                if prediction_response:
                    return prediction_response.get('prediction_label', 'forward')
        except Exception as e:
            print(f"Error with TensorFlow Random Forest: {e}")

        # Fallback to simulation with TensorFlow-specific labels
        time.sleep(1)
        return random.choice(["forward", "backward", "left", "right", "takeoff", "land"])

    def run_deep_learning_pytorch(self):
        """ Deep Learning model processing with PyTorch backend """
        print("Running Deep Learning Model with PyTorch...")
        try:
            # Simulate PyTorch deep learning model processing
            # In a real implementation, this would load and run a PyTorch CNN model
            time.sleep(2)  # Simulate longer processing time for deep learning
            return random.choice(["forward", "backward", "left", "right", "takeoff", "land", "up", "down"])
        except Exception as e:
            print(f"Error with PyTorch Deep Learning: {e}")
            return "forward"

    def run_deep_learning_tensorflow(self):
        """ Deep Learning model processing with TensorFlow backend """
        print("Running Deep Learning Model with TensorFlow...")
        try:
            # Simulate TensorFlow deep learning model processing
            # In a real implementation, this would load and run a TensorFlow CNN model
            time.sleep(2)  # Simulate longer processing time for deep learning
            return random.choice(["forward", "backward", "left", "right", "takeoff", "land", "up", "down"])
        except Exception as e:
            print(f"Error with TensorFlow Deep Learning: {e}")
            return "forward"

    @Slot(str)
    def notWhatIWasThinking(self, manual_action):
        # Handle manual action input
        self.predictions_log.append({
            "count": "manual",
            "server": "manual",
            "label": manual_action
        })
        self.predictionsTableUpdated.emit(self.predictions_log)

        # Also update flight log
        self.flight_log.insert(0, f"Manual Action: {manual_action}")
        self.flightLogUpdated.emit(self.flight_log)
        self.logMessage.emit(f"Manual input: {manual_action}") 

    @Slot()
    def executeAction(self):
        # Execute the current prediction
        if self.current_prediction_label:
            self.flight_log.insert(0, f"Executed: {self.current_prediction_label}")
            self.flightLogUpdated.emit(self.flight_log)
            self.logMessage.emit(f"Executed action: {self.current_prediction_label}")

    @Slot()
    def connectDrone(self):
        # Mock function to simulate drone connection
        self.flight_log.insert(0, "Drone connected.")
        self.flightLogUpdated.emit(self.flight_log)
        self.logMessage.emit("Drone connected.")

    @Slot()
    def keepDroneAlive(self):
        # Mock function to simulate sending keep-alive signal
        self.flight_log.insert(0, "Keep alive signal sent.")
        self.flightLogUpdated.emit(self.flight_log)

    @Slot(str)
    def getDroneAction(self, action):
        threading.Thread(target=self._doAction, args=(action,), daemon=True).start()

    def _doAction(self, action):
        try:
            if action == 'connect':
                self.tello.connect()
                self.connected = True
                self.logMessage.emit("Connected to Tello Drone")
            elif self.connected:
                self.logMessage.emit("Drone did not connect.")
                return

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
            elif action == 'stream':
                if hasattr(self, 'camera_controller'):
                    self.camera_controller.start_camera_stream()
                    self.logMessage.emit("Starting camera stream")
                else:
                    self.logMessage.emit("Camera controller not available")
            else:
                self.logMessage.emit("Unknown action")
        except Exception as e:
            self.logMessage.emit(f"Error during {action}: {e}")


    # Method for returning to home (an approximation)
    def go_home(self):
        # Assuming the home action means moving backward and upwards
        self.tello.move_back(50)  # Move back to home point (adjust distance as needed)
        self.tello.move_up(50)  # Move up to avoid obstacles
        self.logMessage.emit("Returning to home")

    @Slot()
    def check_plots_exist(self):
        """
        Check if all necessary plot PDFs exist in both Rollback and Refresh directories.
        If not, run controller.py to generate them.
        """
        print("\n=== CHECKING IF PLOTS EXIST ===")

        # Create plots base directory if it doesn't exist
        plots_base_dir = Path(self.plots_dir)
        if not plots_base_dir.exists():
            print(f"Creating plots base directory: {plots_base_dir}")
            plots_base_dir.mkdir(parents=True, exist_ok=True)

        # List of datasets to check
        datasets = ["rollback", "refresh"]

        # List of PDF files that should exist for each dataset
        pdf_files = [
            "takeoff_plots.pdf", "forward_plots.pdf", "right_plots.pdf",
            "land_plots.pdf", "backward_plots.pdf", "left_plots.pdf"
        ]

        # Check if all directories and PDFs exist
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

        # If any PDFs are missing, run the controller.py script
        if missing_pdfs:
            print("Some plot files are missing. Running controller.py to generate them...")

            # Get the path to controller.py in the plotscode directory
            controller_path = Path(self.plots_dir).parent / "controller.py"  # plotscode/controller.py
            print(f"Controller path: {controller_path}")
            print(f"Controller exists: {controller_path.exists()}")

            if controller_path.exists():
                try:
                    # Change to the plotscode directory before running the script
                    original_dir = os.getcwd()
                    os.chdir(controller_path.parent)

                    # Run the controller.py script to generate plots for both datasets
                    print(f"Executing: {sys.executable} {controller_path}")
                    result = subprocess.run(
                        [sys.executable, str(controller_path)],
                        check=True,
                        capture_output=True,
                        text=True
                    )

                    # Go back to the original directory
                    os.chdir(original_dir)

                    # Print output for debugging
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

        return True  # All files exist

    @Slot(str)
    def setDataset(self, dataset_name):
        """
        Set the current dataset to display (refresh or rollback).
        :param dataset_name: Name of the dataset ('refresh' or 'rollback')
        """
        if dataset_name.lower() in ["refresh", "rollback"]:
            self.current_dataset = dataset_name.lower()
            print(f"Switched to {self.current_dataset} dataset")
            # Update the displayed images
            self.convert_pdfs_to_images()
        else:
            print(f"Invalid dataset name: {dataset_name}")

    @Slot()
    def convert_pdfs_to_images(self):
        """
        Convert PDF files from the current dataset to images and send to QML.
        """
        print(f"\n=== STARTING CONVERT PDFS TO IMAGES FOR {self.current_dataset.upper()} ===")

        # First check if all plot PDFs exist, and generate them if needed
        success = self.check_plots_exist()
        print(f"Result of check_plots_exist: {success}")

        # Current dataset directory
        dataset_dir = Path(self.plots_dir) / self.current_dataset

        # Convert PDF files to images and send image paths + graph names to QML.
        self.image_paths = []
        graph_titles = ["Takeoff", "Forward", "Right",
                        "Landing", "Backward", "Left"]

        # Load files in the correct order
        pdf_files = [
            "takeoff_plots.pdf", "forward_plots.pdf", "right_plots.pdf",
            "land_plots.pdf", "backward_plots.pdf", "left_plots.pdf"
        ]

        for i, pdf_file in enumerate(pdf_files):
            pdf_path = dataset_dir / pdf_file
            if not pdf_path.exists():
                print(f"Missing file: {pdf_path}")  # Debugging: Check missing PDFs
                continue  # Skip if file does not exist

            images = convert_from_path(str(pdf_path), dpi=150)  # Convert PDF to image
            image_path = dataset_dir / f"{pdf_file.replace('.pdf', '.png')}"
            images[0].save(str(image_path), "PNG")  # Save first page as an image

            # Debugging: Print the generated image path
            print(f"Generated image: {image_path}")

            self.image_paths.append({
                "graphTitle": graph_titles[i],
                "imagePath": QUrl.fromLocalFile(str(image_path)).toString()
            })

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
        # Need to parse the path as the FolderDialog appends file:// in front of the selection
        path = path.replace("file://", "")
        if path.startswith("/C:"):
            path = 'C' + path[2:]

        response = run_file_shuffler.main(path)
        return response

        # Adding Synthetic Data and Live Data Logic (Row 327 to 355) as part of Ticket 186

    @Slot(str, result=str)
    def unify_thoughts(self, base_dir):
        """
        Called from QML when the user picks a directory.
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
        Called from QML when the user picks a directory to remove 8 channel data.
        """
        # Decode URL path
        if base_dir.startswith("file:///"):
            base_dir = urllib.parse.unquote(base_dir.replace("file://", ""))
            if os.name == 'nt' and base_dir.startswith("/"):
                base_dir = base_dir[1:]
        print("Removing 8 Channel data form:", base_dir)
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
        Set data mode to either synthetic or live based on radio button selection.
        """
        if mode == "synthetic":
            self.init_synthetic_board()
            self.logMessage.emit("Switched to Synthetic Data Mode")
        elif mode == "live":
            self.init_live_board()
            self.logMessage.emit("Switched to Live Data Mode")
        else:
            print(f"Unknown data mode: {mode}")

    def init_synthetic_board(self):
        """ Initialize BrainFlow with synthetic board for testing """
        params = BrainFlowInputParams()
        self.board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
        print("\nSynthetic board initialized.")

    def init_live_board(self):
        """ Initialize BrainFlow with a real headset """
        params = BrainFlowInputParams()
        params.serial_port = "/dev/cu.usbserial-D200PMA1"  # Update if different on your system
        self.board = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)
        print("\nLive headset board initialized.")




if __name__ == "__main__":
    os.environ["QT_QUICK_CONTROLS_STYLE"] = "Fusion"
    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()

    # Create our controllers
    tab_controller = TabController()
    print("TabController created")

    # Initialize backend before loading QML
    backend = BrainwavesBackend()
    engine.rootContext().setContextProperty("tabController", tab_controller)
    engine.rootContext().setContextProperty("backend", backend)
    engine.rootContext().setContextProperty("imageModel", [])  # Initialize empty model
    engine.rootContext().setContextProperty("fileShufflerGui", backend)  # For file shuffler
    engine.rootContext().setContextProperty("cameraController", backend.camera_controller)
    print("Controllers exposed to QML")
    engine.rootContext().setContextProperty("fileShufflerGui", backend)  # For file shuffler

    # Load QML
    qml_file = Path(__file__).resolve().parent / "main.qml"

    engine.load(str(qml_file))

    # Convert PDFs after engine load
    try:
        backend.convert_pdfs_to_images()
    except Exception as e:
        print(f"Error converting PDFs: {str(e)}")

    # Ensure image model updates correctly
    backend.imagesReady.connect(lambda images: engine.rootContext().setContextProperty("imageModel", images))

    sys.exit(app.exec())
