import sys
import os
import subprocess
from pathlib import Path
from PySide6.QtCore import QObject, Signal, Slot, QThread
import io
import urllib.parse
import contextlib

# Add the parent directory to the Python path for file-shuffler
sys.path.append(str(Path(__file__).resolve().parent / "file-shuffler"))
sys.path.append(str(Path(__file__).resolve().parent / "file-unify-labels"))
sys.path.append(str(Path(__file__).resolve().parent / "file-remove8channel"))
import unifyTXT
import run_file_shuffler
import remove8channel

class FileShufflerWorker(QObject):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    @Slot()
    def run(self):
        try:
            response = run_file_shuffler.main(self.path)
            if response is None:
                response = "File shuffler finished."
            self.finished.emit(str(response))
        except Exception as e:
            self.error.emit(f"Error during file shuffling: {e}")

class ShufflerAPI(QObject):
    fileShufflerFinished = Signal(str)
    fileShufflerError = Signal(str)

    def __init__(self):
        super().__init__()
        self._thread = None
        self._worker = None

    @Slot()
    def launch_file_shuffler_gui(self):
        # Launch the file shuffler GUI program
        file_shuffler_path = Path(__file__).resolve().parent / "file-shuffler/file-shuffler-gui.py"
        subprocess.Popen(["python", str(file_shuffler_path)])

    @Slot(str)
    def run_file_shuffler_program(self, path):
        # Need to parse the path as the FolderDialog appends file:// in front of the selection
        path = urllib.parse.unquote(path.replace("file://", ""))
        
        if os.name == "nt" and path.startswith("/"):
            path = path[1:]

        self._thread = QThread()
        self._worker = FileShufflerWorker(path)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self.fileShufflerFinished)
        self._worker.error.connect(self.fileShufflerError)

        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)

        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.error.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

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

