# This Python file uses the following encoding: utf-8
import sys
import subprocess
from pathlib import Path

from PySide6.QtCore import QObject, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

class FileShufflerGui(QObject):
    def __init__(self):
        super().__init__()
        
    @Slot(result=str)
    def run_file_shuffler_program(self):
        process = subprocess.Popen(
            ["py", "./run_file_shuffler.py"], #todo - handle `py` not working (an instance where python3 or python works but not py)
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            shell=True, 
            text=True # Ensure output is returned as string
        )
        stdout, stderr = process.communicate()

        # Concatenate stdout and stderr
        return stdout + stderr

if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    bindingContext = FileShufflerGui()
    engine.rootContext().setContextProperty("fileShufflerGui", bindingContext)

    qml_file = Path(__file__).resolve().parent / "main.qml"
    engine.load(qml_file)
    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec())
