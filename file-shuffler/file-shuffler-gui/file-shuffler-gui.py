# This Python file uses the following encoding: utf-8
import sys
from pathlib import Path

import subprocess

from PySide6.QtCore import QObject, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

class FileShufflerGui(QObject):
    def __init__(self):
        super().__init__()
        
    @Slot(str)
    def run_file_shuffler_program(self, id):
        process = subprocess.Popen(["python", "../run_file_shuffler.py"], stdout=subprocess.PIPE, shell=True)
        (file_shuffler_output, err) = process.communicate()

        return file_shuffler_output

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
