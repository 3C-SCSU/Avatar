# This Python file uses the following encoding: utf-8
import sys
import subprocess
from pathlib import Path

from PySide6.QtCore import QObject, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

import run_file_shuffler

class FileShufflerView(QObject):
    def __init__(self):
        super().__init__()

    @Slot(str, result=str)
    def run_file_shuffler_program(self, path):
        # Need to parse the path as the FolderDialog appends file:// in front of the selection
        path = path.replace("file://", "")
        response = run_file_shuffler.main(path)
        return response

if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    bindingContext = FileShufflerView()
    engine.rootContext().setContextProperty("fileShufflerView", bindingContext)

    qml_file = Path(__file__).resolve().parent / "file-shuffler-view.qml"
    engine.load(qml_file)
    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec())
