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

if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    engine.addImportPath(str(Path(__file__).resolve().parent)) #so that I can call the file shuffler view class from here


    bindingContext = FileShufflerGui()
    engine.rootContext().setContextProperty("fileShufflerGui", bindingContext)

    qml_file = Path(__file__).resolve().parent / "main.qml"
    engine.load(qml_file)
    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec())
