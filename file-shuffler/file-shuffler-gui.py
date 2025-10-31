# This Python file uses the following encoding: utf-8
import sys
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QObject, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine


class FileShufflerGui(QObject):
    def __init__(self):
        super().__init__()

    @Slot(str)
    def shuffle_csv_file(self, file_path):
        # Fix for file:/// prefix from QML
        if file_path.startswith("file:///"):
            file_path = file_path.replace("file:///", "")

        print("Received file path:", file_path)

        try:
            if not file_path.endswith(".csv"):
                print("Not a CSV file. Ignored.")
                return

            # Read the CSV file
            df = pd.read_csv(file_path)
            print("CSV file read. First few rows:")
            print(df.head())

            # Shuffle the data
            shuffled_df = df.sample(frac=1).reset_index(drop=True)

            # Save the shuffled file
            output_path = file_path.replace(".csv", "_shuffled.csv")
            shuffled_df.to_csv(output_path, index=False)

            print(f"Shuffled CSV saved at: {output_path}")

        except Exception as e:
            print("Error processing CSV:", str(e))


if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    engine.addImportPath(str(Path(__file__).resolve().parent))

    bindingContext = FileShufflerGui()
    engine.rootContext().setContextProperty("fileShufflerGui", bindingContext)

    qml_file = Path(__file__).resolve().parent / "main.qml"
    print("Trying to load QML from:", qml_file)

    if not qml_file.exists():
        print("ERROR: main.qml not found at path:", qml_file)
        sys.exit(1)

    engine.load(qml_file)
    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec())
