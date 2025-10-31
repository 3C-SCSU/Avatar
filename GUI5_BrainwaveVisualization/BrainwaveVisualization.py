import os
import sys
from pathlib import Path

from pdf2image import convert_from_path
from PySide6.QtCore import QObject, QUrl, Signal, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine


class BrainwaveVisualization(QObject):
    imagesReady = Signal(list)

    def __init__(self):
        super().__init__()
        self.image_paths = []
        # Get the parent directory of GUI5_BrainwaveVisualization
        current_dir = Path(__file__).resolve().parent
        parent_dir = current_dir.parent
        # Set the correct plots directory path
        self.plots_dir = os.path.join(parent_dir, "plotscode", "plots")
        print(f"Plots directory path: {self.plots_dir}")  # Debug print

    @Slot()
    def convert_pdfs_to_images(self):
        """Convert PDF files to images and send paths to QML"""
        self.image_paths = []
        graph_titles = [
            "Takeoff Graph",
            "Forward Graph",
            "Right Graph",
            "Landing Graph",
            "Backward Graph",
            "Left Graph",
        ]

        pdf_files = [
            "takeoff_plots.pdf",
            "forward_plots.pdf",
            "right_plots.pdf",
            "land_plots.pdf",
            "backward_plots.pdf",
            "left_plots.pdf",
        ]

        # Debug print to verify directory exists
        print(f"Checking directory exists: {os.path.exists(self.plots_dir)}")

        for i, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(self.plots_dir, pdf_file)
            print(f"Looking for PDF at: {pdf_path}")  # Debug print

            if not os.path.exists(pdf_path):
                print(f"Missing file: {pdf_path}")
                continue

            # Convert PDF to image
            images = convert_from_path(pdf_path, dpi=150)
            image_path = os.path.join(
                self.plots_dir, f"{pdf_file.replace('.pdf', '.png')}"
            )
            images[0].save(image_path, "PNG")

            print(f"Generated image: {image_path}")

            self.image_paths.append(
                {
                    "graphTitle": graph_titles[i],
                    "imagePath": QUrl.fromLocalFile(image_path).toString(),
                }
            )

        print("Final Image Paths Sent to QML:", self.image_paths)
        self.imagesReady.emit(self.image_paths)


if __name__ == "__main__":
    os.environ["QT_QUICK_CONTROLS_STYLE"] = "Fusion"
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    backend = BrainwaveVisualization()
    engine.rootContext().setContextProperty("backend", backend)
    engine.rootContext().setContextProperty("imageModel", [])

    qml_file = Path(__file__).resolve().parent / "BrainwaveVisualization.qml"
    engine.load(str(qml_file))

    backend.convert_pdfs_to_images()

    backend.imagesReady.connect(
        lambda images: engine.rootContext().setContextProperty("imageModel", images)
    )

    sys.exit(app.exec())
