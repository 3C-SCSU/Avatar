from PyQt5.QtCore import QObject, pyqtSlot, QUrl
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt5.QtQml import QQmlApplicationEngine
import sys
import os
import configparser

# Assuming you have a module for file transfer
# Ensure the path is correct for your project structure
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "file-transfer"))
from sftp import fileTransfer

class TransferFilesWindow(QObject):
    def __init__(self):
        super(TransferFilesWindow, self).__init__()
        self.config = configparser.ConfigParser()
        self.config.optionxform = str

        # Load QML File
        self.engine = QQmlApplicationEngine()
        qml_file = os.path.join(os.path.dirname(__file__), "GUI5TransferData.qml")
        self.engine.load(QUrl.fromLocalFile(qml_file))

        if not self.engine.rootObjects():
            print("Error loading QML file")
            sys.exit(-1)
        else:
            print("QML loaded successfully")

        # Get the root QML object to connect signals
        self.root_object = self.engine.rootObjects()[0]

        # Debug: print all object names
        for obj in self.root_object.children():
            print(obj.objectName())  # This will help you check which objects are available

        # Connect to buttons using their object names
        self.root_object.findChild(QObject, "saveConfigButton").clicked.connect(self.save_config)
        self.root_object.findChild(QObject, "loadConfigButton").clicked.connect(self.load_config)
        self.root_object.findChild(QObject, "clearConfigButton").clicked.connect(self.clear_config)
        self.root_object.findChild(QObject, "uploadButton").clicked.connect(self.upload)
        self.root_object.findChild(QObject, "privateKeyDirButton").clicked.connect(self.browse_private_key_dir)
        self.root_object.findChild(QObject, "sourceDirButton").clicked.connect(self.browse_source_dir)
        self.root_object.findChild(QObject, "targetDirButton").clicked.connect(self.browse_target_dir)

    @pyqtSlot()
    def browse_private_key_dir(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setViewMode(QFileDialog.ViewMode.List)
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                self.root_object.findChild(QObject, "privateKeyDirInput").setProperty("text", file_paths[0])

    @pyqtSlot()
    def browse_source_dir(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setViewMode(QFileDialog.ViewMode.List)
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                self.root_object.findChild(QObject, "sourceDirInput").setProperty("text", file_paths[0])

    @pyqtSlot()
    def browse_target_dir(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setViewMode(QFileDialog.ViewMode.List)
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                self.root_object.findChild(QObject, "targetDirInput").setProperty("text", file_paths[0])

    @pyqtSlot()
    def save_config(self):
        selected_file, _ = QFileDialog.getSaveFileName(
            None,
            "Save config file",
            "",
            "INI Files (*.ini)"
        )

        if selected_file:
            if not selected_file.endswith(".ini"):
                selected_file += ".ini"

            with open(selected_file, 'w') as configfile:
                self.config['data'] = {
                    "-HOST-": self.root_object.findChild(QObject, "hostInput").property("text"),
                    "-USERNAME-": self.root_object.findChild(QObject, "usernameInput").property("text"),
                    "-PRIVATE_KEY-": self.root_object.findChild(QObject, "privateKeyDirInput").property("text"),
                    "-IGNORE_HOST_KEY-": self.root_object.findChild(QObject, "ignoreHostKeyCheckbox").property("checked"),
                    "-SOURCE-": self.root_object.findChild(QObject, "sourceDirInput").property("text"),
                    "-TARGET-": self.root_object.findChild(QObject, "targetDirInput").property("text"),
                }
                self.config.write(configfile)

    @pyqtSlot()
    def load_config(self):
        selected_file, _ = QFileDialog.getOpenFileName(
            None,
            "Load config file",
            "",
            "INI Files (*.ini)"
        )

        try:
            if selected_file:
                self.config.read(selected_file)

                self.root_object.findChild(QObject, "hostInput").setProperty("text", self.config["data"]["-HOST-"])
                self.root_object.findChild(QObject, "usernameInput").setProperty("text", self.config["data"]["-USERNAME-"])
                self.root_object.findChild(QObject, "privateKeyDirInput").setProperty("text", self.config["data"]["-PRIVATE_KEY-"])
                self.root_object.findChild(QObject, "ignoreHostKeyCheckbox").setProperty("checked", self.config["data"]["-IGNORE_HOST_KEY-"].lower() in ("true"))
                self.root_object.findChild(QObject, "sourceDirInput").setProperty("text", self.config["data"]["-SOURCE-"])
                self.root_object.findChild(QObject, "targetDirInput").setProperty("text", self.config["data"]["-TARGET-"])

        except Exception as e:
            QMessageBox.critical(None, "Loading failed", "Error: " + str(e))

    @pyqtSlot()
    def clear_config(self):
        self.root_object.findChild(QObject, "hostInput").setProperty("text", "")
        self.root_object.findChild(QObject, "usernameInput").setProperty("text", "")
        self.root_object.findChild(QObject, "passwordInput").setProperty("text", "")
        self.root_object.findChild(QObject, "privateKeyDirInput").setProperty("text", "")
        self.root_object.findChild(QObject, "ignoreHostKeyCheckbox").setProperty("checked", True)
        self.root_object.findChild(QObject, "sourceDirInput").setProperty("text", "")
        self.root_object.findChild(QObject, "targetDirInput").setProperty("text", "/home/")

    @pyqtSlot()
    def upload(self):
        try:
            svrcon = fileTransfer(
                self.root_object.findChild(QObject, "hostInput").property("text"),
                self.root_object.findChild(QObject, "usernameInput").property("text"),
                self.root_object.findChild(QObject, "privateKeyDirInput").property("text"),
                self.root_object.findChild(QObject, "passwordInput").property("text"),
                self.root_object.findChild(QObject, "ignoreHostKeyCheckbox").property("checked")
            )
            source_dir = self.root_object.findChild(QObject, "sourceDirInput").property("text")
            target_dir = self.root_object.findChild(QObject, "targetDirInput").property("text")

            if source_dir and target_dir:
                svrcon.transfer(source_dir, target_dir)
                QMessageBox.information(None, "Upload complete")
            else:
                QMessageBox.critical(None, "Upload failed", "Please ensure that all fields have been filled!")

        except Exception as e:
            QMessageBox.critical(None, "Upload failed", "Please ensure that your inputs are correct and that the server is running\n\nERROR:\n" + str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TransferFilesWindow()
    sys.exit(app.exec_())
