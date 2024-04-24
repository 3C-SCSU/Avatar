from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QErrorMessage, QMessageBox
from PyQt5.QtCore import QFile, QIODevice

import sys
import os
import configparser
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "file-transfer"))
from sftp import fileTransfer

class TransferFilesWindow(QWidget):
    def __init__(self):
        super(TransferFilesWindow, self).__init__()

        self.loadUi()
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.initUi()

    def loadUi(self):
        ui_file_name = os.path.join("./", "brainwave-prediction-app3", "gui_windows3", "transfer_files_window3.ui")
        self.widget = uic.loadUi(ui_file_name, self)  # Passing self as the parent
        if not self.widget:
            print(loader.errorString())
            sys.exit(-1)

    def initUi(self):
        self.widget.save_config_button.clicked.connect(self.save_config)
        self.widget.load_config_button.clicked.connect(self.load_config)
        self.widget.clear_config_button.clicked.connect(self.clear_config)
        self.widget.upload_button.clicked.connect(self.upload)
        self.widget.private_key_dir_button.clicked.connect(self.browse_private_key_dir)
        self.widget.source_dir_button.clicked.connect(self.browse_source_dir)

    def browse_private_key_dir(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setViewMode(QFileDialog.ViewMode.List)
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                self.widget.private_key_dir_input.setText(file_paths[0])

    def browse_source_dir(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setViewMode(QFileDialog.ViewMode.List)
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                self.widget.source_dir_input.setText(file_paths[0])

    def save_config(self):
        selected_file, _ = QFileDialog.getSaveFileName(
            self,
            "Save config file",
            "",
            "INI Files (*.ini)"
        )

        if selected_file:
            if not selected_file.endswith(".ini"):
                        selected_file += ".ini"

            with open(selected_file, 'w') as configfile:
                # The login data that will be saved
                self.config['data'] = {
                "-HOST-": self.widget.host_input.text(),
                "-USERNAME-": self.widget.username_input.text(),
                "-PRIVATE_KEY-": self.widget.private_key_dir_input.text(),
                "-IGNORE_HOST_KEY-": self.widget.ignore_host_key_checkbox.isChecked(),
                "-SOURCE-": self.widget.source_dir_input.text(),
                "-TARGET-": self.widget.target_dir_input.text(),
                }
                self.config.write(configfile)

    def load_config(self):
        selected_file, _ = QFileDialog.getOpenFileName(
            self,
            "Load config file",
            "",
            "INI Files (*.ini)"
        )

        # The original login data
        oldData = {
            "-HOST-": self.widget.host_input.text(),
            "-USERNAME-": self.widget.username_input.text(),
            "-PRIVATE_KEY-": self.widget.private_key_dir_input.text(),
            "-IGNORE_HOST_KEY-": self.widget.ignore_host_key_checkbox.isChecked(),
            "-SOURCE-": self.widget.source_dir_input.text(),
            "-TARGET-": self.widget.target_dir_input.text(),
        }

        try:
            if selected_file:
                # Attempt to read the selected file
                self.config.read(selected_file)

                # Use the loaded data to set the values
                self.widget.host_input.setText(self.config["data"]["-HOST-"])
                self.widget.username_input.setText(self.config["data"]["-USERNAME-"])
                self.widget.private_key_dir_input.setText(self.config["data"]["-PRIVATE_KEY-"])
                self.widget.ignore_host_key_checkbox.setChecked(self.config["data"]["-IGNORE_HOST_KEY-"].lower() in ("true"))
                self.widget.source_dir_input.setText(self.config["data"]["-SOURCE-"])
                self.widget.target_dir_input.setText(self.config["data"]["-TARGET-"])

        except Exception as e:
            # Reset the values back to the original values
            self.widget.host_input.setText(oldData["-HOST-"])
            self.widget.private_key_dir_input.setText(oldData["-USERNAME-"])
            self.widget.private_key_dir_input.setText(oldData["-PRIVATE_KEY-"])
            self.widget.ignore_host_key_checkbox.setChecked(oldData["-IGNORE_HOST_KEY-"])
            self.widget.source_dir_input.setText(oldData["-SOURCE-"])
            self.widget.target_dir_input.setText(oldData["-TARGET-"])

            # Shows an error window
            QMessageBox.critical(None, "Loading failed", "Please ensure that your config file is not invalid or corrupted\n\nERROR:\n" + str(e))

    def clear_config(self):
        self.widget.host_input.clear()
        self.widget.username_input.clear()
        self.widget.private_key_pass_input.clear()
        self.widget.private_key_dir_input.clear()
        self.widget.ignore_host_key_checkbox.setChecked(True)
        self.widget.source_dir_input.clear()
        self.widget.target_dir_input.setText("/home/")

    def upload(self):
        try:
            svrcon = fileTransfer(
                self.host_input.text(),
                self.username_input.text(),
                self.private_key_dir_input.text(),
                self.private_key_pass_input.text(),
                self.ignore_host_key_checkbox.isChecked()
            )
            source_dir = self.source_dir_input.text()
            target_dir = self.target_dir_input.text()

            if source_dir and target_dir:
                svrcon.transfer(source_dir, target_dir)
                QMessageBox.information(None, "Upload complete")

            else:
                QMessageBox.critical(None, "Upload failed", "Please ensure that all fields have been filled!")

        except Exception as e:
            QMessageBox.critical(None, "Upload failed", "Please ensure that your inputs are correct and that the server is running\n\nERROR:\n" + str(e))