from PySide6.QtWidgets import QFileDialog, QMessageBox
from PySide6.QtCore import QObject, Slot, QProcess, QTimer
import configparser
import os
import sys
from pathlib import Path
from sftp import fileTransfer	

class CloudAPI(QObject):
    def __init__(self):
        super().__init__()
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.root_object = None
        self.opendata_process = None
        self.opendata_timer = None

    def set_root_object(self, root_object):
        self.root_object = root_object

    def connect_buttons(self):
        if self.root_object is None:
            print("Error: root_object not set in CloudAPI")
            return

        try:
            self.root_object.findChild(QObject, "saveConfigButton").clicked.connect(self.save_config)
            self.root_object.findChild(QObject, "loadConfigButton").clicked.connect(self.load_config)
            self.root_object.findChild(QObject, "clearConfigButton").clicked.connect(self.clear_config)
            self.root_object.findChild(QObject, "uploadButton").clicked.connect(self.upload)
            self.root_object.findChild(QObject, "privateKeyDirButton").clicked.connect(self.browse_private_key_dir)
            self.root_object.findChild(QObject, "sourceDirButton").clicked.connect(self.browse_source_dir)
            self.root_object.findChild(QObject, "targetDirButton").clicked.connect(self.browse_target_dir)
            
            # Open Data button
            self.root_object.findChild(QObject, "openDataButton").clicked.connect(self.browse_processed_dir)
            
            print("Cloud API buttons connected successfully")
        except Exception as e:
            print(f"Error connecting cloud API buttons: {e}")

    # Start of change : Added Cloud Computing (Transfer Data) functionality 
    @Slot()
    def browse_private_key_dir(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setViewMode(QFileDialog.ViewMode.List)
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                self.root_object.findChild(QObject, "privateKeyDirInput").setProperty("text", file_paths[0])

    @Slot()
    def browse_source_dir(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setViewMode(QFileDialog.ViewMode.List)
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                self.root_object.findChild(QObject, "sourceDirInput").setProperty("text", file_paths[0])

    @Slot()
    def browse_target_dir(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setViewMode(QFileDialog.ViewMode.List)
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                self.root_object.findChild(QObject, "targetDirInput").setProperty("text", file_paths[0])

    @Slot()
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

    @Slot()
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

    @Slot()
    def clear_config(self):
        self.root_object.findChild(QObject, "hostInput").setProperty("text", "")
        self.root_object.findChild(QObject, "usernameInput").setProperty("text", "")
        self.root_object.findChild(QObject, "privateKeyDirInput").setProperty("text", "")
        self.root_object.findChild(QObject, "ignoreHostKeyCheckbox").setProperty("checked", True)  # Reset checkbox to checked
        self.root_object.findChild(QObject, "sourceDirInput").setProperty("text", "")
        self.root_object.findChild(QObject, "targetDirInput").setProperty("text", "/home/")  # Reset to default

    @Slot()
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

    # Start of change : Added Open Data functionality 
    @Slot()
    def browse_processed_dir(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        file_dialog.setViewMode(QFileDialog.ViewMode.List)
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                processed_dir = file_paths[0]
                self.root_object.findChild(QObject, "processedDirInput").setProperty("text", processed_dir)
                
                # Change button color to yellow when directory is selected
                self.root_object.findChild(QObject, "openDataButtonBackground").setProperty("color", "#F39C12")
                
                # Check if directory contains 'processed' in name and start automatically
                if "processed" in os.path.basename(processed_dir).lower():
                    self.append_console_log(f"Selected directory: {processed_dir}")
                    self.append_console_log("Starting Open Data publishing process...")
                    # Automatically start the opendata process
                    self.start_opendata()
                else:
                    self.append_console_log("Warning: Please select a directory named 'processed'")

    @Slot()
    def start_opendata(self):
        processed_dir = self.root_object.findChild(QObject, "processedDirInput").property("text")
        
        if not processed_dir or "processed" not in os.path.basename(processed_dir).lower():
            QMessageBox.critical(None, "Error", "Please select a valid 'processed' directory first.")
            return
            
        # Check if opendata.py exists
        opendata_script = Path(__file__).parent / "file-opendata" / "opendata.py"
        if not opendata_script.exists():
            QMessageBox.critical(None, "Error", f"Open Data script not found at: {opendata_script}")
            return
            
        # Button color indicates processing state
        self.root_object.findChild(QObject, "openDataButtonBackground").setProperty("color", "#E67E22")
        # Clear console and start logging
        self.root_object.findChild(QObject, "consoleLogArea").setProperty("text", "")
        self.append_console_log("Starting Open Data publishing process...")
        self.append_console_log(f"Script: {opendata_script}")
        self.append_console_log(f"Working directory: {processed_dir}")
        
        # Setup process
        self.opendata_process = QProcess()
        self.opendata_process.readyReadStandardOutput.connect(self.read_opendata_output)
        self.opendata_process.readyReadStandardError.connect(self.read_opendata_error)
        self.opendata_process.finished.connect(self.opendata_finished)
        
        # Set working directory to the parent of processed directory
        working_dir = Path(processed_dir).parent
        
        # Run the opendata script
        try:
            self.opendata_process.setWorkingDirectory(str(working_dir))
            self.opendata_process.start(sys.executable, [str(opendata_script)])
            self.append_console_log("Process started successfully...")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to start Open Data process: {str(e)}")
            self.root_object.findChild(QObject, "openDataButtonBackground").setProperty("color", "#F39C12")

    @Slot()
    def read_opendata_output(self):
        if self.opendata_process:
            data = self.opendata_process.readAllStandardOutput()
            output = data.data().decode('utf-8', errors='ignore')
            self.append_console_log(output.strip())

    @Slot()
    def read_opendata_error(self):
        if self.opendata_process:
            data = self.opendata_process.readAllStandardError()
            error_output = data.data().decode('utf-8', errors='ignore')
            self.append_console_log(f"ERROR: {error_output.strip()}")

    @Slot()
    def opendata_finished(self, exit_code, exit_status):
        if exit_code == 0:
            self.append_console_log("\n✅ Open Data publishing completed successfully!")
            QMessageBox.information(None, "Success", "Open Data publishing completed successfully!")
        else:
            self.append_console_log(f"\n❌ Open Data publishing failed with exit code: {exit_code}")
            QMessageBox.critical(None, "Error", f"Open Data publishing failed with exit code: {exit_code}")
        
        # Reset button color to indicate completion
        self.root_object.findChild(QObject, "openDataButtonBackground").setProperty("color", "#F39C12")
        self.opendata_process = None

    
    def append_console_log(self, message):
        console_area = self.root_object.findChild(QObject, "consoleLogArea")
        if console_area:
            current_text = console_area.property("text")
            if current_text == "Console output will appear here...":
                new_text = message
            else:
                new_text = current_text + "\n" + message
            console_area.setProperty("text", new_text)
            
            # Auto-scroll to bottom
            console_area.setProperty("cursorPosition", len(new_text))

    # End of change : Added Open Data functionality 

    # End of change : Added Cloud Computing (Transfer Data) functionality 
