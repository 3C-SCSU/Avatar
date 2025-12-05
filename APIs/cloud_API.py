import configparser
from sftp import fileTransfer	


    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str

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

    # End of change : Added Cloud Computing (Transfer Data) functionality 
