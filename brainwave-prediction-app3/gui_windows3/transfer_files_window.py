import sys
import configparser
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel,
                             QFileDialog, QCheckBox, QTabWidget, QMessageBox)

from sftp import fileTransfer


class TransferDataTab(QWidget):
    def __init__(self, name='Transfer Data'):
        super().__init__()
        self.widgets: dict[str, QWidget] | None = None
        self.name = name
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        central_layout = QVBoxLayout()
        layout.addStretch(1)
        layout.addLayout(central_layout)
        layout.addStretch(1)

        self.widgets = {
            '-HOST-': QLineEdit(self),
            '-USERNAME-': QLineEdit(self),
            '-PRIVATE_KEY_PASS-': QLineEdit(self),
            '-PRIVATE_KEY-': QLineEdit(self),
            '-IGNORE_HOST_KEY-': QCheckBox('Ignore Host Key', self),
            '-SOURCE-': QLineEdit(self),
            '-TARGET-': QLineEdit(self)
        }
        self.widgets['-PRIVATE_KEY_PASS-'].setEchoMode(QLineEdit.Password)
        self.widgets['-IGNORE_HOST_KEY-'].setChecked(True)
        self.widgets['-TARGET-'].setText('/home/')

        # Labels and layout
        widget_labels = {
            '-HOST-': 'Target IP:',
            '-USERNAME-': 'Target Username',
            '-PRIVATE_KEY_PASS-': 'Private Key Password',
            '-PRIVATE_KEY-': 'Private Key Directory:',
            '-IGNORE_HOST_KEY-': '',
            '-SOURCE-': 'Source Directory:',
            '-TARGET-': 'Target Directory:'
        }

        for key, label in widget_labels.items():
            central_layout.addWidget(QLabel(label))
            if key == '-PRIVATE_KEY-':
                private_key_textbox = self.widgets[key]
                browse_button = QPushButton('Browse', self)
                browse_button.clicked.connect(lambda: self.browse_file(private_key_textbox))
                row = QHBoxLayout()
                row.addWidget(self.widgets[key])
                row.addWidget(browse_button)
                central_layout.addLayout(row)
            elif key == '-SOURCE-':
                source_textbox = self.widgets[key]
                browse_button = QPushButton('Browse', self)
                browse_button.clicked.connect(lambda: self.browse_folder(source_textbox))
                row = QHBoxLayout()
                row.addWidget(self.widgets[key])
                row.addWidget(browse_button)
                central_layout.addLayout(row)
            else:
                central_layout.addWidget(self.widgets[key])

        button_layout = QHBoxLayout()

        button = QPushButton('Save Config', self)
        button.clicked.connect(self.save_config)
        button_layout.addWidget(button)

        button = QPushButton('Load Config', self)
        button.clicked.connect(self.load_config)
        button_layout.addWidget(button)

        button = QPushButton('Clear Config', self)
        button.clicked.connect(self.clear_config)
        button_layout.addWidget(button)

        button = QPushButton('Upload', self)
        button.clicked.connect(self.upload_files)
        button_layout.addWidget(button)

        central_layout.addLayout(button_layout)

    def browse_file(self, line_edit: QLineEdit):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select File')
        if file_name:
            line_edit.setText(file_name)

    def browse_folder(self, line_edit: QLineEdit):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder_path:
            line_edit.setText(folder_path)

    def save_config(self):
        options = {
            'filter': 'INI files (*.ini)',
            'options': QFileDialog.Options(),
            'caption': 'Save configuration file'
        }
        file_name, _ = QFileDialog.getSaveFileName(self, **options)
        if file_name:
            if not file_name.endswith('.ini'):
                file_name += '.ini'
            self.config['data'] = {key: self.widgets[key].text() for key in self.widgets}
            self.config['data']['-IGNORE_HOST_KEY-'] = str(self.widgets['-IGNORE_HOST_KEY-'].isChecked())
            with open(file_name, 'w') as configfile:
                self.config.write(configfile)

    def load_config(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Load configuration file', filter='INI files (*.ini)')
        if file_name:
            self.config.read(file_name)
            for key, value in self.config['data'].items():
                if key == '-IGNORE_HOST_KEY-':
                    self.widgets[key].setChecked(value.lower() == 'true')
                else:
                    self.widgets[key].setText(value)

    def clear_config(self):
        for key in self.widgets:
            if isinstance(self.widgets[key], QCheckBox):
                self.widgets[key].setChecked(False)
            else:
                self.widgets[key].setText('')

    def upload_files(self):
        host = self.widgets['-HOST-'].text()
        username = self.widgets['-USERNAME-'].text()
        private_key = self.widgets['-PRIVATE_KEY-'].text()
        private_key_pass = self.widgets['-PRIVATE_KEY_PASS-'].text()
        ignore_host_key = self.widgets['-IGNORE_HOST_KEY-'].isChecked()
        source = self.widgets['-SOURCE-'].text()
        target = self.widgets['-TARGET-'].text()

        try:
            ft = fileTransfer(host, username, private_key, private_key_pass, ignore_host_key)
            ft.transfer(source, target)
            QMessageBox.information(self, 'Success', 'File upload completed successfully!')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Error during upload: {str(e)}')
