import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton, QPushButton, QLabel, QTableWidget, QTableWidgetItem, QListWidget, QTextEdit

class Brainwaves(QWidget):

    signal_system = None

    def __init__(self):
        super(Brainwaves, self).__init__()

        self.keep_alive_toggle = False
        self.flight_log = []
        self.predictions_log = []
        self.predictions_headings = ['Predictions Count', 'Server Predictions', 'Prediction Label']
        self.response_headings = ['Count', 'Label']
        self.count = 0
        self.predictions_list = ['backward', 'down', 'forward', 'land', 'left', 'right', 'takeoff', 'up']
        self.action_index = 0
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        top_left_group = QGroupBox("Control")
        top_left_layout = QVBoxLayout()
        manual_control_radio = QRadioButton("Manual Control")
        autopilot_radio = QRadioButton("Autopilot")
        autopilot_radio.setChecked(True)
        read_mind_button = QPushButton("Read my mind...")
        read_mind_button.clicked.connect(self.read_mind)
        top_left_layout.addWidget(manual_control_radio)
        top_left_layout.addWidget(autopilot_radio)
        top_left_layout.addWidget(read_mind_button)
        top_left_group.setLayout(top_left_layout)

        table_group = QGroupBox("Server Response")
        table_layout = QVBoxLayout()
        self.server_table = QTableWidget()
        self.server_table.setColumnCount(len(self.response_headings))
        self.server_table.setHorizontalHeaderLabels(self.response_headings)
        self.server_table.setRowCount(1)
        table_layout.addWidget(self.server_table)
        table_group.setLayout(table_layout)

        bottom_left_group = QGroupBox("Flight Log")
        bottom_left_layout = QVBoxLayout()
        self.log_list = QListWidget()
        bottom_left_layout.addWidget(self.log_list)
        bottom_left_group.setLayout(bottom_left_layout)

        bottom_right_group = QGroupBox("Console Log")
        bottom_right_layout = QVBoxLayout()
        self.console_log = QTextEdit()
        self.console_log.setReadOnly(True)
        bottom_right_layout.addWidget(self.console_log)
        bottom_right_group.setLayout(bottom_right_layout)

        button_layout = QHBoxLayout()
        execute_button = QPushButton("Execute")
        execute_button.clicked.connect(self.execute_action)
        connect_button = QPushButton("Connect")
        connect_button.clicked.connect(self.drone_connect)  # Connect to drone method
        keep_alive_button = QPushButton("Keep Drone Alive")
        keep_alive_button.clicked.connect(self.keep_alive)
        button_layout.addWidget(execute_button)
        button_layout.addWidget(connect_button)
        button_layout.addWidget(keep_alive_button)

        main_layout.addWidget(top_left_group)
        main_layout.addWidget(table_group)
        main_layout.addWidget(bottom_left_group)
        main_layout.addWidget(bottom_right_group)
        main_layout.addLayout(button_layout)

        self.setWindowTitle("Brainwave Reading")
        self.resize(800, 600)

    def read_mind(self):
        if self.signal_system:
            # Placeholder implementation
            prediction_response = {"prediction_count": 0, "prediction_label": "Placeholder"}
            self.count = prediction_response['prediction_count']
            prediction_label = prediction_response['prediction_label']

            # Update server response table
            self.server_table.setItem(0, 0, QTableWidgetItem(str(self.count)))
            self.server_table.setItem(0, 1, QTableWidgetItem(prediction_label))
        else:
            print("Signal system not initialized.")

    def execute_action(self):
        prediction_label = self.predictions_list[self.action_index]
        self.flight_log.insert(0, prediction_label)
        self.log_list.clear()
        self.log_list.addItems(self.flight_log)
        self.get_drone_action(prediction_label)
        print("Executing action...")

    def drone_connect(self):  # Renamed the method
        # Log the connection event
        self.flight_log.insert(0, "Connect button pressed")
        self.console_log.appendPlainText("Connect button pressed")

        # Connect to the drone
        if self.signal_system:
            self.get_drone_action('Connect')
            self.flight_log.insert(0, "Done.")
            self.console_log.appendPlainText("Done.")
        else:
            print("Signal system not initialized.")

    def keep_alive(self):
        # Placeholder for keep_alive method
        pass


def main():
    app = QApplication(sys.argv)
    window = Brainwaves()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
