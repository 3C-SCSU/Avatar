import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton, QPushButton, QLabel, QTableWidget, QTableWidgetItem, QListWidget, QTextEdit

class Brainwaves(QWidget):

    signal_system = None

    def __init__(self, get_drone_action=None):
        super(Brainwaves, self).__init__()
        self.get_drone_action = get_drone_action

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
        central_widget = self
        # self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

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
        connect_button.clicked.connect(self.connect)
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
        # Placeholder for reading mind functionality
        print("Reading mind...")

    def execute_action(self):
        prediction_label = "Forward"  # Placeholder for prediction label
        self.flight_log.insert(0, prediction_label)
        self.log_list.clear()
        self.log_list.addItems(self.flight_log)
        print("Executing action...")

    def connect(self):
        print("Connecting...")

    def keep_alive(self):
        print("Keeping drone alive...")

def main():
    app = QApplication(sys.argv)
    window = Brainwaves(None)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()