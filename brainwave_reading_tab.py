from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHeaderView, QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QListWidget, QLineEdit, QTextEdit, QRadioButton, QGroupBox, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QIcon, QColor

class BrainwaveReading_Tab(QWidget):
    button_pressed = pyqtSignal(str)
    
    def __init__(self, get_drone_action, use_brainflow):
        super(BrainwaveReading_Tab, self).__init__()
        self.get_drone_action = get_drone_action
        self.use_brainflow = use_brainflow
        self.predictions_log = []
        self.flight_log = []

        # Main Layout
        main_layout = QHBoxLayout()  # Main layout with left and right sections

        # Left Area - Control Mode, Brainwave Prediction, and Actions
        left_layout = QVBoxLayout()

        # Control Mode with Radio Buttons
        control_group_box = QGroupBox("")
        control_group_box.setMaximumWidth(300)
        control_group_box.setStyleSheet("color: white; background-color: #64778D; border: none;")
        radio_layout = QHBoxLayout()  # Set radio buttons side by side
        self.manual_control_radio = QRadioButton("Manual Control")
        self.autopilot_radio = QRadioButton("Autopilot")
        self.manual_control_radio.setChecked(True)  # Default to Manual Control
        self.manual_control_radio.setStyleSheet("color: white;")
        self.autopilot_radio.setStyleSheet("color: white;")
        radio_layout.addWidget(self.manual_control_radio)
        radio_layout.addWidget(self.autopilot_radio)
        control_group_box.setLayout(radio_layout)

        # Brainwave Image and Button to "Read my mind..."
        brainwave_button_layout = QVBoxLayout()
        brainwave_image = QLabel()
        pixmap = QPixmap("brainwave-prediction-app/images/brain.png")  # Set brain image
        brainwave_image.setPixmap(pixmap)
        brainwave_image.setFixedSize(120, 120)
        brainwave_image.setScaledContents(True)
        brainwave_image.setAlignment(Qt.AlignCenter)

        read_mind_button = QPushButton("Read my mind...")
        read_mind_button.setIcon(QIcon(pixmap))  # Set icon as brain image
        read_mind_button.setFixedSize(160, 40)
        read_mind_button.setStyleSheet("background-color: #1b3a4b; color: white; border-radius: 5px;")
        read_mind_button.setToolTip("Click to read your mind") # Set tooltip for the read my mind button
        read_mind_button.clicked.connect(self.read_mind)

        brainwave_button_layout.addWidget(brainwave_image, alignment=Qt.AlignCenter)
        brainwave_button_layout.addWidget(read_mind_button, alignment=Qt.AlignCenter)

        # Model Prediction Section
        model_says_label = QLabel("The model says ...")
        model_says_label.setAlignment(Qt.AlignLeft)
        model_says_label.setStyleSheet("color: white; font-size: 14px;")
        model_says_label.setFixedHeight(20) #set the height of the label widget to render as table heading

        # Server Response Table
        self.server_table = QTableWidget(1, 2)
        self.server_table.setHorizontalHeaderLabels(['Count', 'Label'])
        self.server_table.setFixedHeight(200)
        self.server_table.setFixedWidth(400)
        self.server_table.setStyleSheet("background-color: #1b3a4b; border: 1px solid white;  gridline-color: white;") # add gridline-color to set the white borders inside the table
        self.server_table.verticalHeader().setVisible(False) # hide default row headers index

        header_server_table = self.server_table.horizontalHeader()
        header_server_table.setSectionResizeMode(QHeaderView.Stretch) # stretch the header to fill up the specified space

        # Model Prediction Section Layout with Reduced Gap
        model_prediction_layout = QVBoxLayout()
        model_prediction_layout.setSpacing(5)  # Reduce spacing between components
        model_prediction_layout.addWidget(model_says_label, alignment=Qt.AlignLeft)
        model_prediction_layout.addWidget(self.server_table)

        # Action Buttons Layout
        button_layout = QHBoxLayout()
        not_thinking_button = QPushButton("Not what I was thinking...")
        execute_button = QPushButton("Execute")
        execute_button.setToolTip('Click to execute action') # Set tooltip for execute button

        not_thinking_button.setFixedSize(180, 40)
        not_thinking_button.setStyleSheet("background-color: #1b3a4b; color: white; border-radius: 5px;")
        execute_button.setFixedSize(160, 40)
        execute_button.setStyleSheet("background-color: #1b3a4b; color: white; border-radius: 5px;")

        not_thinking_button.clicked.connect(self.not_thinking)
        execute_button.clicked.connect(self.execute_prediction)

        button_layout.addWidget(not_thinking_button, alignment=Qt.AlignLeft)
        button_layout.addWidget(execute_button, alignment=Qt.AlignLeft)

        # Manual Input Field and Keep Drone Alive Button in One Line
        manual_input_and_keep_alive_layout = QHBoxLayout()
        empty_input = QLineEdit()
        empty_input.setPlaceholderText("Manual Command (Optional)")
        empty_input.setFixedWidth(300)
        empty_input.setFixedHeight(40) # set height of the field to match with button
        empty_input.setStyleSheet("background-color: #1b3a4b; color: white; border: 1px solid white;")

        keep_drone_alive_button = QPushButton("Keep Drone Alive")
        keep_drone_alive_button.setFixedSize(160, 40)
        keep_drone_alive_button.setStyleSheet("background-color: #1b3a4b; color: white; border-radius: 5px;")
        keep_drone_alive_button.clicked.connect(lambda: self.get_drone_action('keep alive'))

        manual_input_and_keep_alive_layout.addWidget(empty_input, alignment=Qt.AlignLeft)
        manual_input_and_keep_alive_layout.addWidget(keep_drone_alive_button, alignment=Qt.AlignLeft)

        # Flight Log Section
        flight_log_label = QLabel("Flight Log")
        flight_log_label.setStyleSheet("color: white; font-size: 14px; margin-top: 14px;") # add margin-top to align with right layout console log
        flight_log_label.setFixedHeight(40) # add fixed height for label so it will render as heading
        self.flight_log_list = QListWidget()
        self.flight_log_list.setFixedSize(250, 150)
        self.flight_log_list.setStyleSheet("background-color: #1b3a4b; color: white; border: 1px solid white;")

        # Connect Button Section
        connect_button = QPushButton("Connect")
        connect_pixmap = QPixmap("brainwave-prediction-app/images/connect.png")  # Set connect image
        connect_button.setIcon(QIcon(connect_pixmap))
        connect_button.setFixedSize(150, 50)
        connect_button.setToolTip('Click to connect to the drone') # tooltip for connect button
        connect_button.setStyleSheet("background-color: #1b3a4b; color: white; border-radius: 5px;")
        connect_button.clicked.connect(self.connect_drone)

        # Add all components to left layout in the given order
        left_layout.addWidget(control_group_box, alignment=Qt.AlignLeft)
        left_layout.addLayout(brainwave_button_layout)
        left_layout.addLayout(model_prediction_layout)
        left_layout.addLayout(button_layout)
        left_layout.addLayout(manual_input_and_keep_alive_layout)  # Add manual input and keep alive button layout
        left_layout.addWidget(flight_log_label)
        left_layout.addWidget(self.flight_log_list)
        left_layout.addWidget(connect_button, alignment=Qt.AlignLeft)

        # Right Area - Predictions Table and Console Log
        right_layout = QVBoxLayout()

        # Predictions Table
        self.predictions_table = QTableWidget(0, 3)
        self.predictions_table.setHorizontalHeaderLabels(['Predictions Count', 'Server Predictions', 'Prediction Label'])
        self.predictions_table.setFixedSize(800, 542) # Set to 542 height to match with the contents on left layout
        self.predictions_table.setStyleSheet("background-color: #1b3a4b; border: 1px solid white;  gridline-color: white;")

        # Use QHeaderView to stretch the headers to fill up the fixed width set for the table
        header_predictions_table = self.predictions_table.horizontalHeader() 
        header_predictions_table.setSectionResizeMode(QHeaderView.Stretch)
        
        # Console Log
        console_log_layout = QVBoxLayout()
        console_log_label = QLabel("Console Log")
        console_log_label.setAlignment(Qt.AlignRight)
        console_log_label.setStyleSheet("color: white; font-size: 14px; margin-top: 14px;") # Set margin top to align with flight log label on left layout
        console_log_label.setFixedHeight(40) # Set fixed height for label to render as heading above the console log
        self.console_log = QTextEdit()
        self.console_log.setFixedSize(400, 200)
        self.console_log.setStyleSheet("background-color: #1b3a4b; color: white; border: 1px solid white;")

        # Add console log label and text box to console log layout
        console_log_layout.addWidget(console_log_label, alignment=Qt.AlignRight)
        console_log_layout.addWidget(self.console_log, alignment=Qt.AlignRight)

        # Add components to right layout
        right_layout.addWidget(self.predictions_table)
        right_layout.addLayout(console_log_layout)

        # Combine left and right layouts
        main_layout.addLayout(left_layout)
        main_layout.addStretch()
        main_layout.addLayout(right_layout)

        # Set the main layout for the widget and set the background color
        self.setLayout(main_layout)
        self.setStyleSheet("background-color: #64778D;")

    def read_mind(self):
        """Handle the 'Read my mind' action."""
        prediction_response = self.use_brainflow()
        self.prediction_label = prediction_response['prediction_label']
        count = prediction_response['prediction_count']

        # Adding items with center alignment
        count_item = QTableWidgetItem(str(count))
        label_item = QTableWidgetItem(self.prediction_label)

        # Set text alignment to center
        count_item.setTextAlignment(Qt.AlignCenter)
        label_item.setTextAlignment(Qt.AlignCenter)
        
        # Set text color to white
        count_item.setForeground(QColor(255, 255, 255))  
        label_item.setForeground(QColor(255, 255, 255))  

        # Update the server table with the new data
        self.server_table.setItem(0, 0, count_item)
        self.server_table.setItem(0, 1, label_item)

    def not_thinking(self):
        """Handle 'Not what I was thinking' button."""
        drone_input = self.drone_input.text() if self.drone_input.text() else "manual input"
        self.get_drone_action(drone_input)

        prediction_record = [len(self.predictions_log) + 1, "manual", drone_input]
        self.predictions_log.append(prediction_record)

        # Update the predictions table
        self.predictions_table.setRowCount(len(self.predictions_log))
        for i, record in enumerate(self.predictions_log):
            for j, data in enumerate(record):
                item = QTableWidgetItem(str(data))
                 # Set text alignment to center
                item.setTextAlignment(Qt.AlignCenter)
                # Set text color to white
                item.setForeground(QColor(255, 255, 255))  
                # Set the item in the predictions table
                self.predictions_table.setItem(i, j, item)

    def execute_prediction(self):
        """Handle 'Execute' button."""
        self.flight_log.append(self.prediction_label)
        self.flight_log_list.addItem(self.prediction_label)
        self.get_drone_action(self.prediction_label)
        self.console_log.append("Executed action: " + self.prediction_label)

    def connect_drone(self):
        """Handle the 'Connect' button."""
        self.flight_log.append("Connecting to drone...")
        self.flight_log_list.addItem("Connecting to drone...")
        self.get_drone_action("connect")
        self.flight_log.append("Connected.")
        self.flight_log_list.addItem("Connected.")