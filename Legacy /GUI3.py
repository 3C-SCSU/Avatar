import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QDesktopWidget, QVBoxLayout
from PyQt5.QtCore import Qt
from djitellopy import Tello
import importlib

# The folder path has hyphens, so we have to load the pages with importlib
manual_drone_control_module = importlib.import_module("brainwave-prediction-app3.gui_windows3.manual_drone_control_window3")
ManDroneCont_Tab = manual_drone_control_module.ManDroneCont_Tab

transfer_files_module = importlib.import_module("brainwave-prediction-app3.gui_windows3.transfer_files_window3")
TransferFilesWindow = transfer_files_module.TransferFilesWindow

# Import the new BrainwaveReading_Tab class
from brainwave_reading_tab import BrainwaveReading_Tab

# Ensure QMainWindow is imported
tello = Tello()

def get_drone_action(action):
    if action == 'Connect':
        tello.connect()
        print("tello.connect()")
    elif action == 'Back':
        tello.move_back(30)
        print('tello.move_back(30)')
    elif action == 'Down':
        tello.move_down(30)
        print('tello.move_down(30)')
    elif action == 'Forward':
        tello.move_forward(30)
        print('tello.move_forward(30)')
    elif action == 'Land':
        tello.land()
        print('tello.land')
    elif action == 'Left':
        tello.move_left(30)
        print('tello.move_left(30)')
    elif action == 'Right':
        tello.move_right(30)
        print('tello.move_right(30)')
    elif action == 'Takeoff':
        tello.takeoff()
        print('tello.takeoff')
    elif action == 'Up':
        tello.move_up(30)
        print('tello.move_up(30)')
    elif action == 'Turn Left':
        tello.rotate_counter_clockwise(45)
        print('tello.rotate_counter_clockwise(45)')
    elif action == 'Turn Right':
        tello.rotate_clockwise(45)
        print('tello.rotate_clockwise(45)')
    elif action == 'Flip':
        tello.flip_back()
        print("tello.flip('b')")
    elif action == 'Keep Alive':
        bat = tello.query_battery()
        print(bat)
    elif action == 'Stream':
        tello.streamon()
        frame_read = tello.get_frame_read()
        while True:
            print("truu")
            img = frame_read.frame
            cv2.imshow("drone", img)

    return ("Done")

# Creates the Main Window
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Get the screen size
        screen = QDesktopWidget().screenGeometry()
        screen_width = screen.width()
        screen_height = screen.height()

        # Create the main layout
        main_layout = QVBoxLayout()

        # Create a tab widget
        self.tabWidget = QTabWidget()

        # Set blue background color for the tab widget
        self.tabWidget.setStyleSheet("""
                QTabWidget::pane {
                    background-color: #64778D;
                    border: none;
                }
                QTabBar {
                    background-color: white;
                }
                                """)

        # Create tab widgets
        # Brainwave Reading Tab
        brainwave_tab = BrainwaveReading_Tab(get_drone_action, self.use_brainflow)
        brainwave_tab.button_pressed.connect(self.handle_brainwave_action)

        # Manual Control Tab
        tab2 = ManDroneCont_Tab()
        tab2.button_pressed.connect(get_drone_action)
        tab2.goHome.connect(self.go_home)

        # Transfer Data Tab
        tab3 = TransferFilesWindow()
        tab3.layout = QVBoxLayout(tab3)
        tab3.setLayout(tab3.layout)

        # Add tabs to the tab widget
        self.tabWidget.addTab(brainwave_tab, "Brainwave Reading")
        self.tabWidget.addTab(tab2, "Manual Drone Control")
        self.tabWidget.addTab(tab3, "Transfer Data")

        # Add the tab widget to the main layout
        main_layout.addWidget(self.tabWidget)

        # Set the main layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Resize the window to about 2/3 of the screen size
        self.resize(int(screen_width * 2 / 3), int(screen_height * 2 / 3))

        self.setWindowTitle("Avatar Project")

    # Slot to handle Brainwave Tab actions
    def handle_brainwave_action(self, action):
        if action == 'read_mind':
            # Handle starting brainwave reading (e.g., initiate brainflow connection)
            self.use_brainflow()
            print("Brainwave Reading Started")
        elif action == 'not_thinking':
            # Handle "Not what I was thinking" action
            print("Not what I was thinking triggered")
        elif action == 'execute':
            # Handle the execute action
            print("Executing prediction")
        elif action == 'keep_drone_alive':
            self.get_drone_action('keep alive')

    # Placeholder function for brainwave connection
    def use_brainflow(self):
        print("Connecting to brainwave device...")
        # Add your brainwave connection code here

    # Slot to go to the Home tab
    def go_home(self):
        self.tabWidget.setCurrentIndex(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
