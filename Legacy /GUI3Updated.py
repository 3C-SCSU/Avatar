import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QDesktopWidget
import importlib

# Import the necessary modules
manual_drone_control_module = importlib.import_module("brainwave-prediction-app3.gui_windows3.manual_drone_control_window3")
ManDroneCont_Tab = manual_drone_control_module.ManDroneCont_Tab

transfer_files_module = importlib.import_module("brainwave-prediction-app3.gui_windows3.transfer_files_window3")
TransferFilesWindow = transfer_files_module.TransferFilesWindow

# Add your module path to the system path
module_path = '/Users/divyadarsi/Avatar/AvatarGUI3'  # Update this path
if module_path not in sys.path:
    sys.path.append(module_path)

from BrainwaveReadingTab import Ui_Avatar  # Import your generated UI class

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

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Create QTabWidget
        self.tabWidget = QTabWidget()

        # Create and setup your brainwave tab with QMainWindow, not QWidget
        brainwave_tab_widget = QMainWindow()  # Change to QMainWindow
        brainwave_tab = Ui_Avatar()
        brainwave_tab.setupUi(brainwave_tab_widget)

        # Add the brainwave tab widget to the tab widget
        self.tabWidget.addTab(brainwave_tab_widget, "Brainwave Reading")

        # Manual Control Tab
        tab2 = ManDroneCont_Tab()
        tab2.button_pressed.connect(get_drone_action)
        tab2.goHome.connect(self.go_home)

        # Transfer Data Tab
        tab3 = TransferFilesWindow()
        tab3.layout = QVBoxLayout(tab3)
        tab3.setLayout(tab3.layout)

        # Add tabs to the tab widget
        self.tabWidget.addTab(tab2, "Manual Drone Control")
        self.tabWidget.addTab(tab3, "Transfer Data")

        # Set the central widget
        self.setCentralWidget(self.tabWidget)

        # Resize the window to about 2/3 of the screen size
        screen_geometry = QDesktopWidget().screenGeometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        self.resize(int(screen_width * 2 / 3), int(screen_height * 2 / 3))
        self.setWindowTitle("Avatar Project")

    def go_home(self):
        self.tabWidget.setCurrentIndex(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
