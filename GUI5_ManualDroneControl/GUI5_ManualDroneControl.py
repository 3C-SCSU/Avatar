import sys
from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal, QCoreApplication
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtQml import QQmlApplicationEngine, QQmlContext
from djitellopy import Tello

# Drone control class
class DroneController(QObject):
    logMessage = pyqtSignal(str)  # Signal to emit log messages

    def __init__(self):
        super().__init__()
        self.tello = Tello()
        try:
            self.tello.connect()
            self.logMessage.emit("Connected to Tello Drone")
        except Exception as e:
            self.logMessage.emit(f"Error during initialization: {e}")

    @pyqtSlot(str)
    def getDroneAction(self, action):
        try:
            if action == 'connect':
                self.tello.connect()
                self.logMessage.emit("Connected to Tello Drone")
            elif action == 'up':
                self.tello.move_up(30)
                self.logMessage.emit("Moving up")
            elif action == 'down':
                self.tello.move_down(30)
                self.logMessage.emit("Moving down")
            elif action == 'forward':
                self.tello.move_forward(30)
                self.logMessage.emit("Moving forward")
            elif action == 'backward':
                self.tello.move_back(30)
                self.logMessage.emit("Moving backward")
            elif action == 'left':
                self.tello.move_left(30)
                self.logMessage.emit("Moving left")
            elif action == 'right':
                self.tello.move_right(30)
                self.logMessage.emit("Moving right")
            elif action == 'turn_left':
                self.tello.rotate_counter_clockwise(45)
                self.logMessage.emit("Rotating left")
            elif action == 'turn_right':
                self.tello.rotate_clockwise(45)
                self.logMessage.emit("Rotating right")
            elif action == 'takeoff':
                self.tello.takeoff()
                self.logMessage.emit("Taking off")
            elif action == 'land':
                self.tello.land()
                self.logMessage.emit("Landing")
            elif action == 'go_home':
                self.go_home()
            else:
                self.logMessage.emit("Unknown action")
        except Exception as e:
            self.logMessage.emit(f"Error during {action}: {e}")

    # Method for returning to home (an approximation)
    def go_home(self):
        # Assuming the home action means moving backward and upwards
        self.tello.move_back(50)  # Move back to home point (adjust distance as needed)
        self.tello.move_up(50)    # Move up to avoid obstacles
        self.logMessage.emit("Returning to home")

if __name__ == '__main__':
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    drone_controller = DroneController()
    context = engine.rootContext()
    context.setContextProperty("droneController", drone_controller)

    # Load QML file
    engine.load('GUI5_ManualDroneControl/GUI5_ManualDroneControl.qml')

    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec_())
