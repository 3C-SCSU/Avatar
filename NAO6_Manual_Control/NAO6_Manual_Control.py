from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal

class ManualNAOController(QObject):
    logMessage = pyqtSignal(str)  # emit messages to QML

    def __init__(self, parent=None):
        super().__init__(parent)
        print("=============ManualNAOController initialized=============")

    @pyqtSlot()
    def connectNao(self):
        # minimal implementation — extend to actually connect to NAO robot
        try:
            # placeholder: real connection logic goes here
            self.logMessage.emit("ManualNAOController: connectNao called")
        except Exception as e:
            self.logMessage.emit(f"ManualNAOController error: {e}")