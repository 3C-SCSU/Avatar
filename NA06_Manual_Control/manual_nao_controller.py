from PySide6.QtCore import QObject, Signal, Slot

class ManualNaoController(QObject):
  logMessage = Signal(str)  # emit messages to QML

  def __init__(self, parent=None) -> None:
    super().__init__(parent)
    print("=============Manual Nao Controller initialized=============")

  @Slot()
  def connectNao(self):
      try:
          self.logMessage.emit("Manual Nao Controller: connectNao called")
      except Exception as e:
          self.logMessage.emit(f"Manual Nao Controller error: {e}")