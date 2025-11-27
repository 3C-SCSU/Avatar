from PySide6.QtCore import QObject, Signal, Slot

class ManualNaoController(QObject):
  logMessage = Signal(str)  # emit messages to QML

  def __init__(self, parent=None) -> None:
    super().__init__(parent)
    print("=============Manual Nao Controller initialized=============")

  @Slot(str, str)
  def connectNao(self, ip, port):
      try:
          self.logMessage.emit(f"ManualNaoController: Connecting to NAO at {ip}:{port}")
      except Exception as e:
          self.logMessage.emit(f"ManualNaoController error: {e}")