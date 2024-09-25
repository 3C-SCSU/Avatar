from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtWidgets import QGridLayout, QRadioButton, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QWidget, QTextBrowser, QTableWidget, QStackedWidget

class BrainwaveReading_Tab(QWidget):
        def __init__(self):
                super().__init__()
                self.initUI()

        def initUI(self):

                def makeLog(log, name):
                        vbox = QVBoxLayout(log)
                        label = QLabel(name)
                        textbox = QTextBrowser()
                        textbox.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
                        vbox.addWidget(label)
                        vbox.addWidget(textbox)

                def makeButton(name, pixmap=None, padding=None):
                        button = QPushButton(name)
                        if pixmap:
                                button.setFixedSize(QSize(150,150))
                                button.setStyleSheet(f"background-image: url(brainwave-prediction-app3/images3/{pixmap}.png);")
                        else:
                                button.setStyleSheet(f"padding: {padding}px")
                        # button.clicked.connect() 
                        button.setCursor(Qt.CursorShape.PointingHandCursor)
                        return button

                gridLayout = QGridLayout()

                        #CONTROLS
                controls = QWidget()
                controls.setContentsMargins(150,0,150,0)
                controlsVBox = QVBoxLayout(controls)

                #Radio buttons
                radioButtons = QWidget()
                hbox = QHBoxLayout(radioButtons)
                manualControlRadioButton = QRadioButton("Manual Control")
                autopilotRadioButton = QRadioButton("Autopilot")
                hbox.addWidget(manualControlRadioButton)
                hbox.addWidget(autopilotRadioButton)
                controlsVBox.addWidget(radioButtons)

                #Set Manual Control checked by default
                manualControlRadioButton.setChecked(True)
                
                #Manual Control Frame
                manualControlFrame = QWidget()
                mcvbox = QVBoxLayout(manualControlFrame)
                mcvbox.setSpacing(10)

                readMyMindButton = makeButton("Read my mind", "brain")
                
                label = QLabel("The model says:")
                table1 = QTableWidget(0,2)
                table1.setHorizontalHeaderLabels(["Count","Label"])
                
                buttons = QWidget()
                hbox = QHBoxLayout(buttons)
                notButton = makeButton("Not what I was thinking",padding=25)
                executeButton = makeButton("Execute",padding=25)
                hbox.addWidget(notButton)
                hbox.addWidget(executeButton)

                keepDroneAlive = QWidget()
                hbox = QHBoxLayout(keepDroneAlive)
                textbox = QTextBrowser()
                textbox.append("-----")
                keepDroneAliveButton = makeButton("Keep Drone Alive", padding=0)
                hbox.addWidget(textbox)
                hbox.addWidget(keepDroneAliveButton)

                mcvbox.addWidget(readMyMindButton)
                mcvbox.addWidget(label)
                mcvbox.addWidget(table1)
                mcvbox.addWidget(buttons)
                mcvbox.addWidget(keepDroneAliveButton)

                #Autopilot Frame
                autopilotFrame = QWidget()
                vbox = QVBoxLayout(autopilotFrame)
                label = QLabel("This is Autopilot Frame Placeholder")
                vbox.addWidget(label)

                #Change frame according to which option is selected
                stackedWidget = QStackedWidget()
                controlsVBox.addWidget(stackedWidget)

                stackedWidget.addWidget(manualControlFrame) #Index 0 
                stackedWidget.addWidget(autopilotFrame) #Index 1 

                def updateFrame():
                        if manualControlRadioButton.isChecked():
                                stackedWidget.setCurrentIndex(0)
                        elif autopilotRadioButton.isChecked():
                                stackedWidget.setCurrentIndex(1)

                updateFrame()

                        #PREDICTION TABLE
                predictionTable = QTableWidget(0,3)
                predictionTable.setHorizontalHeaderLabels(["Predictions Count","Server Predictions","Prediction Label"])
                predictionTable.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

                #TODO Populate the predictions table with data
                def loaddata(self):
                        row = 0
        
                        #FLIGHT LOG
                flightLog = QWidget()
                makeLog(flightLog, "Flight Log")
                
                        #CONSOLE LOG
                consoleLog = QWidget()
                makeLog(consoleLog, "Console Log")

                        #CONNECT BUTTON
                connectButton = makeButton("Connect", "Connect")
                
                gridLayout.addWidget(controls,0,0)
                gridLayout.addWidget(predictionTable,0,1)
                gridLayout.addWidget(flightLog,1,0)
                gridLayout.addWidget(consoleLog,1,1)
                gridLayout.addWidget(connectButton, 2,0)

                self.setLayout(gridLayout)

                self.setStyleSheet("""
                        QPushButton {
                                background-color: #283b5b;
                                color: white;
                                background-position: center;
                                }
                        QPushButton:pressed {
                                background-color: white;
                                }
                        QLabel {
                                color: white;
                                }
                        QTableWidget{
                                background-color: #64778D;
                                border: 1px groove;
                                }
                        QTextBrowser {
                                background-color: white;
                                }
                                """)