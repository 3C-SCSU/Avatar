from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtWidgets import QGridLayout, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QWidget, QTextBrowser

class ManDroneCont_Tab(QWidget):
        #signals
        button_pressed = pyqtSignal(str)
        goHome = pyqtSignal()

        def __init__(self):
                super().__init__()
                self.initUI()

        #Making Buttons
        #the name variable is the words on the button, the filepath to the image, and the signal sent to GUI3 all at once
        def makeButton(self, name, widget, fixed):
                button = QPushButton(name, widget)
                if fixed == True:
                        button.setFixedSize(QSize(150, 150))
                else:
                        button.setMinimumSize(QSize(150, 150))
                button.clicked.connect(lambda: self.button_pressed.emit(name))
                button.setStyleSheet(f"background-image: url(brainwave-prediction-app3/images3/{name}.png);")
                return button

        def initUI(self):
                        #LAYOUT
                self.gridLayout = QGridLayout(self)
                self.gridLayout.setVerticalSpacing(6)
                self.gridLayout.setContentsMargins(6, 6, 6, 6)

                #Top
                self.top = QWidget(self)
                self.top_H = QHBoxLayout(self.top)
                self.top_H.setContentsMargins(0, 0, 0, 0)

                self.top_right = QWidget(self.top)
                self.top_right.setMaximumSize(QSize(325, 125))
                self.topright_V = QVBoxLayout(self.top_right)
                self.topright_V.setContentsMargins(0, 0, 0, 0)

                self.gridLayout.addWidget(self.top, 0, 0, 1, 1)

                #Middle
                self.middle = QWidget(self)
                self.middle.setMinimumSize(QSize(0, 150))
                self.middle_H = QHBoxLayout(self.middle)
                self.middle_H.setSpacing(6)
                self.middle_H.setContentsMargins(0, 0, 0, 0)

                self.gridLayout.addWidget(self.middle, 2, 0, 1, 1)

                #Bottom
                self.bottom = QWidget(self)
                self.bottom.setMinimumSize(QSize(0, 150))
                self.bottom_H = QHBoxLayout(self.bottom)
                self.bottom_H.setContentsMargins(0, 0, 0, 0)

                self.gridLayout.addWidget(self.bottom, 4, 0, 1, 1)


                        #BUTTONS
                #Home
                self.home = self.makeButton("Home",self.top,True)
                self.home = QPushButton("Home", self.top)
                self.home.setFixedSize(QSize(150, 150))
                self.home.clicked.connect(self.goHome.emit)
                self.home.setStyleSheet(f"background-image: url(GUI_Pics/home.png);")
                self.top_H.addWidget(self.home)
                #Up
                self.up = self.makeButton("Up",self.top,False)
                self.top_H.addWidget(self.up)
                #Forward
                self.forward = self.makeButton("Forward",self,False)
                self.gridLayout.addWidget(self.forward, 1, 0, 1, 1)
                #Turn Left
                self.turnLeft = self.makeButton("Turn Left",self.middle,False)
                self.middle_H.addWidget(self.turnLeft)
                #Left
                self.left = self.makeButton("Left",self.middle,False)
                self.middle_H.addWidget(self.left)
                #Stream
                self.stream = self.makeButton("Stream",self.middle,False)
                self.middle_H.addWidget(self.stream)
                #Right
                self.right = self.makeButton("Right",self.middle,False)
                self.middle_H.addWidget(self.right)
                #Turn Right
                self.turnRight = self.makeButton("Turn Right",self.middle,False)
                self.middle_H.addWidget(self.turnRight)
                #Back
                self.back = self.makeButton("Back",self,False)
                self.gridLayout.addWidget(self.back, 3, 0, 1, 1)
                #Connect
                self.connect = self.makeButton("Connect",self.bottom,True)
                self.bottom_H.addWidget(self.connect)
                #Down
                self.down = self.makeButton("Down",self.bottom,False)
                self.bottom_H.addWidget(self.down)
                #Takeoff
                self.takeoff = self.makeButton("Takeoff",self.bottom,True)
                self.bottom_H.addWidget(self.takeoff)
                #Land
                self.land = self.makeButton("Land",self.bottom,True)
                self.bottom_H.addWidget(self.land)

                #TODO add Log functionality
                                #LOG
                self.label = QLabel(self.top_right)
                self.label.setStyleSheet(u"color: white")
                self.label.setText("Flight Log")
                self.topright_V.addWidget(self.label)

                self.textbox = QTextBrowser(self.top_right)
                self.textbox.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
                self.textbox.setStyleSheet("background-color: white;")
                self.topright_V.addWidget(self.textbox)

                self.top_H.addWidget(self.top_right)

                self.textbox.append("------------- NEW LOG -------------")
                
                self.setStyleSheet("""
                        QPushButton {
                                background-color: #283B5B;
                                color: white;
                                background-repeat: none;
                                background-position: center;
                        }
                                   
                        QPushButton:pressed { background-color: white; }
                                   """)