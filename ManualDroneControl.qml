import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtQuick3D 6.7
import QtQuick.Dialogs
import Qt.labs.platform
import "GUI5_ManualDroneControl/cameraview"

// Manual Drone Control Tab view
Rectangle {
    color: "#718399"

    // connection
    Connections {
        target: backend
        function onLogMessage(message){
            if(flightLogTextArea.text = ""){
                flightLogTextArea.text =message;
            } else{
                flightLogTextArea.text = message + "\n" + flightLogTextArea.text;
            }
        }
    }

    Row {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 20

        // Left side - Drone Controls
        Rectangle {
            width: parent.width * 0.65
            height: parent.height
            color: "transparent"

            Column {
                anchors.fill: parent
                spacing: 5

                // Top Row - Home, Up, Flight Log
                Row {
                    width: parent.width
                    height: parent.height * 0.19
                    anchors.top: parent.top
                    anchors.topMargin: parent.height * 0.0
                    spacing: parent.width * 0.1
                    // Home Button
                    Rectangle {
                        id: homeButton
                        width: parent.width * 0.15
                        height: parent.height
                        anchors.left: parent.left
                        color: "#242c4d"
                        border.color: "black"

                        Image {
                            source: "GUI_Pics/home.png"
                            width: 150
                            height: 150
                            anchors.centerIn: parent
                        }

                        Text {
                            text: "Home"
                            font.bold: true
                            color: "white"
                            font.pixelSize: Math.max(12, parent.width * 0.05)
                            anchors.horizontalCenter: parent.horizontalCenter
                            anchors.bottom: parent.bottom
                            anchors.bottomMargin: 10
                        }

                        MouseArea {
                            anchors.fill:parent
                            onEntered: homeButton.color ="white"
                            onExited: homeButton.color = "#242c4d"
                            onClicked: backend.getDroneAction("home")
                            }
                            
                            
                    }

                    // Up Button
                    Rectangle {
                        id: upButton
                        width: parent.width * 0.6
                        height: parent.height
                        anchors.horizontalCenter: parent.horizontalCenter
                        color: "#242c4d"
                        border.color: "black"

                        Image {
                            source: "GUI_Pics/up.png"
                            width: 150
                            height: 150
                            anchors.centerIn: parent
                        }

                        Text {
                            text: "Up"
                            font.bold: true
                            color: "white"
                            font.pixelSize: Math.max(12, parent.width * 0.01)
                            anchors.horizontalCenter: parent.horizontalCenter
                            anchors.bottom: parent.bottom
                            anchors.bottomMargin: 10
                        }

                        MouseArea {
                            anchors.fill: parent
                            onEntered: upButton.color = "white"
                            onExited: upButton.color = "#242c4d"
                            onClicked: backend.getDroneAction("up");
                            
                            
                            
                        }
                    }

                    // Flight Log
                    Rectangle {
                        width: parent.width * 0.4
                        height: parent.height
                        anchors.right: parent.right
                        color: "black"
                        border.color: "#2E4053"

                        Column{
                            anchors.fill:parent
                            anchors.margins:5
                        

                        Text {
                            text: "Flight Log"
                            font.bold: true
                            font.pixelSize: Math.max(12, parent.width * 0.05)
                            color: "white"
                            anchors.horizontalCenter: parent.horizontalCenter
                            
                        }

                        ScrollView {
                            width: parent.width
                            height:parent.height - 30
                            clip: true
                        

                        TextArea {
                            id: flightLogTextArea
                            font.pixelSize: Math.max(10, parent.width * 0.03)
                            color: "white"
                            readOnly: true
                            wrapMode: TextArea.Wrap
                            selectByMouse: true
                        }
                    }
                        
                    }
                        
                }
            }

                // Forward Button
                Rectangle {
                     
                    width: parent.width
                    height: parent.height * 0.19
                    anchors.top: parent.top
                    anchors.topMargin: parent.height * 0.20
                    color: "transparent"

                    Rectangle {
                        id: forwardButton
                        width: parent.width
                        height: parent.height
                        anchors.horizontalCenter: parent.horizontalCenter
                        color: "#242c4d"
                        border.color: "black"

                        Image {
                            source: "GUI_Pics/Forward.png"
                            width: 150
                            height: 150
                            anchors.centerIn: parent
                        }

                        Text {
                            text: "Forward"
                            font.bold: true
                            color: "white"
                            font.pixelSize: parent.width * 0.01
                            anchors.horizontalCenter: parent.horizontalCenter
                            anchors.bottom: parent.bottom
                            anchors.bottomMargin: 10
                        }

                        MouseArea {
                            anchors.fill: parent
                            onEntered: forwardButton.color = "white"
                            onExited: forwardButton.color = "#242c4d"
                            onClicked: backend.getDroneAction("forward")
                        }
                    }
                }

                // Directional Buttons (Turn Left, Left, Stream, Right, Turn Right)
                Row {
                    width: parent.width
                    height: parent.height * 0.18
                    anchors.top: parent.top
                    anchors.topMargin: parent.height * 0.4
                    spacing: width * 0.065

                    // Turn Left Button
                    Rectangle {
                        id: turnLeftButton
                        width: parent.width * 0.15
                        height: parent.height
                        color: "#242c4d"
                        border.color: "black"

                        Image {
                            source: "GUI_Pics/turnLeft.png"
                            width: 150
                            height: 150
                            anchors.centerIn: parent
                        }

                        Text {
                            text: "Turn Left"
                            font.bold: true
                            color: "white"
                            font.pixelSize: Math.max(12, width * 0.2)
                            anchors.horizontalCenter: parent.horizontalCenter
                            anchors.bottom: parent.bottom
                            anchors.bottomMargin: 10
                        }

                        MouseArea {
                            anchors.fill: parent
                            onEntered: turnLeftButton.color = "white"
                            onExited: turnLeftButton.color = "#242c4d"
                            onClicked: backend.getDroneAction("turn_left")
                        }
                    }

                    // Left Button
                    Rectangle {
                        id: leftButton
                        width: parent.width * 0.15
                        height: parent.height
                        color: "#242c4d"
                        border.color: "black"

                        Image {
                            source: "GUI_Pics/left.png"
                            width: 150
                            height: 150
                            anchors.centerIn: parent
                        }

                        Text {
                            text: "Left"
                            font.bold: true
                            color: "white"
                            font.pixelSize: Math.max(12, width * 0.2)
                            anchors.horizontalCenter: parent.horizontalCenter
                            anchors.bottom: parent.bottom
                            anchors.bottomMargin: 10
                        }

                        MouseArea {
                            anchors.fill: parent
                            onEntered: leftButton.color = "white"
                            onExited: leftButton.color = "#242c4d"
                            onClicked: backend.getDroneAction("left")
                        }
                    }

                    // Stream Button
                    Rectangle {
                        id: streamButton
                        width: parent.width * 0.15
                        height: parent.height
                        color: "#242c4d"
                        border.color: "black"

                        Image {
                            source: "GUI_Pics/Stream.png"
                            width: 150
                            height: 150
                            anchors.centerIn: parent
                        }

                        Text {
                            text: "Stream"
                            font.bold: true
                            color: "white"
                            font.pixelSize: Math.max(12, width * 0.2)
                            anchors.horizontalCenter: parent.horizontalCenter
                            anchors.bottom: parent.bottom
                            anchors.bottomMargin: 10
                        }

                        MouseArea {
                            anchors.fill: parent
                            onEntered: streamButton.color = "white"
                            onExited: streamButton.color = "#242c4d"
                            onClicked: backend.getDroneAction("stream")
                        }
                    }

                    // Right Button
                    Rectangle {
                        id: rightButton
                        width: parent.width * 0.15
                        height: parent.height
                        color: "#242c4d"
                        border.color: "black"

                        Image {
                            source: "GUI_Pics/right.png"
                            width: 150
                            height: 150
                            anchors.centerIn: parent
                        }

                        Text {
                            text: "Right"
                            font.bold: true
                            color: "white"
                            font.pixelSize: Math.max(12, width * 0.2)
                            anchors.horizontalCenter: parent.horizontalCenter
                            anchors.bottom: parent.bottom
                            anchors.bottomMargin: 10
                        }

                        MouseArea {
                            anchors.fill: parent
                            onEntered: rightButton.color = "white"
                            onExited: rightButton.color = "#242c4d"
                            onClicked: backend.getDroneAction("right")
                        }
                    }

                    // Turn Right Button
                    Rectangle {
                        id: turnRightButton
                        width: parent.width * 0.15
                        height: parent.height
                        color: "#242c4d"
                        border.color: "black"

                        Image {
                            source: "GUI_Pics/turnRight.png"
                            width: 150
                            height: 150
                            anchors.centerIn: parent
                        }

                        Text {
                            text: "Turn Right"
                            font.bold: true
                            color: "white"
                            font.pixelSize: Math.max(12, width * 0.2)
                            anchors.horizontalCenter: parent.horizontalCenter
                            anchors.bottom: parent.bottom
                            anchors.bottomMargin: 10
                        }

                        MouseArea {
                            anchors.fill: parent
                            onEntered:turnRightButton.color = "white"
                            onExited: turnRightButton.color = "#242c4d"
                            onClicked: backend.getDroneAction("turn_right")
                        }
                    }
                }

                // Back Button
                Rectangle {
                    width: parent.width
                    height: parent.height * 0.18
                    anchors.top: parent.top
                    anchors.topMargin: parent.height * 0.6
                    color: "transparent"

                    Rectangle {
                        id: backButton
                        width: parent.width
                        height: parent.height
                        anchors.horizontalCenter: parent.horizontalCenter
                        color: "#242c4d"
                        border.color: "black"

                        Image {
                            source: "GUI_Pics/back.png"
                            width: 150
                            height: 150
                            anchors.centerIn: parent
                        }

                        Text {
                            text: "Back"
                            font.bold: true
                            color: "white"
                            font.pixelSize: parent.width * 0.01
                            anchors.horizontalCenter: parent.horizontalCenter
                            anchors.bottom: parent.bottom
                            anchors.bottomMargin: 10
                        }

                        MouseArea {
                            anchors.fill: parent
                            onEntered: backButton.color = "white"
                            onExited: backButton.color = "#242c4d"
                            onClicked: backend.getDroneAction("backward")
                        }
                    }
                }

                // Connect, Down, Takeoff, Land Buttons
                Rectangle {
                    width: parent.width
                    height: parent.height * 0.20
                    anchors.top: parent.top
                    anchors.topMargin: parent.height * 0.8
                    color: "transparent"

                    Row {
                        width: parent.width
                        height: parent.height
                        spacing: parent.width * 0.0165

                        // Connect Button
                        Rectangle {
                            id: connectButton
                            width: parent.width * 0.15
                            height: parent.height
                            color: "#242c4d"
                            border.color: "black"

                            Image {
                                source: "GUI_Pics/connect.png"
                                width: 150
                                height: 150
                                anchors.centerIn: parent
                            }

                            Text {
                                text: "Connect"
                                font.bold: true
                                color: "white"
                                font.pixelSize: Math.max(12, parent.width * 0.05)
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 10
                            }

                            MouseArea {
                                anchors.fill: parent
                                onEntered: connectButton.color = "white"
                                onExited: connectButton.color = "#242c4d"
                                onClicked: backend.getDroneAction("connect")
                            }
                        }

                        // Down Button
                        Rectangle {
                            id: downButton
                            width: parent.width * 0.5
                            height: parent.height
                            color: "#242c4d"
                            border.color: "black"

                            Image {
                                source: "GUI_Pics/down.png"
                                width: 150
                                height: 150
                                anchors.centerIn: parent
                            }

                            Text {
                                text: "Down"
                                font.bold: true
                                color: "white"
                                font.pixelSize: Math.max(12, parent.width * 0.01)
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 10
                            }

                            MouseArea {
                                anchors.fill: parent
                                onEntered: downButton.color = "white"
                                onExited: downButton.color = "#242c4d"
                                onClicked: backend.getDroneAction("down")
                            }
                        }

                        // Takeoff Button
                        Rectangle {
                            id: takeoffButton
                            width: parent.width * 0.15
                            height: parent.height
                            color: "#242c4d"
                            border.color: "black"

                            Image {
                                source: "GUI_Pics/takeoff.png"
                                width: 150
                                height: 150
                                anchors.centerIn: parent
                            }

                            Text {
                                text: "Takeoff"
                                font.bold: true
                                color: "white"
                                font.pixelSize: Math.max(12, parent.width * 0.05)
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 10
                            }

                            MouseArea {
                                anchors.fill: parent
                                onEntered:takeoffButton.color = "white"
                                onExited: takeoffButton.color = "#242c4d"
                                onClicked: backend.getDroneAction("takeoff")
                            }
                        }

                        // Land Button
                        Rectangle {
                            id: landButton
                            width: parent.width * 0.15
                            height: parent.height
                            color: "#242c4d"
                            border.color: "black"

                            Image {
                                source: "GUI_Pics/land.png"
                                width: 150
                                height: 150
                                anchors.centerIn: parent
                            }

                            Text {
                                text: "Land"
                                font.bold: true
                                color: "white"
                                font.pixelSize: Math.max(12, parent.width * 0.05)
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 10
                            }

                            MouseArea {
                                anchors.fill: parent
                                onEntered: landButton.color = "white"
                                onExited: landButton.color = "#242c4d"
                                onClicked: backend.getDroneAction("land")
                            }
                        }
                    }
                }
            }
        }

        // Right side of Camera View
        CameraView {
            width: parent.width * 0.3
            height: parent.height
            cameraController: cameraController
        }
    }
}
