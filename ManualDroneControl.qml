import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtQuick3D 6.7
import QtQuick.Dialogs
import Qt.labs.platform
import "cameraview"

// Manual Drone Control Tab view
Rectangle {
    color: "#718399"

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
                            anchors.fill: parent
                            onEntered: {
                                buttonBackground.color = "white";
                                buttonText.color = "black";
                            }
                            onExited: {
                                buttonBackground.color = "#242c4d";
                                buttonText.color = "white";
                            }
                            onClicked: {
                                backend.doDroneTAction("go_home");
                            }
                        }
                    }

                    // Up Button
                    Rectangle {
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
                            onEntered: {
                                upButtonBackground.color = "white";
                                upButtonText.color = "black";
                            }
                            onExited: {
                                upButtonBackground.color = "#242c4d";
                                upButtonText.color = "white";
                            }
                            onClicked: {
                                backend.doDroneTAction("up");
                            }
                        }
                    }

                    // Flight Log
                    Rectangle {
                        width: parent.width * 0.2
                        height: parent.height
                        anchors.right: parent.right
                        color: "#ffffff"
                        border.color: "#2E4053"

                        Text {
                            text: "Flight Log"
                            font.bold: true
                            font.pixelSize: Math.max(10, parent.width * 0.045) // slightly smaller
                            color: "black"
                            anchors.horizontalCenter: parent.horizontalCenter
                            anchors.top: parent.top
                            anchors.topMargin: 10
                        }

                        TextArea {
                            id: flightLog
                            readOnly: true
                            wrapMode: TextArea.Wrap
                            font.pixelSize: Math.max(10, parent.width * 0.04)
                            color: "black"
                            anchors.top: parent.top
                            anchors.topMargin: 40
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.bottom: parent.bottom
                            clip: true

                            background: Rectangle { color: "white" }

                            Component.onCompleted: {
                                backend.flightLogUpdated.connect(function(logList) {
                                    flightLog.text = logList.join("\n")
                                })
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
                        width: parent.width
                        height: parent.height
                        anchors.horizontalCenter: parent.horizontalCenter
                        color: "#242c4d"
                        border.color: "black"

                        Image {
                            source: "GUI_Pics/forward.png"
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
                            onEntered: parent.color = "white"
                            onExited: parent.color = "#242c4d"
                            onClicked: backend.doDroneTAction("forward")
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
                            onEntered: parent.color = "white"
                            onExited: parent.color = "#242c4d"
                            onClicked: backend.doDroneTAction("turn_left")
                        }
                    }

                    // Left Button
                    Rectangle {
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
                            onEntered: parent.color = "white"
                            onExited: parent.color = "#242c4d"
                            onClicked: backend.doDroneTAction("left")
                        }
                    }

                    // Stream Button
                    Rectangle {
                        width: parent.width * 0.15
                        height: parent.height
                        color: "#242c4d"
                        border.color: "black"

                        Image {
                            source: "GUI_Pics/stream.png"
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
                            onEntered: parent.color = "white"
                            onExited: parent.color = "#242c4d"
                            onClicked: backend.doDroneTAction("stream")
                        }
                    }

                    // Right Button
                    Rectangle {
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
                            onEntered: parent.color = "white"
                            onExited: parent.color = "#242c4d"
                            onClicked: backend.doDroneTAction("right")
                        }
                    }

                    // Turn Right Button
                    Rectangle {
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
                            onEntered: parent.color = "white"
                            onExited: parent.color = "#242c4d"
                            onClicked: backend.doDroneTAction("turn_right")
                        }
                    }
                }
                // Back Button Row with Flip Buttons
                Rectangle {
                    width: parent.width
                    height: parent.height * 0.18
                    anchors.top: parent.top
                    anchors.topMargin: parent.height * 0.6
                    color: "transparent"
                    
                    Row {
                        width: parent.width
                        height: parent.height
                        spacing: parent.width * 0.0165
                // Flip Forward Button
                        Rectangle {
                            width: (parent.width - parent.spacing * 4) / 5
                            height: parent.height
                            color: "#242c4d"
                            border.color: "black"
                            
                            Text {
                                text: "↻"
                                font.bold: true
                                color: "#76ff03"
                                font.pixelSize: 80
                                anchors.centerIn: parent
                                anchors.verticalCenterOffset: -15
                            }
                            
                            Text {
                                text: "Flip Forward"
                                font.bold: true
                                color: "white"
                                font.pixelSize: 11
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 5
                            }
                            
                            MouseArea {
                                anchors.fill: parent
                                onEntered: parent.color = "white"
                                onExited: parent.color = "#242c4d"
                                onClicked: backend.doDroneTAction("flip_forward")
                            }
                        }
                        
                        // Flip Back Button
                        Rectangle {
                            width: (parent.width - parent.spacing * 4) / 5
                            height: parent.height
                            color: "#242c4d"
                            border.color: "black"
                            
                            Text {
                                text: "↺"
                                font.bold: true
                                color: "#76ff03"
                                font.pixelSize: 80
                                anchors.centerIn: parent
                                anchors.verticalCenterOffset: -15
                            }
                            
                            Text {
                                text: "Flip Back"
                                font.bold: true
                                color: "white"
                                font.pixelSize: 11
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 5
                            }
                            
                            MouseArea {
                                anchors.fill: parent
                                onEntered: parent.color = "white"
                                onExited: parent.color = "#242c4d"
                                onClicked: backend.doDroneTAction("flip_back")
                            }
                        }
                        
                        // Back Button (Center)
                        Rectangle {
                            width: (parent.width - parent.spacing * 4) / 5
                            height: parent.height
                            color: "#242c4d"
                            border.color: "black"
                            Image {
                                source: "GUI_Pics/back.png"
                                width: 100
                                height: 100
                                anchors.centerIn: parent
                            }
                            Text {
                                text: "Back"
                                font.bold: true
                                color: "white"
                                font.pixelSize: 11
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 5
                            }
                            MouseArea {
                                anchors.fill: parent
                                onEntered: parent.color = "white"
                                onExited: parent.color = "#242c4d"
                                onClicked: backend.doDroneTAction("backward")
                            }
                        }
                        
                        // Flip Left Button
                        Rectangle {
                            width: (parent.width - parent.spacing * 4) / 5
                            height: parent.height
                            color: "#242c4d"
                            border.color: "black"
                            
                            Text {
                                text: "⟲"
                                font.bold: true
                                color: "#76ff03"
                                font.pixelSize: 150
                                anchors.centerIn: parent
                                anchors.verticalCenterOffset: -15
                            }
                            
                            Text {
                                text: "Flip Left"
                                font.bold: true
                                color: "white"
                                font.pixelSize: 11
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 5
                            }
                            
                            MouseArea {
                                anchors.fill: parent
                                onEntered: parent.color = "white"
                                onExited: parent.color = "#242c4d"
                                onClicked: backend.doDroneTAction("flip_left")
                            }
                        }
                        
                        // Flip Right Button
                        Rectangle {
                            width: (parent.width - parent.spacing * 4) / 5
                            height: parent.height
                            color: "#242c4d"
                            border.color: "black"
                            
                            Text {
                                text: "⟳"
                                font.bold: true
                                color: "#76ff03"
                                font.pixelSize: 150
                                anchors.centerIn: parent
                                anchors.verticalCenterOffset: -15
                            }
                            
                            Text {
                                text: "Flip Right"
                                font.bold: true
                                color: "white"
                                font.pixelSize: 11
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 5
                            }
                            
                            MouseArea {
                                anchors.fill: parent
                                onEntered: parent.color = "white"
                                onExited: parent.color = "#242c4d"
                                onClicked: backend.doDroneTAction("flip_right")
                            }
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
                                onEntered: parent.color = "white"
                                onExited: parent.color = "#242c4d"
                                onClicked: backend.doDroneTAction("connect")
                            }
                        }

                        // Down Button
                        Rectangle {
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
                                onEntered: parent.color = "white"
                                onExited: parent.color = "#242c4d"
                                onClicked: backend.doDroneTAction("down")
                            }
                        }

                        // Takeoff Button
                        Rectangle {
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
                                onEntered: parent.color = "white"
                                onExited: parent.color = "#242c4d"
                                onClicked: backend.takeoff()
                            }
                        }

                        // Land Button
                        Rectangle {
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
                                onEntered: parent.color = "white"
                                onExited: parent.color = "#242c4d"
                                onClicked: backend.doDroneTAction("land")
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