import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtQuick3D 6.7
import QtQuick.Dialogs
import Qt.labs.platform
import "GUI5_ManualDroneControl/cameraview"

// Manual Drone Control view
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
                                backend.getDroneAction("home");
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
                                backend.getDroneAction("up");
                            }
                        }
                    }

                    // Flight Log
                    Rectangle {
                        width: parent.width * 0.2
                        height: parent.height
                        anchors.right: parent.right
                        color: "white"
                        border.color: "#2E4053"

                        Text {
                            text: "Flight Log"
                            font.bold: true
                            font.pixelSize: Math.max(12, parent.width * 0.05)
                            color: "black"
                            anchors.horizontalCenter: parent.horizontalCenter
                            anchors.top: parent.top
                            anchors.topMargin: 10
                        }

                        TextArea {
                            anchors.fill: parent
                            anchors.topMargin: 30
                            font.pixelSize: Math.max(10, parent.width * 0.03)
                            color: "black"
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
                            onEntered: parent.color = "white"
                            onExited: parent.color = "#242c4d"
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
                            onClicked: backend.getDroneAction("turn_left")
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
                            onClicked: backend.getDroneAction("left")
                        }
                    }

                    // Stream Button
                    Rectangle {
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
                            onEntered: parent.color = "white"
                            onExited: parent.color = "#242c4d"
                            onClicked: backend.getDroneAction("stream")
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
                            onClicked: backend.getDroneAction("right")
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
                            onEntered: parent.color = "white"
                            onExited: parent.color = "#242c4d"
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
                                onClicked: backend.getDroneAction("connect")
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
                                onClicked: backend.getDroneAction("down")
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
                                onClicked: backend.getDroneAction("takeoff")
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
