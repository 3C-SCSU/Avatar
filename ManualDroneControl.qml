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

    readonly property color buttonBg: "#242c4d"
    readonly property color buttonBorder: "#3a4a62"
    readonly property int  buttonRadius: 10

    RowLayout {
        anchors.fill: parent
        anchors.margins: 10

        // ====================================
        // ==== Left side - Drone Controls ====
        // ====================================
        Rectangle {
            id: controlsPanel
            Layout.fillWidth: false
            Layout.preferredWidth: parent.width * 0.7
            Layout.fillHeight: true
            color: "transparent"

            // control row container
            ColumnLayout {
                anchors.fill: parent
                spacing: 5

                // ************************************
                // >> Top Row - Home, Up, Flight Log <<
                // ************************************
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: parent.height * 0.19
                    color: "transparent"

                    RowLayout {
                        anchors.fill: parent

                        // Home Button
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/home.png"
                                }

                                Text { visible: false }
                            }

                            Text {
                                text: "Home"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: backend.doDroneTAction("go_home")
                            }
                        }

                        // empty space - required by design
                        Rectangle {
                            Layout.preferredWidth: 0.095 * parent.width
                            Layout.fillHeight: true
                            color: "transparent"
                        }

                        // Up Button
                        Rectangle {
                            Layout.preferredWidth: 0.405 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/up.png"
                                }

                                Text { visible: false }
                            }

                            Text {
                                text: "Up"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: backend.doDroneTAction("up")
                            }
                        }

                        // empty space - required by design
                        Rectangle {
                            Layout.preferredWidth: 0.09 * parent.width
                            Layout.fillHeight: true
                            color: "transparent"
                        }

                        // Flight Log
                        Rectangle {
                            id: flightLogContainer
                            Layout.preferredWidth: 0.19 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: "#ffffff"
                            border.color: "#2E4053"
                            border.width: 1

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 3
                                anchors.topMargin: 5
                                spacing: 4

                                Text {
                                    text: "FLIGHT LOG"
                                    font.weight: 700
                                    color: "black"
                                    font.pixelSize: 12
                                    Layout.alignment: Qt.AlignHCenter
                                    font.letterSpacing: 1.1
                                }

                                ScrollView {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    clip: false

                                    TextArea {
                                        id: flightLog
                                        readOnly: true
                                        wrapMode: TextArea.Wrap
                                        font.pixelSize: Math.max(10, controlsPanel.height * 0.015)
                                        color: "black"

                                        background: Rectangle { color: "white" }

                                        Component.onCompleted: {
                                            backend.flightLogUpdated.connect(function (logList) {
                                                flightLog.text = logList.join("\n")
                                                flightLog.moveCursorSelection(TextArea.End)  // auto-scroll to bottom
                                            })
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // *******************************************
                // >>        2nd Row - Forward Button       <<
                // *******************************************
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: parent.height * 0.19
                    color: "transparent"

                    RowLayout {
                        anchors.fill: parent

                        // empty space - required by design
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            color: "transparent"
                        }

                        // Flip Forward Button
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {visible: false}

                                Text {
                                    visible: true
                                    text: "↻"
                                    font.bold: true
                                    color: "#76ff03"
                                    font.pixelSize: parent.height * 0.8
                                    anchors.centerIn: parent
                                }
                            }

                            Text {
                                text: "Flip Forward"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: backend.doDroneTAction("flip_forward")
                            }
                        }

                        // forward
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/forward.png"
                                }

                                Text { visible: false }

                            }

                            Text {
                                text: "Forward"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: backend.doDroneTAction("forward")
                            }
                        }

                        // flip right
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: false
                                }

                                Text {
                                    visible: true
                                    text: "⟳"
                                    font.bold: true
                                    color: "#76ff03"
                                    anchors.centerIn: parent
                                    font.pixelSize: parent.height * 0.8
                                }
                            }

                            Text {
                                text: "Flip Right"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: backend.doDroneTAction("flip_right")
                            }
                        }

                        // empty space - required by design
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            color: "transparent"
                        }
                    }
                }

                // **************************************************************************************
                // >> 3rd Row - Directional Buttons (Turn Left, Left, Stream, Right, Turn Right)       <<
                // **************************************************************************************
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: parent.height * 0.19
                    color: "transparent"

                    RowLayout {
                        anchors.fill: parent

                        // Turn Left Button
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/turnLeft.png"
                                }

                                Text { visible: false }
                            }

                            Text {
                                text: "Turn Left"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: backend.doDroneTAction("turn_left")
                            }
                        }

                        // Left Button
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/left.png"
                                }

                                Text { visible: false }
                            }

                            Text {
                                text: "Left"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: backend.doDroneTAction("left")
                            }
                        }

                        // Stream Button
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/stream.png"
                                }

                                Text { visible: false }
                            }

                            Text {
                                text: "Stream"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: backend.doDroneTAction("stream")
                            }
                        }

                        // Right Button
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/right.png"
                                }

                                Text { visible: false }
                            }

                            Text {
                                text: "Right"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: backend.doDroneTAction("right")
                            }
                        }

                        // Turn Right Button
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/turnRight.png"
                                }

                                Text { visible: false }
                            }

                            Text {
                                text: "Turn Right"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: backend.doDroneTAction("turn_right")
                            }
                        }
                    }
                }

                // *******************************************************
                // >> 4th Row - Back Button Row with Flip Buttons       << DONE
                // *******************************************************
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: parent.height * 0.19
                    color: "transparent"

                    RowLayout {
                        anchors.fill: parent

                        // empty space - required by design
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            color: "transparent"
                        }

                        // Flip Back Button
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {visible: false}

                                Text {
                                    visible: true
                                    text: "↺"
                                    font.bold: true
                                    color: "#76ff03"
                                    font.pixelSize: parent.height * 0.8
                                    anchors.centerIn: parent
                                }
                            }

                            Text {
                                text: "Flip Back"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: backend.doDroneTAction("flip_back")
                            }
                        }

                        // Back Button (Center)
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/back.png"
                                }

                                Text { visible: false }

                            }

                            Text {
                                text: "Back"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: backend.doDroneTAction("backward")
                            }
                        }

                        // Flip Left Button
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {visible: false}

                                Text {
                                    visible: true
                                    text: "⟲"
                                    font.bold: true
                                    color: "#76ff03"
                                    font.pixelSize: parent.height * 0.8
                                    anchors.centerIn: parent
                                }
                            }

                            Text {
                                text: "Flip Left"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: backend.doDroneTAction("flip_left")
                            }
                        }

                        // empty space - required by design
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            color: "transparent"
                        }
                    }
                }
                // *******************************************************
                // >> 5th Row - Connect, Down, Takeoff, Land Buttons    <<
                // *******************************************************
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: parent.height * 0.19
                    color: "transparent"

                    RowLayout {
                        anchors.fill: parent

                        // Connect Button
                        Rectangle {
                            Layout.preferredWidth: 0.195 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/connect.png"
                                }

                                Text { visible: false }
                            }

                            Text {
                                text: "Connect"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: backend.doDroneTAction("connect")
                            }
                        }

                        // empty space - required by design
                        Rectangle {
                            Layout.preferredWidth: 0.09 * parent.width
                            Layout.fillHeight: true
                            color: "transparent"
                        }

                        // Down Button
                        Rectangle {
                            Layout.preferredWidth: 0.405 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/down.png"
                                }

                                Text { visible: false }
                            }

                            Text {
                                text: "Down"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: backend.doDroneTAction("down")
                            }
                        }

                        // empty space - required by design
                        Rectangle {
                            Layout.preferredWidth: 0.085 * parent.width
                            Layout.fillHeight: true
                            color: "transparent"
                        }

                        // Takeoff Button
                        Rectangle {
                            Layout.preferredWidth: 0.0975 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/takeoff.png"
                                }

                                Text { visible: false }
                            }

                            Text {
                                text: "Takeoff"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                onEntered: parent.color = "white"
                                onExited: parent.color = "#242c4d"
                                onClicked: backend.takeoff()
                                cursorShape: Qt.PointingHandCursor
                            }
                        }

                        // Land Button
                        Rectangle {
                            Layout.preferredWidth: 0.0975 * parent.width
                            Layout.fillHeight: true
                            radius: buttonRadius
                            color: buttonBg
                            border.color: buttonBorder
                            border.width: 1

                            Item {
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.top: parent.top
                                anchors.topMargin: parent.height * 0.10
                                width: parent.width * 0.6
                                height: parent.height * 0.55

                                Image {
                                    visible: true
                                    anchors.fill: parent
                                    fillMode: Image.PreserveAspectFit
                                    source: "GUI_Pics/land.png"
                                }

                                Text { visible: false }
                            }

                            Text {
                                text: "Land"
                                font.bold: true
                                color: "white"
                                font.pixelSize: parent.height * 0.1
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.bottom: parent.bottom
                                anchors.bottomMargin: 8
                            }

                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: backend.doDroneTAction("land")

                                hoverEnabled: true
                                onEntered:  root.state = "hovered"
                                onExited:   root.state = "normal"
                                onPressed:  scaleTr.xScale = scaleTr.yScale = 0.97
                                onReleased: root.state = "hovered"
                            }
                        }
                    }
                }
            }
        }

        // ====================================
        // ==== Right side of Camera View =====
        // ====================================

        CameraView {
            id: cameraPanel
            Layout.preferredWidth: parent.width * 0.3
            Layout.minimumWidth: 300
            Layout.fillHeight: true
            // width: parent.width * 0.3
            // height: parent.height
            cameraController: cameraController
        }
    }
}
