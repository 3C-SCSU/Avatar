/*
    Avatar - NAO Manual Control Panel
    Author: Youssef Elkhouly
    Date: October 2025

    Description:
        This QML file provides the NAO Robot Control Panel interface with a 3D viewer
        and connection configuration. Users can control the NAO robot's movements and
        configure the connection IP and Port dynamically.

    Features:
        - 3D NAO model visualization with lighting
        - Movement control buttons (Forward, Backward, Left, Right, Takeoff, Land)
        - Dynamic IP and Port configuration
        - Console logging for connection status
        - Smooth animations for robot movements
*/

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick3D

// Import NAO component from Nao.mesh/Nao.qml
import "../Nao.mesh" as NaoMesh

Item {
    id: root
    anchors.fill: parent
    focus: true
    property url imgBase: Qt.resolvedUrl("../GUI_Pics/")

    Rectangle {
        id: rootPanel
        anchors.fill: parent
        color: "transparent"

        // --- State / movement parameters ---
        property int  rotationStep: 90
        property real moveDistance: 50
        property real verticalStep: 50
        property int  verticalState: 0
        property int  maxVerticalState: 3
        property bool animationInProgress: false
        property int  modelRotationY: 0

        RowLayout {
            anchors.fill: parent
            spacing: parent.width * 0.01

            // ================= LEFT PANEL =================
            Rectangle {
                id: leftPanel
                Layout.fillHeight: true
                Layout.preferredWidth: parent.width * 0.38
                color: "#718399"
                radius: 8

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: parent.width * 0.015
                    spacing: parent.height * 0.025

                    Label {
                        text: "Nao Robot Control Panel"
                        horizontalAlignment: Text.AlignHCenter
                        font.pixelSize: parent.height * 0.05
                        font.bold: true
                        color: "white"
                        Layout.alignment: Qt.AlignHCenter
                    }

                    // ---------- CONTROL GRID ----------
                    GridLayout {
                        id: buttonGrid
                        columns: 2
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        columnSpacing: parent.width * 0.03
                        rowSpacing: parent.height * 0.03

                        Repeater {
                            model: [
                                { label: "Backward", icon: "back.png",    fn: "moveBackward" },
                                { label: "Forward",  icon: "forward.png", fn: "moveForward" },
                                { label: "Right",    icon: "right.png",   fn: "turnRight" },
                                { label: "Left",     icon: "left.png",    fn: "turnLeft" },
                                { label: "Takeoff",  icon: "takeoff.png", fn: "moveUp" },
                                { label: "Land",     icon: "land.png",    fn: "moveDown" }
                            ]

                            delegate: Rectangle {
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                color: "#2c3e50"
                                radius: 10

                                Column {
                                    anchors.centerIn: parent
                                    spacing: parent.height * 0.05

                                    Image {
                                        source: imgBase + modelData.icon
                                        width: parent.width * 0.7
                                        height: parent.width * 0.7
                                        fillMode: Image.PreserveAspectFit
                                    }

                                    Text {
                                        text: modelData.label
                                        color: "white"
                                        font.pixelSize: leftPanel.height * 0.03
                                        horizontalAlignment: Text.AlignHCenter
                                    }
                                }

                                MouseArea {
                                    anchors.fill: parent
                                    onClicked: {
                                        rootPanel.appendLog(modelData.label + " Button Clicked!")
                                        if (typeof rootPanel[modelData.fn] === "function") {
                                            rootPanel[modelData.fn]()
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // ---------- RESPONSIVE CONSOLE LOG ----------
                    GroupBox {
                        title: "Console Log"
                        Layout.fillWidth: true
                        Layout.preferredHeight: parent.height * 0.25
                        Layout.minimumHeight: 150
                        Layout.maximumHeight: parent.height * 0.35

                        ScrollView {
                            anchors.fill: parent
                            clip: true
                            ScrollBar.vertical.policy: ScrollBar.AsNeeded

                            Rectangle {
                                anchors.fill: parent
                                color: "#0D1117"
                                radius: 6
                                border.color: "#2E2E2E"

                                TextArea {
                                    id: consoleLog1
                                    anchors.fill: parent
                                    readOnly: true
                                    wrapMode: Text.Wrap
                                    color: "white"
                                    font.family: "Consolas"
                                    font.pixelSize: leftPanel.height * 0.025
                                    background: null
                                }
                            }
                        }
                    }
                }
            }

            // ================= RIGHT PANEL (3D VIEW) =================
            Rectangle {
                id: rightPanel
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: "#2f2f2f"
                radius: 6

                View3D {
                    anchors.fill: parent
                    environment: SceneEnvironment {
                        backgroundMode: SceneEnvironment.Color
                        clearColor: "#2e2e2e"
                    }

                    PerspectiveCamera {
                        id: camera
                        position: Qt.vector3d(0, 250, 800)
                        eulerRotation.x: -15
                        clipFar: 5000
                    }

                    DirectionalLight {
                        eulerRotation.x: -45
                        eulerRotation.y: 45
                        brightness: 1.2
                    }
                    DirectionalLight {
                        eulerRotation.x: 30
                        eulerRotation.y: -60
                        brightness: 1.0
                    }
                    PointLight {
                        position: Qt.vector3d(0, 400, 400)
                        brightness: 800
                    }

                    NaoMesh.Nao {
                        id: naoModel
                        scale: Qt.vector3d(100, 100, 100)
                        position: Qt.vector3d(0, -100, 0)
                    }
                }

                // ---------- CONNECT BUTTON ----------
                Rectangle {
                    anchors.right: parent.right
                    anchors.bottom: parent.bottom
                    width: parent.width * 0.15
                    height: parent.height * 0.15
                    color: "#242c4d"
                    radius: 6

                    Image {
                        source: imgBase + "connect.png"
                        anchors.fill: parent
                        fillMode: Image.PreserveAspectFit
                    }

                    Text {
                        text: "Connect"
                        anchors.centerIn: parent
                        font.pixelSize: parent.height * 0.3
                        font.bold: true
                        color: "white"
                    }
                // Connection Panel with IP and Port input fields
                Rectangle {
                    anchors.right: parent.right
                    anchors.bottom: parent.bottom
                    width: parent.width * 0.25
                    height: parent.height * 0.25
                    color: "#242c4d"
                    radius: 8
                    border.color: "#3a4a6d"
                    border.width: 2

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 12
                        spacing: 10

                        Text {
                            text: "NAO Connection"
                            font.pixelSize: 14
                            font.bold: true
                            color: "white"
                            Layout.alignment: Qt.AlignHCenter
                        }

                        // IP Address Input
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8

                            Text {
                                text: "IP:"
                                color: "white"
                                font.pixelSize: 12
                                Layout.preferredWidth: 30
                            }

                            TextField {
                                id: ipInput
                                Layout.fillWidth: true
                                placeholderText: "192.168.23.53"
                                text: "192.168.23.53"
                                font.pixelSize: 11
                                background: Rectangle {
                                    color: "#1a1f3a"
                                    border.color: "#4a5a7d"
                                    border.width: 1
                                    radius: 4
                                }
                                color: "white"
                            }
                        }

                        // Port Input
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8

                            Text {
                                text: "Port:"
                                color: "white"
                                font.pixelSize: 12
                                Layout.preferredWidth: 30
                            }

                            TextField {
                                id: portInput
                                Layout.fillWidth: true
                                placeholderText: "9559"
                                text: "9559"
                                font.pixelSize: 11
                                background: Rectangle {
                                    color: "#1a1f3a"
                                    border.color: "#4a5a7d"
                                    border.width: 1
                                    radius: 4
                                }
                                color: "white"
                            }
                        }

                        // Connect Button
                        Button {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 36
                            text: "Connect"
                            font.bold: true
                            font.pixelSize: 12

                            background: Rectangle {
                                color: parent.pressed ? "#1a4d2e" : "#2d7a4a"
                                radius: 4
                                border.color: "#4a9d6f"
                                border.width: 1
                            }

                            contentItem: Text {
                                text: parent.text
                                color: "white"
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                                font: parent.font
                            }

                            onClicked: {
                                var ip = ipInput.text.trim()
                                var port = portInput.text.trim()
                                rootPanel.appendLog("Connecting to NAO at " + ip + ":" + port)
                                if (typeof backend !== "undefined" && backend.connectNao) {
                                    backend.connectNao(ip, port)
                                }
                            }
                        }
                    }
                }
            }
        }

        // ---------- ANIMATIONS ----------
        PropertyAnimation {
            id: moveAnim
            target: naoModel
            property: "position"
            duration: 800
            onStopped: rootPanel.animationInProgress = false
        }
        PropertyAnimation {
            id: rotateAnim
            target: naoModel
            property: "eulerRotation"
            duration: 800
            onStopped: rootPanel.animationInProgress = false
        }

        // ---------- MOVEMENT FUNCTIONS ----------
        function moveForward()  { if (!animationInProgress) animateMove(1) }
        function moveBackward() { if (!animationInProgress) animateMove(-1) }
        function turnLeft()     { if (!animationInProgress) animateRotate(-rotationStep) }
        function turnRight()    { if (!animationInProgress) animateRotate(rotationStep) }
        function moveUp()       { if (!animationInProgress && verticalState < maxVerticalState) animateVertical(verticalStep) }
        function moveDown()     { if (!animationInProgress && verticalState > 0) animateVertical(-verticalStep) }

        function animateMove(direction) {
            var angleRad = modelRotationY * Math.PI / 180.0
            var start = naoModel.position
            var end = Qt.vector3d(start.x + direction * moveDistance * Math.sin(angleRad),
                                  start.y,
                                  start.z + direction * moveDistance * Math.cos(angleRad))
            moveAnim.from = start
            moveAnim.to = end
            animationInProgress = true
            moveAnim.start()
        }

        function animateRotate(deg) {
            modelRotationY += deg
            var start = naoModel.eulerRotation
            var end = Qt.vector3d(start.x, start.y + deg, start.z)
            rotateAnim.from = start
            rotateAnim.to = end
            animationInProgress = true
            rotateAnim.start()
        }

        function animateVertical(step) {
            var start = naoModel.position
            var end = Qt.vector3d(start.x, start.y + step, start.z)
            moveAnim.from = start
            moveAnim.to = end
            animationInProgress = true
            moveAnim.start()
            verticalState += step > 0 ? 1 : -1
        }

        function appendLog(msg) {
            var timestamp = new Date().toLocaleTimeString()
            consoleLog1.append(`[${timestamp}] ${msg}`)
        }
    }

    // ---------- KEYBOARD SHORTCUTS ----------
    Keys.onPressed: (e) => {
        switch (e.key) {
        case Qt.Key_W: case Qt.Key_Up:     rootPanel.moveForward();  break
        case Qt.Key_S: case Qt.Key_Down:   rootPanel.moveBackward(); break
        case Qt.Key_A: case Qt.Key_Left:   rootPanel.turnLeft();     break
        case Qt.Key_D: case Qt.Key_Right:  rootPanel.turnRight();    break
        case Qt.Key_T:                     rootPanel.moveUp();       break
        case Qt.Key_L:                     rootPanel.moveDown();     break
        }
    }
}