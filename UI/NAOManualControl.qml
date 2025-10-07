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

        // --- UI state / movement params ---
        property int  rotationStep: 90
        property real moveDistance: 50
        property real verticalStep: 50
        property int  verticalState: 0
        property int  maxVerticalState: 3
        property bool animationInProgress: false
        property int  modelRotationY: 0

        RowLayout {
            anchors.fill: parent
            spacing: 0

            // ================= LEFT PANEL =================
            Rectangle {
                id: leftPanel
                Layout.preferredWidth: 510
                Layout.fillHeight: true
                color: "#718399"

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 40

                    Label {
                        text: "Nao Robot Control Panel"
                        horizontalAlignment: Text.AlignHCenter
                        font.pixelSize: 20
                        font.bold: true
                        color: "white"
                        Layout.alignment: Qt.AlignHCenter
                    }

                    GridLayout {
                        id: buttonGrid
                        columns: 2
                        columnSpacing: 20
                        rowSpacing: 20
                        Layout.alignment: Qt.AlignHCenter
                        Layout.fillWidth: true
                        Layout.preferredHeight: 400

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
                                Layout.preferredWidth: 150
                                Layout.preferredHeight: 150
                                color: "#2c3e50"
                                radius: 10

                                Column {
                                    anchors.centerIn: parent
                                    spacing: 8

                                    Image {
                                        source: imgBase + modelData.icon
                                        width: 80; height: 80
                                        fillMode: Image.PreserveAspectFit
                                        onStatusChanged: if (status === Image.Error) console.log("❌ Image not found:", source)
                                    }

                                    Text {
                                        text: modelData.label
                                        color: "white"
                                        font.pixelSize: 16
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

                    GroupBox {
                        title: "Console Log"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.preferredHeight: 200

                        ScrollView {
                            anchors.fill: parent
                            ScrollBar.vertical.policy: ScrollBar.AlwaysOn

                            TextArea {
                                id: consoleLog1
                                readOnly: true
                                wrapMode: Text.Wrap
                                color: "white"
                                font.pixelSize: 10
                                background: Rectangle { color: "black" }
                            }
                        }
                    }
                }
            }

            // ================= RIGHT PANEL (3D Viewer) =================
            Rectangle {
                id: rightPanel
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: "#2f2f2f"
                radius: 6

                // 3D scene
                View3D {
                    id: v3d
                    anchors.fill: parent

                    environment: SceneEnvironment {
                        backgroundMode: SceneEnvironment.Color
                        clearColor: "#2e2e2e"
                        aoStrength: 0.2
                        probeExposure: -0.25
                    }

                    PerspectiveCamera {
                        id: camera
                        position: Qt.vector3d(0, 300, 900)
                        eulerRotation.x: -20
                        clipNear: 1
                        clipFar: 10000
                    }

                    // Lights
                    DirectionalLight {
                        eulerRotation.x: -35
                        eulerRotation.y: 35
                        brightness: 1.2   // was 2.5
                        castsShadow: true
                    }
                    DirectionalLight {
                        eulerRotation.x: 25
                        eulerRotation.y: -45
                        brightness: 0.8   // was 1.5
                    }
                    PointLight {
                        position: Qt.vector3d(0, 250, 350)
                        brightness: 500    // was 1200
                    }

                    // >>> NAO model from ../Nao.mesh/Nao.qml
                    NaoMesh.Nao {
                        id: naoModel
                        scale: Qt.vector3d(100, 100, 100)
                        position: Qt.vector3d(0, -100, 0)
                        eulerRotation: Qt.vector3d(0, 0, 0)
                    }

                    // Ground plane
                    Model {
                        source: "#Cube"
                        scale: Qt.vector3d(2000, 10, 2000)
                        position: Qt.vector3d(0, -200, 0)
                    }
                }

                // Connect button overlay (outside View3D)
                Rectangle {
                    anchors.right: parent.right
                    anchors.bottom: parent.bottom
                    width: parent.width * 0.2
                    height: parent.height * 0.2
                    color: "#242c4d" // Dark blue background

                    Image {
                        source: imgBase + "connect.png"
                        anchors.fill: parent
                        fillMode: Image.PreserveAspectFit
                        onStatusChanged: if (status === Image.Error) console.log("❌ Image not found:", source)
                    }

                    Text {
                        text: "Connect"
                        anchors.horizontalCenter: parent.horizontalCenter
                        anchors.bottom: parent.bottom
                        anchors.bottomMargin: 10
                        font.pixelSize: 18
                        font.bold: true
                        color: "white"
                    }

                    MouseArea {
                        anchors.fill: parent
                        onClicked: {
                            rootPanel.appendLog("Connect button clicked!")
                            if (typeof backend !== "undefined" && backend.connectNao) backend.connectNao()
                        }
                    }
      
                }
            }
        }

        // ================= Animations =================
        PropertyAnimation {
            id: moveAnim
            target: naoModel
            property: "position"
            duration: 1000
            onStopped: rootPanel.animationInProgress = false
        }

        PropertyAnimation {
            id: rotateAnim
            target: naoModel
            property: "eulerRotation"
            duration: 800
            onStopped: rootPanel.animationInProgress = false
        }

        // ================= Movement functions =================
        function moveForward() {
            if (animationInProgress) { rootPanel.appendLog("Cannot move Forward - action already in progress!"); return }
            var angleRad = modelRotationY * Math.PI / 180.0
            var dirX = Math.sin(angleRad)
            var dirZ = Math.cos(angleRad)
            var start = naoModel.position
            var end = Qt.vector3d(start.x + moveDistance * dirX, start.y, start.z + moveDistance * dirZ)
            animationInProgress = true
            moveAnim.from = start
            moveAnim.to   = end
            moveAnim.start()
        }

        function moveBackward() {
            if (animationInProgress) { rootPanel.appendLog("Cannot move Backward - action already in progress!"); return }
            var angleRad = modelRotationY * Math.PI / 180.0
            var dirX = Math.sin(angleRad)
            var dirZ = Math.cos(angleRad)
            var start = naoModel.position
            var end = Qt.vector3d(start.x - moveDistance * dirX, start.y, start.z - moveDistance * dirZ)
            animationInProgress = true
            moveAnim.from = start
            moveAnim.to   = end
            moveAnim.start()
        }

        function turnLeft() {
            if (animationInProgress) { rootPanel.appendLog("Cannot turn Left - action already in progress!"); return }
            modelRotationY = (modelRotationY - rotationStep) % 360
            var start = naoModel.eulerRotation
            var end   = Qt.vector3d(start.x, start.y + rotationStep, start.z)
            animationInProgress = true
            rotateAnim.from = start
            rotateAnim.to   = end
            rotateAnim.start()
        }

        function turnRight() {
            if (animationInProgress) { rootPanel.appendLog("Cannot turn Right - action already in progress!"); return }
            modelRotationY = (modelRotationY + rotationStep) % 360
            var start = naoModel.eulerRotation
            var end   = Qt.vector3d(start.x, start.y - rotationStep, start.z)
            animationInProgress = true
            rotateAnim.from = start
            rotateAnim.to   = end
            rotateAnim.start()
        }

        function moveUp() {
            if (animationInProgress || verticalState >= maxVerticalState) {
                rootPanel.appendLog("Cannot Takeoff - already at max height!")
                return
            }
            var start = naoModel.position
            var end   = Qt.vector3d(start.x, start.y + verticalStep, start.z)
            animationInProgress = true
            moveAnim.from = start
            moveAnim.to   = end
            moveAnim.start()
            verticalState++
        }

        function moveDown() {
            if (animationInProgress || verticalState <= 0) {
                rootPanel.appendLog("Cannot Land - already at ground!")
                return
            }
            var start = naoModel.position
            var end   = Qt.vector3d(start.x, start.y - verticalStep, start.z)
            animationInProgress = true
            moveAnim.from = start
            moveAnim.to   = end
            moveAnim.start()
            verticalState--
        }

        // ================= Console Logger =================
        function appendLog(msg) {
            var timestamp = new Date().toLocaleString()
            consoleLog1.append(msg + " at " + timestamp)
        }
    }

    // Optional keyboard shortcuts
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
