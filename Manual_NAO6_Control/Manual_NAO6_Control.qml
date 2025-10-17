import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtQuick3D 6.7
import "../Nao.mesh"

Rectangle {
    id: rootPanel
    property int rotationStep: 90
    property real moveDistance: 50
    property real verticalStep: 50
    property int verticalState: 0
    property int maxVerticalState: 3
    property bool animationInProgress: false
    property int modelRotationY: 0

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
                            {label: "Backward", icon: "../GUI_Pics/back.png", fn: "moveBackward"},
                            {label: "Forward",  icon: "../GUI_Pics/forward.png", fn: "moveForward"},
                            {label: "Right",    icon: "../GUI_Pics/right.png", fn: "turnRight"},
                            {label: "Left",     icon: "../GUI_Pics/left.png", fn: "turnLeft"},
                            {label: "Takeoff",  icon: "../GUI_Pics/takeoff.png", fn: "moveUp"},
                            {label: "Land",     icon: "../GUI_Pics/land.png", fn: "moveDown"}
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
                                    source: modelData.icon
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

            View3D {
                anchors.fill: parent

                environment: SceneEnvironment {
                    clearColor: '#2e2e2e'
                    backgroundMode: SceneEnvironment.Color
                }

                PerspectiveCamera {
                    id: camera
                    position: Qt.vector3d(0, 250, 800)
                    eulerRotation.x: -10
                    clipFar: 5000
                }

                DirectionalLight {
                    eulerRotation.x: -45
                    eulerRotation.y: 45
                    brightness: 1.5
                    castsShadow: true
                }

                DirectionalLight {
                    eulerRotation.x: 30
                    eulerRotation.y: -60
                    brightness: 1.2
                }

                PointLight {
                    position: Qt.vector3d(0, 400, 400)
                    brightness: 800
                }

                // Load NAO model
                Nao {
                    id: naoModel
                    scale: Qt.vector3d(100, 100, 100)
                    position: Qt.vector3d(0, -100, 0)
                }

                // Connect Button with Image in bottom-right corner
                Rectangle {
                    anchors.right: parent.right
                    anchors.bottom: parent.bottom
                    width: parent.width * 0.2
                    height: parent.height * 0.2
                    color: "#242c4d"

                    Image {
                        id: connectImage
                        source: "GUI_Pics/connect.png"
                        anchors.fill: parent
                        fillMode: Image.PreserveAspectFit
                    }

                    Text {
                        text: "Connect"
                        anchors.centerIn: parent
                        font.pixelSize: 20
                        font.bold: true
                        color: "white"
                    }

                    MouseArea {
                        anchors.fill: parent
                        onClicked: {
                            rootPanel.appendLog("Connect button clicked!")
                            manualNaoController.connectNao()    // was backend.connectNao()
                        }
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
        var angleRad = modelRotationY * Math.PI/180.0
        var dirX = Math.sin(angleRad)
        var dirZ = Math.cos(angleRad)
        var start = naoModel.position
        var end = Qt.vector3d(start.x + moveDistance*dirX, start.y, start.z + moveDistance*dirZ)
        animationInProgress = true
        moveAnim.from = start
        moveAnim.to = end
        moveAnim.start()
    }

    function moveBackward() {
        if (animationInProgress) { rootPanel.appendLog("Cannot move Backward - action already in progress!"); return }
        var angleRad = modelRotationY * Math.PI/180.0
        var dirX = Math.sin(angleRad)
        var dirZ = Math.cos(angleRad)
        var start = naoModel.position
        var end = Qt.vector3d(start.x - moveDistance*dirX, start.y, start.z - moveDistance*dirZ)
        animationInProgress = true
        moveAnim.from = start
        moveAnim.to = end
        moveAnim.start()
    }

    function turnLeft() {
        if (animationInProgress) { rootPanel.appendLog("Cannot turn Left - action already in progress!"); return }
        modelRotationY = (modelRotationY - rotationStep) % 360
        var start = naoModel.eulerRotation
        var end = Qt.vector3d(start.x, start.y + rotationStep, start.z)
        animationInProgress = true
        rotateAnim.from = start
        rotateAnim.to = end
        rotateAnim.start()
    }

    function turnRight() {
        if (animationInProgress) { rootPanel.appendLog("Cannot turn Right - action already in progress!"); return }
        modelRotationY = (modelRotationY + rotationStep) % 360
        var start = naoModel.eulerRotation
        var end = Qt.vector3d(start.x, start.y - rotationStep, start.z)
        animationInProgress = true
        rotateAnim.from = start
        rotateAnim.to = end
        rotateAnim.start()
    }

    function moveUp() {
        if (animationInProgress || verticalState >= maxVerticalState) { rootPanel.appendLog("Cannot Takeoff - already at max height!"); return }
        var start = naoModel.position
        var end = Qt.vector3d(start.x, start.y + verticalStep, start.z)
        animationInProgress = true
        moveAnim.from = start
        moveAnim.to = end
        moveAnim.start()
        verticalState++
    }

    function moveDown() {
        if (animationInProgress || verticalState <= 0) { rootPanel.appendLog("Cannot Land - already at ground!"); return }
        var start = naoModel.position
        var end = Qt.vector3d(start.x, start.y - verticalStep, start.z)
        animationInProgress = true
        moveAnim.from = start
        moveAnim.to = end
        moveAnim.start()
        verticalState--
    }

    // ================= Console Logger =================
    function appendLog(msg) {
        var timestamp = new Date().toLocaleString()
        consoleLog1.append(msg + " at " + timestamp)
    }
}
//