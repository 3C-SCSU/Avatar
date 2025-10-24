import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15

Rectangle {
    id: root
    anchors.fill: parent
    // color: "#5f6f7f"        // outer panel background (adjust)
    radius: 8

    property var cameraController: null
    property bool recording: false
    property string latestFrameDebug: ""

    Connections {
        target: cameraController
        onVideoFrame: function(data) {
            // controller should emit a data-uri string (data:image/...) or debug payload
            if (typeof data === "string" && data.indexOf("data:image") === 0) {
                feedImage.source = data
                latestFrameDebug = ""
            } else {
                latestFrameDebug = data ? data.toString() : ""
            }
        }
        onLogMessage: function(msg) {
            console.log("droneCameraController:", msg)
        }
    }

    // Header
    Rectangle {
        id: header
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        height: 56
        color: "#385166"
        radius: 6
        radiusTopLeft: 8
        radiusTopRight: 8

        Text {
            anchors.centerIn: parent
            text: "CAMERA VIEW"
            color: "white"
            font.bold: true
            font.pixelSize: 18
        }
    }

    // Main feed area
    Rectangle {
        id: feedWrapper
        anchors {
            top: header.bottom
            left: parent.left; right: parent.right
            bottom: controls.top
            margins: 12
        }
        color: "#11161a"
        radius: 6
        border.color: "#1f2933"
        border.width: 2
        clip: true

        // top-right red recording indicator
        Rectangle {
            id: recDot
            anchors.top: parent.top; anchors.right: parent.right
            anchors.margins: 10
            width: 18; height: 18
            radius: width/2
            color: recording ? "#e74c3c" : "transparent"
            border.color: recording ? "#b03030" : "transparent"
            border.width: 1
            z: 10
        }

        // Image feed (fills wrapper)
        Image {
            id: feedImage
            anchors.fill: parent
            fillMode: Image.PreserveAspectFit
            source: ""
            asynchronous: true
            cache: false
        }

        // centered placeholder when no feed
        Column {
            anchors.centerIn: parent
            spacing: 8
            visible: feedImage.source === "" || feedImage.status === Image.Error

            Rectangle {
                width: 84; height: 84
                radius: 42
                color: "#3e4f61"
                border.color: "#2b3a49"
                border.width: 2
                Image {
                    anchors.centerIn: parent
                    source: "../GUI_Pics/camera_icon.png"
                    width: 48; height: 48
                    fillMode: Image.PreserveAspectFit
                }
            }

            Text {
                text: "Camera Ready"
                color: "#b6c0c8"
                font.pixelSize: 14
                horizontalAlignment: Text.AlignHCenter
            }
        }

        // debug small text for non-image payloads
        Text {
            id: debugText
            anchors.left: parent.left; anchors.bottom: parent.bottom
            anchors.margins: 8
            color: "#9aa7b3"
            font.pixelSize: 11
            text: latestFrameDebug
            visible: latestFrameDebug !== ""
        }
    }

    // Controls row
    Rectangle {
        id: controls
        anchors.left: parent.left; anchors.right: parent.right; anchors.bottom: parent.bottom
        height: 64
        color: "transparent"

        RowLayout {
            anchors.fill: parent
            anchors.margins: 12
            spacing: 12

            // Spacer left
            Item { Layout.preferredWidth: 8 }

            Button {
                text: "Start Stream"
                Layout.alignment: Qt.AlignVCenter
                background: Rectangle { color: "#40526a"; radius: 8 }
                onClicked: {
                    recording = true
                    if (cameraController && cameraController.start) cameraController.start()
                }
            }

            Button {
                text: "Stop Stream"
                Layout.alignment: Qt.AlignVCenter
                background: Rectangle { color: "#40526a"; radius: 8 }
                onClicked: {
                    recording = false
                    if (cameraController && cameraController.stop) cameraController.stop()
                }
            }

            Button {
                text: "Capture"
                Layout.alignment: Qt.AlignVCenter
                background: Rectangle { color: "#40526a"; radius: 8 }
                onClicked: {
                    if (cameraController && cameraController.capture) cameraController.capture()
                }
            }

            // spacer to push buttons to left
            Item { Layout.fillWidth: true }

            // optional frame indicator
            Rectangle {
                width: 44; height: 44; radius: 8
                color: "#2f3f4f"
                border.color: "#24313a"
                border.width: 1
                Row {
                    anchors.centerIn: parent
                    spacing: 6
                    Text { text: "FPS"; color: "#9aa7b3"; font.pixelSize: 11 }
                    Text { text: ""; color: "#9aa7b3"; font.pixelSize: 11 } // populate if you expose fps
                }
            }
        }
    }

    // connections to cameraController signals
    Connections {
        target: cameraController
        onVideoFrame: {
            // if controller emits data URI (data:image/...), show it
            if (typeof message === "string") {
                if (message.indexOf("data:image") === 0) {
                    feedImage.source = message
                    latestFrameDebug = ""
                } else {
                    // not an image (debug payload), show in text area
                    latestFrameDebug = message
                }
            }
        }
        onLogMessage: {
            console.log("drone camera log:", message)
        }
    }

    Component.onCompleted: {
        console.log("DroneCameraView loaded; cameraController:", cameraController)
    }
}