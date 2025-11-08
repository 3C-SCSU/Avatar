import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15

Rectangle {
    id: droneCameraViewRoot
    anchors.fill: parent
    color: "#2C3E50"
    border.color: "#34495E"
    border.width: 2

    property var cameraController: null
    property bool recording: false
    property string latestFrameDebug: ""

    signal logToParent(string message)

    Connections {
        target: cameraController
        onVideoFrame: function(data) {
            // controller expecting to get emitted a data-uri string (data:image/...)
            if (typeof data === "string" && data.indexOf("data:image") === 0) {
                feedImage.source = data
                latestFrameDebug = ""
            } else {
                latestFrameDebug = data ? data.toString() : ""
            }
        }
        onLogMessage: function(msg) {
            console.log("drone camera log:", msg)
            logToParent(msg)
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
        radius: 5

        Text {
            anchors.centerIn: parent
            text: "CAMERA VIEW"
            color: "white"
            font.bold: true
            font.pixelSize: 16
        }
    }

    // Main feed area
    Rectangle {
        id: cameraFeedContainer
        anchors {
            top: header.bottom
            left: parent.left;
            right: parent.right
            bottom: controls.top
            margins: 12
        }
        color: "#11161a"
        radius: 6
        border.color: "#1f2933"
        border.width: 2
        clip: true

        // Image feed (fills wrapper)
        Image {
            id: videoFeed
            anchors.fill: parent
            fillMode: Image.PreserveAspectFit
            source: ""
            asynchronous: true
            cache: false
        }

        // Placeholder when no video
        Rectangle {
            anchors.centerIn: parent
            width: 200
            height: 100
            color: "transparent"
            visible: feedImage.source === "" || feedImage.status === Image.Error

            Column {
                anchors.centerIn: parent
                spacing: 10

                Rectangle {
                    anchors.horizontalCenter: parent.horizontalCenter
                    width: 64
                    height: 64
                    color: "#4A5B7B"
                    radius: 32

                    Text {
                        anchors.centerIn: parent
                        text: "📹"
                        font.pixelSize: 32
                        color: "white"
                    }
                }

                Text {
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: "Camera Ready"
                    font.pixelSize: 14
                    color: "#BDC3C7"
                }
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

        // Stream status indicator
        Rectangle {
            anchors.top: parent.top
            anchors.right: parent.right
            anchors.margins: 10
            width: 10
            height: 10
            radius: 10
            color: "#E74C3C"
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
            spacing: 10

            Button {
                text: "Start Stream"
                Layout.fillWidth: true
                Layout.preferredHeight: 40
                Layout.alignment: Qt.AlignVCenter
                background: Rectangle {
                    color: "#4A5B7B"
                    radius: 8
                    border.width: 1
                    border.color: "#3A4B6B"
                }
                contentItem: Text {
                    text: parent.text
                    font.pixelSize: 14
                    font.bold: true
                    color: "white"
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                onClicked: {
                    recording = true
                    logToParent("Starting camera stream")
                    if (cameraController && cameraController.start_camera_stream) {
                        cameraController.start_camera_stream()
                    }
                }
            }

            Button {
                text: "Stop Stream"
                Layout.fillWidth: true
                Layout.preferredHeight: 40
                Layout.alignment: Qt.AlignVCenter
                background: Rectangle {
                    color: "#4A5B7B"
                    radius: 8
                    border.width: 1
                    border.color: "#3A4B6B"
                }

                contentItem: Text {
                    text: parent.text
                    font.pixelSize: 14
                    font.bold: true
                    color: "white"
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                onClicked: {
                    recording = false
                    logToParent("Stopping camera stream")
                    if (cameraController && cameraController.stop_camera_stream) {
                        cameraController.stop_camera_stream()
                    }
                }
            }

            Button {
                text: "Capture"
                Layout.fillWidth: true
                Layout.preferredHeight: 40
                Layout.alignment: Qt.AlignVCenter
                background: Rectangle {
                    color: "#4A5B7B"
                    radius: 8
                    border.width: 1
                    border.color: "#3A4B6B"
                }
                contentItem: Text {
                    text: parent.text
                    font.pixelSize: 14
                    font.bold: true
                    color: "white"
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                onClicked: {
                    if (cameraController && cameraController.capture_photo) {
                        cameraController.capture_photo()
                    }
                }
            }
        }
    }

    Component.onCompleted: {
        console.log("DroneCameraView loaded; cameraController:", cameraController)
    }
}
