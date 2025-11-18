import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Rectangle {
    id: cameraViewRoot
    width: 400
    height: 300
    color: "#2C3E50"
    border.color: "#34495E"
    border.width: 2
    radius: 10

    property var cameraController: null

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 10

        // Header
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 40
            color: "#34495E"
            radius: 5

            Text {
                anchors.centerIn: parent
                text: "CAMERA VIEW"
                font.pixelSize: 16
                font.bold: true
                color: "white"
            }
        }

        // Video Display Area
        Rectangle {
            id: videoContainer
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: "#1A252F"
            border.color: "#4A5B7B"
            border.width: 1
            radius: 5

            Image {
                id: videoFrame
                anchors.fill: parent
                anchors.margins: 5
                fillMode: Image.PreserveAspectFit
                source: ""
                
                // Placeholder when no video
                Rectangle {
                    anchors.centerIn: parent
                    width: 200
                    height: 100
                    color: "transparent"
                    visible: videoFrame.source == ""

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
            }

            // Stream status indicator
            Rectangle {
                anchors.top: parent.top
                anchors.right: parent.right
                anchors.margins: 10
                width: 20
                height: 20
                radius: 10
                color: "#E74C3C"
            }
        }

        // Control Buttons
        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 50
            spacing: 10

            Button {
                Layout.fillWidth: true
                Layout.preferredHeight: 40
                text: "Start Stream"

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
                    if (cameraController) {
                        cameraController.start_camera_stream()
                    }
                }
            }

            Button {
                Layout.fillWidth: true
                Layout.preferredHeight: 40
                text: "Stop Stream"

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
                    if (cameraController) {
                        cameraController.stop_camera_stream()
                    }
                }
            }

            Button {
                Layout.fillWidth: true
                Layout.preferredHeight: 40
                text: "Capture"

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
                    if (cameraController) {
                        cameraController.capture_photo()
                    }
                }
            }
        }
    }
}