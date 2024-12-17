import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

ApplicationWindow {
    visible: true
    width: 1200
    height: 800
    title: "Avatar - Brainwave Reading"

    ColumnLayout {
        anchors.fill: parent
        spacing: 10

        // Tab Bar for Navigation
        TabBar {
            id: tabBar
            Layout.fillWidth: true
            height: 40

            TabButton {
                text: "Brainwave Reading"
                onClicked: stackLayout.currentIndex = 0
            }
            TabButton {
                text: "Transfer Data"
                onClicked: stackLayout.currentIndex = 1
            }
            TabButton {
                text: "Manual Drone Control"
                onClicked: stackLayout.currentIndex = 2
            }
        }

        // StackLayout for Switching Views
        StackLayout {
            id: stackLayout
            Layout.fillWidth: true
            Layout.fillHeight: true

            // Brainwave Reading View
            Rectangle {
                color: "#3b4b57"
                Layout.fillWidth: true
                Layout.fillHeight: true

                ColumnLayout {
                    spacing: 20
                    anchors.centerIn: parent

                    // Brainwave Image Section
                    Item {
                        Layout.preferredWidth: 200
                        Layout.preferredHeight: 200

                        Image {
                            source: "images/brain.png"
                            anchors.centerIn: parent
                            width: 120
                            height: 120
                            fillMode: Image.PreserveAspectFit
                        }
                    }

                    // "Read My Mind" Button
                    Button {
                        text: "Read my mind..."
                        Layout.alignment: Qt.AlignHCenter
                        onClicked: {
                            console.log("Read my mind clicked");
                            backend.readMyMind(); // Ensure `backend` is correctly defined elsewhere.
                        }
                    }

                    // Manual Command Input Section
                    TextField {
                        id: manualInput
                        placeholderText: "Enter manual command..."
                        Layout.fillWidth: true
                        Layout.preferredWidth: 400
                    }

                    Button {
                        text: "Execute"
                        Layout.preferredWidth: 120
                        Layout.alignment: Qt.AlignHCenter
                        onClicked: {
                            console.log("Executing manual action...");
                            backend.executeAction(); // Call backend logic.
                        }
                    }
                }
            }

            // Transfer Data View
            Rectangle {
                color: "#4a5b7b"
                Layout.fillWidth: true
                Layout.fillHeight: true

                ScrollView {
                    anchors.centerIn: parent
                    width: Math.min(parent.width * 0.9, 600)
                    height: Math.min(parent.height * 0.9, contentHeight)

                    ColumnLayout {
                        spacing: 15

                        Label {
                            text: "Target IP"
                            color: "white"
                            Layout.alignment: Qt.AlignLeft
                        }
                        TextField { Layout.fillWidth: true }

                        Label {
                            text: "Target Username"
                            color: "white"
                            Layout.alignment: Qt.AlignLeft
                        }
                        TextField { Layout.fillWidth: true }

                        Label {
                            text: "Target Password"
                            color: "white"
                            Layout.alignment: Qt.AlignLeft
                        }
                        TextField {
                            Layout.fillWidth: true
                            echoMode: TextInput.Password
                        }

                        Button {
                            text: "Save Config"
                            Layout.alignment: Qt.AlignHCenter
                            onClicked: {
                                console.log("Save Configuration clicked.");
                            }
                        }
                    }
                }
            }

            // Manual Drone Control View
            Rectangle {
                color: "#d3d3d3"
                Layout.fillWidth: true
                Layout.fillHeight: true

                Text {
                    anchors.centerIn: parent
                    text: "Manual Drone Control View"
                    font.pixelSize: 20
                    color: "black"
                }
            }
        }
    }
}
