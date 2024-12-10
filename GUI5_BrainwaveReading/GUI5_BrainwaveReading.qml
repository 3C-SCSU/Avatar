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

        // Tab bar
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

        // Stack layout for switching views
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
                    spacing: 10
                    anchors.centerIn: parent

                    // Brainwave Image Section
                    Item {
                        width: 200
                        height: 200
                        Image {
                            source: "images/brain.png"
                            anchors.centerIn: parent
                            width: 100
                            height: 100
                            fillMode: Image.PreserveAspectFit
                        }

                        Button {
                            text: "Read my mind..."
                            anchors.centerIn: parent
                            onClicked: {
                                console.log("Read my mind clicked");
                                backend.readMyMind(); // Ensure `backend` is correctly defined elsewhere.
                            }
                        }
                    }

                    // Manual Input
                    TextField {
                        id: manualInput
                        placeholderText: "Enter manual command..."
                        Layout.fillWidth: true
                        Layout.preferredWidth: 400
                    }

                    Button {
                        text: "Execute"
                        Layout.preferredWidth: 120
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
                ScrollView {
                    anchors.centerIn: parent
                    width: Math.min(parent.width * 0.9, 600)
                    height: Math.min(parent.height * 0.9, contentHeight)

                    ColumnLayout {
                        spacing: 10

                        Label { text: "Target IP"; color: "white" }
                        TextField { Layout.fillWidth: true }

                        Label { text: "Target Username"; color: "white" }
                        TextField { Layout.fillWidth: true }

                        Label { text: "Target Password"; color: "white" }
                        TextField {
                            Layout.fillWidth: true
                            echoMode: TextInput.Password
                        }

                        Button {
                            text: "Save Config"
                            onClicked: console.log("Save Configuration clicked.")
                        }
                    }
                }
            }

            // Manual Drone Control View
            Rectangle {
                color: "lightgrey"
                Layout.fillWidth: true
                Layout.fillHeight: true
                Text {
                    anchors.centerIn: parent
                    text: "Manual Drone Control View"
                    color: "black"
                }
            }
        }
    }
}
