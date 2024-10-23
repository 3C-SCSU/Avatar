import QtQuick 2.5
import QtQuick.Controls 1.4
import QtQuick.Layouts 1.3

ApplicationWindow {
    visible: true
    width: 1200
    height: 800
    title: "Brainwave Reading Tab"
    color: "#3b4b57"

    Column {
        anchors.fill: parent

        // Tab Bar Row
        Row {
            spacing: 10
            anchors.horizontalCenter: parent.horizontalCenter

            Button { text: "Brainwave Reading"; onClicked: stackView.currentIndex = 0 }
            Button { text: "Manual Drone Control"; onClicked: stackView.currentIndex = 1 }
            Button { text: "Transfer Data"; onClicked: stackView.currentIndex = 2 }
        }

        // StackView for Tab Content
        StackView {
            id: stackView
            anchors.fill: parent
            initialItem: brainwaveReadingTab

            // Brainwave Reading Tab
            Item {
                id: brainwaveReadingTab
                width: parent.width
                height: parent.height

                Grid {
                    columns: 2
                    anchors.fill: parent
                    anchors.margins: 20

                    // Left Column Layout
                    Column {
                        spacing: 20
                        anchors.verticalCenter: parent.verticalCenter

                        // Control Mode Section
                        GroupBox {
                            title: "Control Mode"
                            Row {
                                RadioButton { text: "Manual Control"; checked: true }
                                RadioButton { text: "Autopilot" }
                            }
                        }

                        // Brainwave Image and Read Button
                        Image {
                            source: "brainwave-prediction-app/images/brain.png"
                            width: 120
                            height: 120
                            fillMode: Image.PreserveAspectFit
                        }

                        Button {
                            text: "Read my mind..."
                            width: 160
                            height: 40
                        }

                        // Server Response Section
                        GroupBox {
                            title: "Server Response"
                            Column {
                                Text { text: "Count: 1"; color: "white" }
                                Text { text: "Label: Forward"; color: "white" }
                            }
                        }

                        // Action Buttons
                        Row {
                            Button { text: "Not what I was thinking..."; width: 160; height: 40 }
                            Button { text: "Execute"; width: 160; height: 40 }
                        }

                        // Manual Command and Keep Alive
                        Row {
                            TextField { placeholderText: "Manual Command"; width: 200 }
                            Button { text: "Keep Drone Alive"; width: 160; height: 40 }
                        }

                        // Flight Log Section
                        GroupBox {
                            title: "Flight Log"
                            width: 300
                            height: 150
                        }

                        // Connect Button
                        Button {
                            text: "Connect"
                            width: 150
                            height: 50
                        }
                    }

                    // Right Column Layout
                    Column {
                        spacing: 20

                        // Predictions Table
                        GroupBox {
                            title: "Predictions Table"
                            width: 400
                            height: 300
                        }

                        // Console Log
                        GroupBox {
                            title: "Console Log"
                            width: 400
                            height: 200
                        }
                    }
                }
            }
        }
    }
}
