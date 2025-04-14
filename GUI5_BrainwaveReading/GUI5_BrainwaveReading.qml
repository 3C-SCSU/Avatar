import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

ApplicationWindow {
    visible: true
    width: 1200
    height: 800 
    title: "Avatar - Brainwave Reading"

    // Global text color setting
    property color textColor: "white"
    property color headerBackground: "#1b3a4b"
    property color inputBackground: "#2c3e50"

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
                contentItem: Text {
                    text: parent.text
                    color: textColor
                    horizontalAlignment: Text.AlignHCenter
                }
            }
            TabButton {
                text: "Transfer Data"
                onClicked: stackLayout.currentIndex = 1
                contentItem: Text {
                    text: parent.text
                    color: textColor
                    horizontalAlignment: Text.AlignHCenter
                }
            }
            TabButton {
                text: "Manual Drone Control"
                onClicked: stackLayout.currentIndex = 2
                contentItem: Text {
                    text: parent.text
                    color: textColor
                    horizontalAlignment: Text.AlignHCenter
                }
            }
        }

        // Stack layout for different views
        StackLayout {
            id: stackLayout
            Layout.fillWidth: true
            Layout.fillHeight: true

            // Brainwave Reading view
            Rectangle {
                color: "#3b4b57"
                Layout.fillWidth: true
                Layout.fillHeight: true

                RowLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    spacing: 20

                    // Left Column (Components)
                    ColumnLayout {
                        Layout.preferredWidth: 600
                        Layout.fillHeight: true
                        spacing: 10

                        // Control Mode
                        RowLayout {
                            Layout.alignment: Qt.AlignHCenter
                            spacing: 10
                            RadioButton {
                                text: "Manual Control"
                                checked: true
                                contentItem: Text {
                                    text: parent.text
                                    color: textColor
                                    leftPadding: parent.indicator.width + parent.spacing
                                }
                            }
                            RadioButton {
                                text: "Autopilot"
                                contentItem: Text {
                                    text: parent.text
                                    color: textColor
                                    leftPadding: parent.indicator.width + parent.spacing
                                }
                            }
                        }

                        // Brainwave Image with Transparent Button
                        Rectangle {
                            width: 150
                            height: 150
                            color: headerBackground
                            Layout.alignment: Qt.AlignHCenter

                            Image {
                                source: "GUI_Pics/brain.png"
                                width: 130
                                height: 130
                                anchors.centerIn: parent
                                fillMode: Image.PreserveAspectFit
                            }

                            Button {
                                width: 130
                                height: 130
                                anchors.centerIn: parent
                                background: Item {}
                                contentItem: Text {
                                    text: "Read my mind..."
                                    color: textColor
                                    anchors.centerIn: parent
                                }
                                onClicked: backend.readMyMind()
                            }
                        }

                        // Model Prediction Section
                        Label {
                            text: "The model says ..."
                            color: textColor
                            Layout.alignment: Qt.AlignHCenter
                        }

                        GroupBox {
                            Layout.preferredWidth: 300
                            Layout.preferredHeight: 80
                            Layout.alignment: Qt.AlignHCenter
                            label: Label {
                                color: textColor
                                text: parent.title
                            }

                            // Header
                            RowLayout {
                                spacing: 1
                                Rectangle {
                                    color: headerBackground
                                    width: 145
                                    height: 20
                                    Text {
                                        text: "Count"
                                        font.bold: true
                                        color: textColor
                                        anchors.centerIn: parent
                                    }
                                }
                                Rectangle {
                                    color: headerBackground
                                    width: 145
                                    height: 20
                                    Text {
                                        text: "Label"
                                        font.bold: true
                                        color: textColor
                                        anchors.centerIn: parent
                                    }
                                }
                            }

                            ListView {
                                id: predictionListView
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                model: ListModel {}
                                delegate: RowLayout {
                                    spacing: 150
                                    Text { 
                                        text: model.count
                                        color: textColor
                                        width: 80 
                                    }
                                    Text { 
                                        text: model.label
                                        color: textColor
                                        width: 80 
                                    }
                                }
                            }
                        }

                        // Action Buttons
                        RowLayout {
                            spacing: 10
                            Layout.alignment: Qt.AlignHCenter
                            Button {
                                text: "Not what I was thinking..."
                                Layout.preferredWidth: 160
                                Layout.preferredHeight: 80
                                background: Rectangle {
                                    color: headerBackground
                                }
                                contentItem: Text {
                                    text: parent.text
                                    color: textColor
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                }
                                onClicked: backend.notWhatIWasThinking(manualInput.text)
                            }
                            Button {
                                text: "Execute"
                                Layout.preferredWidth: 160
                                Layout.preferredHeight: 80
                                background: Rectangle {
                                    color: headerBackground
                                }
                                contentItem: Text {
                                    text: parent.text
                                    color: textColor
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                }
                                onClicked: backend.executeAction()
                            }
                        }

                        // Manual Input and Keep Alive
                        GridLayout {
                            columns: 2
                            columnSpacing: 10
                            Layout.fillWidth: true
                            Layout.alignment: Qt.AlignHCenter
                            TextField {
                                id: manualInput
                                placeholderText: "Manual Command"
                                placeholderTextColor: "lightgray"
                                color: textColor
                                Layout.preferredWidth: 400
                                Layout.alignment: Qt.AlignHCenter
                                background: Rectangle {
                                    color: inputBackground
                                    radius: 5
                                }
                            }
                            Button {
                                text: "Keep Drone Alive"
                                width: 130
                                height: 40
                                background: Rectangle {
                                    color: headerBackground
                                }
                                contentItem: Text {
                                    text: parent.text
                                    color: textColor
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                }
                                onClicked: backend.keepDroneAlive()
                            }
                        }

                        // Flight Log
                        GroupBox {
                            title: "Flight Log"
                            Layout.preferredWidth: 230
                            Layout.preferredHeight: 170
                            label: Label {
                                color: textColor
                                text: parent.title
                            }
                            Rectangle {
                                color: headerBackground
                                anchors.fill: parent
                                ListView {
                                    id: flightLogView
                                    anchors.fill: parent
                                    model: ListModel {}
                                    delegate: Text {
                                        text: log
                                        color: textColor
                                        anchors.horizontalCenter: parent.horizontalCenter
                                    }
                                }
                            }
                        }

                        // Connect Image with Transparent Button
                        Rectangle {
                            width: 150
                            height: 150
                            color: headerBackground
                            Layout.alignment: Qt.AlignHCenter

                            Image {
                                source: "GUI_Pics/connect.png"
                                width: 80
                                height: 80
                                anchors.centerIn: parent
                                fillMode: Image.PreserveAspectFit
                            }

                            Button {
                                width: 80
                                height: 80
                                anchors.centerIn: parent
                                background: Item {}
                                contentItem: Text {
                                    text: "Connect"
                                    color: textColor
                                    anchors.centerIn: parent
                                }
                                onClicked: backend.connectDrone()
                            }
                        }
                    }

                    // Right Column (Prediction Table and Console Log)
                    ColumnLayout {
                        Layout.preferredWidth: 700
                        Layout.fillHeight: true
                        spacing: 10

                        // Predictions Table
                        GroupBox {
                            title: "Predictions Table"
                            Layout.preferredWidth: 700
                            Layout.preferredHeight: 550
                            label: Label {
                                color: textColor
                                text: parent.title
                            }

                            // Header
                            RowLayout {
                                spacing: 1
                                Rectangle {
                                    color: headerBackground
                                    width: 230
                                    height: 20
                                    Text {
                                        text: "Predictions Count"
                                        font.bold: true
                                        color: textColor
                                        anchors.centerIn: parent
                                    }
                                }
                                Rectangle {
                                    color: headerBackground
                                    width: 230
                                    height: 20
                                    Text {
                                        text: "Server Predictions"
                                        font.bold: true
                                        color: textColor
                                        anchors.centerIn: parent
                                    }
                                }
                                Rectangle {
                                    color: headerBackground
                                    width: 230
                                    height: 20
                                    Text {
                                        text: "Prediction Label"
                                        font.bold: true
                                        color: textColor
                                        anchors.centerIn: parent
                                    }
                                }
                            }

                            ListView {
                                Layout.preferredWidth: 700
                                Layout.preferredHeight: 550
                                model: ListModel {
                                    ListElement { count: "1"; server: "Prediction A"; label: "Label A" }
                                    ListElement { count: "2"; server: "Prediction B"; label: "Label B" }
                                }
                                delegate: RowLayout {
                                    spacing: 50
                                    Text { 
                                        text: model.count
                                        color: textColor
                                        width: 120 
                                    }
                                    Text { 
                                        text: model.server
                                        color: textColor
                                        width: 200 
                                    }
                                    Text { 
                                        text: model.label
                                        color: textColor
                                        width: 120 
                                    }
                                }
                            }
                        }

                        // Console Log
                        GroupBox {
                            title: "Console Log"
                            Layout.preferredWidth: 230
                            Layout.preferredHeight: 170
                            label: Label {
                                color: textColor
                                text: parent.title
                            }
                            Rectangle {
                                color: headerBackground
                                anchors.fill: parent
                                ListView {
                                    id: consolelog
                                    anchors.fill: parent
                                    model: ListModel {}
                                    delegate: Text {
                                        text: log
                                        color: textColor
                                        anchors.horizontalCenter: parent.horizontalCenter
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Transfer Data view (unchanged)
            Rectangle {
                color: "#4a5b7b"
                ScrollView {
                    anchors.centerIn: parent
                    width: Math.min(parent.width * 0.9, 600)
                    height: Math.min(parent.height * 0.9, contentHeight)
                    clip: true

                    ColumnLayout {
                        id: contentLayout
                        width: parent.width
                        spacing: 10

                        Label { text: "Target IP"; color: textColor }
                        TextField { 
                            Layout.fillWidth: true 
                            color: textColor
                            background: Rectangle { color: inputBackground }
                        }

                        Label { text: "Target Username"; color: textColor }
                        TextField { 
                            Layout.fillWidth: true 
                            color: textColor
                            background: Rectangle { color: inputBackground }
                        }

                        Label { text: "Target Password"; color: textColor }
                        TextField {
                            Layout.fillWidth: true
                            echoMode: TextInput.Password
                            color: textColor
                            background: Rectangle { color: inputBackground }
                        }

                        Label { text: "Private Key Directory:"; color: textColor }
                        RowLayout {
                            Layout.fillWidth: true
                            TextField {
                                id: privateKeyDirInput
                                Layout.fillWidth: true
                                color: textColor
                                background: Rectangle { color: inputBackground }
                            }
                            Button {
                                text: "Browse"
                                contentItem: Text {
                                    text: parent.text
                                    color: textColor
                                }
                                background: Rectangle { color: headerBackground }
                            }
                        }

                        CheckBox {
                            text: "Ignore Host Key"
                            checked: true
                            contentItem: Text {
                                text: parent.text
                                color: textColor
                                leftPadding: parent.indicator.width + parent.spacing
                            }
                        }

                        Label { text: "Source Directory:"; color: textColor }
                        RowLayout {
                            Layout.fillWidth: true
                            TextField {
                                id: sourceDirInput
                                Layout.fillWidth: true
                                color: textColor
                                background: Rectangle { color: inputBackground }
                            }
                            Button {
                                text: "Browse"
                                contentItem: Text {
                                    text: parent.text
                                    color: textColor
                                }
                                background: Rectangle { color: headerBackground }
                            }
                        }

                        Label { text: "Target Directory:"; color: textColor }
                        TextField {
                            Layout.fillWidth: true
                            text: "/home/"
                            placeholderText: "/home/"
                            color: textColor
                            background: Rectangle { color: inputBackground }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            Button {
                                text: "Save Config"
                                contentItem: Text {
                                    text: parent.text
                                    color: textColor
                                }
                                background: Rectangle { color: headerBackground }
                            }
                            Button {
                                text: "Load Config"
                                contentItem: Text {
                                    text: parent.text
                                    color: textColor
                                }
                                background: Rectangle { color: headerBackground }
                            }
                            Button {
                                text: "Clear Config"
                                contentItem: Text {
                                    text: parent.text
                                    color: textColor
                                }
                                background: Rectangle { color: headerBackground }
                            }
                            Button {
                                text: "Upload"
                                contentItem: Text {
                                    text: parent.text
                                    color: textColor
                                }
                                background: Rectangle { color: headerBackground }
                            }
                        }
                    }
                }
            }

            // Manual Drone Control view (unchanged)
            Rectangle {
                color: "#3b4b57"
                Layout.fillWidth: true
                Layout.fillHeight: true
                Text {
                    anchors.centerIn: parent
                    text: "Manual Drone Control View"
                    color: textColor
                }
            }
        }
    }
}
