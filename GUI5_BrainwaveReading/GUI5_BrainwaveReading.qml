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

        // Stack layout for different views
        StackLayout {
            id: stackLayout
            Layout.fillWidth: true
            Layout.fillHeight: true

            // Brainwave Reading view
            Rectangle {
                color: "#3b4b57" // Background color
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
                                checked: true
                                contentItem: Text {
                                    text: "Manual Control"
                                    color: "white"
                                    font.bold: true
                                    anchors.centerIn: parent
                                }
                            }

                            RadioButton {
                                contentItem: Text {
                                    text: "Autopilot"
                                    color: "white"
                                    font.bold: true
                                    anchors.centerIn: parent
                                }
                            }
                        }

                        // Brainwave Image with Transparent Button
                        Rectangle {
                            width: 150
                            height: 150
                            color: "#1b3a4b" // Dark blue background
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
                                background: Item {} // No background
                                contentItem: Text {
                                    text: "Read my mind..."
                                    color: "white" // Set text color to white
                                    anchors.centerIn: parent
                                }
                                onClicked: backend.readMyMind()
                            }
                        }

                        // Model Prediction Section
                        Label {
                            text: "The model says ..."
                            color: "white"
                            font.bold: true
                            Layout.alignment: Qt.AlignHCenter
                        }

                        GroupBox {
                            Layout.preferredWidth: 300
                            Layout.preferredHeight: 80
                            Layout.alignment: Qt.AlignHCenter

                            // Header with white background
                            RowLayout {
                                spacing: 1
                                Rectangle {
                                    color: "white"
                                    width: 145
                                    height: 20
                                    Text {
                                        text: "Count"
                                        font.bold: true
                                        color: "black"
                                        anchors.centerIn: parent
                                    }
                                }
                                Rectangle {
                                    color: "white"
                                    width: 145
                                    height: 20
                                    Text {
                                        text: "Label"
                                        font.bold: true
                                        color: "black"
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
                                    Text { text: model.count; color: "white"; width: 80 }
                                    Text { text: model.label; color: "white"; width: 80 }
                                }
                            }
                        }
                         // Action Buttons
                            RowLayout {
                                spacing: 10
                                Layout.alignment: Qt.AlignHCenter

                                Button {
                                    Layout.preferredWidth: 160
                                    Layout.preferredHeight: 80
                                    background: Rectangle {
                                        color: "#1b3a4b"
                                    } 
                                    contentItem: Text {
                                        text: "Not what I was thinking..."
                                        color: "white"
                                        font.bold: true
                                        anchors.centerIn: parent
                                    }
                                    onClicked: backend.notWhatIWasThinking(manualInput.text)
                                }

                                Button {
                                    Layout.preferredWidth: 160
                                    Layout.preferredHeight: 80
                                    background: Rectangle {
                                        color: "#1b3a4b"
                                    } 
                                    contentItem: Text {
                                        text: "Execute"
                                        color: "white"
                                        font.bold: true
                                        anchors.centerIn: parent
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
                                Layout.preferredWidth: 400
                                Layout.alignment: Qt.AlignHCenter
                                background: Rectangle{
                                    color: "white"
                                    radius: 5
                                }
                                
                            }
                           Button {
                            width: 130
                            height: 40
                            background: Rectangle {
                                color: "#1b3a4b"
                            }
                            contentItem: Text {
                                text: "Keep Drone Alive"
                                color: "white"
                                font.bold: true
                                anchors.centerIn: parent
                            }
                            onClicked: backend.keepDroneAlive()
                        }

                        // Flight Log
                        GroupBox {
                            Layout.preferredWidth: 230
                            Layout.preferredHeight: 170
                            
                            ColumnLayout {
                                spacing: 5
                                anchors.fill: parent

                                Text {
                                    text: "Flight Log"
                                    font.bold: true
                                    font.pixelSize: 20
                                    color: "white"
                                    horizontalAlignment: Text.AlignHCenter
                                    Layout.alignment: Qt.AlignHCenter
                                }

                            // Background Rectangle inside the ListView
                            Rectangle {
                                color: "white"  // Set only the box area color to white
                                anchors.fill: parent

                                ListView {
                                    id: flightLogView
                                    anchors.fill: parent  // Fill the Rectangle background with ListView content
                                    model: ListModel {
                                        
                                    }
                                    delegate: Text {
                                        text: log
                                        color: "black"  // Set text color for readability
                                        anchors.horizontalCenter: parent.horizontalCenter
                                    }
                                }
                            }
                        }


                        // Connect Image with Transparent Button
                        Rectangle {
                            width: 150
                            height: 150
                            color: "#1b3a4b" // Dark blue background

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
                                background: Item {} // No background
                                contentItem: Text {
                                    text: "Connect"
                                    color: "white"
                                    font.bold: true
                                    anchors.centerIn: parent
                                }
                                onClicked: backend.connectDrone()
                            }
                        }


                    // Right Column (Prediction Table and Console Log)
                    ColumnLayout {
                        Layout.preferredWidth: 700
                        Layout.fillHeight: true
                        spacing: 10

                        // Predictions Table
                        GroupBox {
                            Layout.preferredWidth: 700
                            Layout.preferredHeight: 550
                            
                             ColumnLayout {
                                spacing: 5
                                anchors.fill: parent

                                Text {
                                    text: "Predictions Table"
                                    font.bold: true
                                    font.pixelSize: 20
                                    color: "white"
                                    horizontalAlignment: Text.AlignHCenter
                                    Layout.alignment: Qt.AlignHCenter
                                }


                            // Header with white background
                            RowLayout {
                                spacing: 1
                                Rectangle {
                                    color: "white"
                                    width: 230
                                    height: 20
                                    Text {
                                        text: "Predictions Count"
                                        font.bold: true
                                        color: "black"
                                        anchors.centerIn: parent
                                    }
                                }
                                Rectangle {
                                    color: "white"
                                    width: 230
                                    height: 20
                                    Text {
                                        text: "Server Predictions"
                                        font.bold: true
                                        color: "black"
                                        anchors.centerIn: parent
                                    }
                                }
                                Rectangle {
                                    color: "white"
                                    width: 230
                                    height: 20
                                    Text {
                                        text: "Prediction Label"
                                        font.bold: true
                                        color: "black"
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
                                    Text { text: model.count; color: "white"; width: 120 }
                                    Text { text: model.server; color: "white"; width: 200 }
                                    Text { text: model.label; color: "white"; width: 120 }
                                }
                            }
                        }

                    


                        // Console Log
                        GroupBox {
                            Layout.preferredWidth: 230
                            Layout.preferredHeight: 170
                            
                            ColumnLayout {
                                spacing: 5
                                anchors.fill: parent

                                Text {
                                    text: "Console Log"
                                    font.weight: Font.Bold
                                    font.pixelSize: 20
                                    color: "white"
                                    horizontalAlignment: Text.AlignHCenter
                                    Layout.alignment: Qt.AlignHCenter
                                }

                            // Background Rectangle inside the ListView
                            Rectangle {
                                color: "white"  // Set only the box area color to white
                                anchors.fill: parent

                                ListView {
                                    id: consolelog
                                    anchors.fill: parent  // Fill the Rectangle background with ListView content
                                    model: ListModel {
                                        
                                    }
                                    delegate: Text {
                                        text: log
                                        color: "black"  // Set text color for readability
                                        anchors.horizontalCenter: parent.horizontalCenter
                                    }
                                }
                            }
                        }
                    }
                }
            }  
                    
                                                       

            // Transfer Data view
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

                        Label {
                            text: "Target IP"
                            color: "white"
                            font.bold: true
                        }
                        TextField { Layout.fillWidth: true }

                        Label {
                            text: "Target Username"
                            color: "white"
                            font.bold: true
                        }
                        TextField { Layout.fillWidth: true }

                        Label {
                            text: "Target Password"
                            color: "white"
                            font.bold: true
                        }
                        TextField {
                            Layout.fillWidth: true
                            echoMode: TextInput.Password
                        }

                        Label {
                            text: "Private Key Directory:"
                            color: "white"
                            font.bold: true
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            TextField {
                                id: privateKeyDirInput
                                Layout.fillWidth: true
                            }
                            Button {
                                text: "Browse"
                                onClicked: console.log("Browse for Private Key Directory")
                            }
                        }

                        CheckBox {
                            text: "Ignore Host Key"
                            checked: true
                            contentItem: Text {
                                text: parent.text
                                color: "white"
                                leftPadding: parent.indicator.width + parent.spacing
                            }
                        }

                        Label {
                            text: "Source Directory:"
                            color: "white"
                            font.bold: true
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            TextField {
                                id: sourceDirInput
                                Layout.fillWidth: true
                            }
                            Button {
                                text: "Browse"
                                onClicked: console.log("Browse for Source Directory")
                            }
                        }

                        Label {
                            text: "Target Directory:"
                            color: "white"
                            font.bold: true
                        }

                        TextField {
                            Layout.fillWidth: true
                            text: "/home/"
                            placeholderText: "/home/"
                        }

                       RowLayout {
                            Layout.fillWidth: true

                            Button {
                                background: Rectangle { color: "#1b3a4b" }
                                contentItem: Text {
                                    text: "Save Config"
                                    color: "white"
                                    font.bold: true
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                }
                                onClicked: console.log("Save Config clicked")
                            }

                            Button {
                                background: Rectangle { color: "#1b3a4b" }
                                contentItem: Text {
                                    text: "Load Config"
                                    color: "white"
                                    font.bold: true
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                }
                                onClicked: console.log("Load Config clicked")
                            }

                            Button {
                                background: Rectangle { color: "#1b3a4b" }
                                contentItem: Text {
                                    text: "Clear Config"
                                    color: "white"
                                    font.bold: true
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                }
                                onClicked: console.log("Clear Config clicked")
                            }

                            Button {
                                background: Rectangle { color: "#1b3a4b" }
                                contentItem: Text {
                                    text: "Upload"
                                    color: "white"
                                    font.bold: true
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                }
                                onClicked: console.log("Upload clicked")
                            }
                        }


                    }  // end of ColumnLayout (contentLayout)
                }  // end of ScrollView
            }  // end of Transfer Data Rectangle

            // Manual Drone Control view
            Rectangle {
                color: "lightgrey"
                Layout.fillWidth: true
                Layout.fillHeight: true
                Text {
                    anchors.centerIn: parent
                    text: "Manual Drone Control View"
                }
            }
        }  // end of StackLayout
    }  // end of outer ColumnLayout
}  // end of ApplicationWindow
                }
            }
        }
    }
}