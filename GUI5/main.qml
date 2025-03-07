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
            background: Rectangle {
                color: "#2C3E50"  //Dark
            }

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
                                text: "Manual Control"
                                checked: true
                            }
                            RadioButton {
                                text: "Autopilot"
                            }
                        }

                        // Brainwave Image with Transparent Button
                        Rectangle {
                            width: 150
                            height: 150
                            color: "#1b3a4b" // Dark blue background
                            Layout.alignment: Qt.AlignHCenter
          

                            Image {
                                source: "images/brain.png"
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
                            Layout.alignment: Qt.AlignHCenter
                        }

                        GroupBox {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
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
                                text: backend.current_prediction_label //Automatic label generated from backend
                                Layout.preferredWidth: 160
                                Layout.preferredHeight: 80
                                background: Rectangle {
                                    color: "#1b3a4b"
                                }
                                contentItem: Text {
                                    text: parent.text  // Use the Button's `text` property
                                    font.pixelSize: 15  // Set the font size
                                    color: "white"  // Set the text color
                                    anchors.centerIn: parent  // Center the text
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
                                    color: "#1b3a4b"
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
                            }
                            Button {
                                text: "Keep Drone Alive"
                                width: 130
                                height: 40
                                background: Rectangle {
                                    color: "#1b3a4b"
                                }
                                onClicked: backend.keepDroneAlive()
                            }
                        }

                        // Flight Log
                        GroupBox {
                            title: "Flight Log"
                            Layout.preferredWidth: 230
                            Layout.preferredHeight: 170
                            Flickable {
                                id: flightLogFlickable
                                width: parent.width
                                height: parent.height

                                contentHeight: flightLogView.contentHeight
                                clip: true

                                ColumnLayout {
                                    ListView {
                                        id: flightLogView
                                        Layout.preferredWidth: 230
                                        Layout.preferredHeight: 170
                                        model:ListModel {}
                                        delegate: Text {
                                            text: log
                                            color: "white"
                                        }
                                    }
                                }
                                ScrollBar.vertical: ScrollBar {
                                    width: 10
                                    policy: ScrollBar.AlwaysOn
                                }
                            }

                        }


                        // Connect Image with Transparent Button
                        Rectangle {
                            width: 150
                            height: 150
                            color: "#1b3a4b" // Dark blue background

                            Image {
                                source: "images/connect.png"
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
                                    color: "white" // Set text color to white
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

                        // Console Log Section
                        GroupBox {
                            title: "Console Log"
                            Layout.preferredWidth: 300
                            Layout.preferredHeight: 250
                            Layout.alignment: Qt.AlignRight

                            TextArea {
                                id: consoleLog
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                text: "Console output here..."
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

                        Label { text: "Target IP"; color: "white" }
                        TextField { Layout.fillWidth: true }

                        Label { text: "Target Username"; color: "white" }
                        TextField { Layout.fillWidth: true }

                        Label { text: "Target Password"; color: "white" }
                        TextField {
                            Layout.fillWidth: true
                            echoMode: TextInput.Password
                        }

                        Label { text: "Private Key Directory:"; color: "white" }
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

                        Label { text: "Source Directory:"; color: "white" }
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

                        Label { text: "Target Directory:"; color: "white" }
                        TextField {
                            Layout.fillWidth: true
                            text: "/home/"
                            placeholderText: "/home/"
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            Button {
                                text: "Save Config"
                                onClicked: console.log("Save Config clicked")
                            }
                            Button {
                                text: "Load Config"
                                onClicked: console.log("Load Config clicked")
                            }
                            Button {
                                text: "Clear Config"
                                onClicked: console.log("Clear Config clicked")
                            }
                            Button {
                                text: "Upload"
                                onClicked: console.log("Upload clicked")
                            }
                        }
                    }
                }
            }


            // Manual Drone Control view - completely revised to match the image
            Rectangle {
                id: droneControlView
                color: "#2C3E50" // Dark blue background matching the image
                Layout.fillWidth: true
                Layout.fillHeight: true

                GridLayout {
                    anchors.fill: parent
                    rows: 5
                    columns: 5
                    rowSpacing: 5
                    columnSpacing: 5

                    // Row 1: Home, Up, Flight Log
                    DroneButton {
                        Layout.row: 0
                        Layout.column: 0
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        buttonText: "Home"
                        imagePath: "images/home.png"
                        onClicked: droneController.getDroneAction("home")
                    }

                    DroneButton {
                        Layout.row: 0
                        Layout.column: 1
                        Layout.columnSpan: 3
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        buttonText: "Up"
                        imagePath: "images/up.png"
                        onClicked: droneController.getDroneAction("up")
                    }

                    // Flight Log box
                    Rectangle {
                        Layout.row: 0
                        Layout.column: 4
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        color: "#2C3E50"
                        border.color: "#1B2631"
                        
                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 5
                            spacing: 5
                            
                            Text {
                                text: "Flight Log"
                                font.pixelSize: 16
                                color: "white"
                            }
                            
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                color: "white"
                                border.color: "#1B2631"
                                
                                TextArea {
                                    id: flightLogSpace
                                    width: 400 // Adjust to account for scrollbar width
                                    height: 100
                                    // Ensure vertical scrollbar is always on

                            
                                }
                                ScrollBar {
                                    id: flightLogScrollBar
                                    orientation: Qt.Vertical
                                    anchors.right: parent.right
                                    anchors.top: parent.top
                                    anchors.bottom: parent.bottom
                                    width: 20 // Set width for the scrollbar
                                }
                            }
                        }
                    }

                    // Row 2: Forward button (full width)
                    DroneButton {
                        Layout.row: 1
                        Layout.column: 0
                        Layout.columnSpan: 5
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        buttonText: "Forward"
                        imagePath: "images/forward.png"
                        onClicked: droneController.getDroneAction("forward")
                    }

                    // Row 3: Turn Left, Left, Stream, Right, Turn Right
                    DroneButton {
                        Layout.row: 2
                        Layout.column: 0
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.preferredWidth: 60
                        buttonText: "Turn Left"
                        imagePath: "images/turnLeft.png"
                        onClicked: droneController.getDroneAction("turn_left")
                    }

                    DroneButton {
                        Layout.row: 2
                        Layout.column: 1
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.preferredWidth: 60
                        buttonText: "Left"
                        imagePath: "images/left.png"
                        onClicked: droneController.getDroneAction("left")
                    }

                    DroneButton {
                        Layout.row: 2
                        Layout.column: 2
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.preferredWidth: 60
                        buttonText: "Stream"
                        imagePath: "images/drone.png"
                        onClicked: droneController.getDroneAction("stream")
                    }

                    DroneButton {
                        Layout.row: 2
                        Layout.column: 3
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.preferredWidth: 60
                        buttonText: "Right"
                        imagePath: "images/right.png"
                        onClicked: droneController.getDroneAction("right")
                    }

                    DroneButton {
                        Layout.row: 2
                        Layout.column: 4
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.preferredWidth: 60
                        buttonText: "Turn Right"
                        imagePath: "images/turnRight.png"
                        onClicked: droneController.getDroneAction("turn_right")
                    }

                    // Row 4: Back button (full width)
                    DroneButton {
                        Layout.row: 3
                        Layout.column: 0
                        Layout.columnSpan: 5
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        buttonText: "Back"
                        imagePath: "images/back.png"
                        onClicked: droneController.getDroneAction("backward")
                    }

                    // Row 5: Connect, Down, Empty, Takeoff, Land
                    DroneButton {
                        Layout.row: 4
                        Layout.column: 0
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        buttonText: "Connect"
                        imagePath: "images/connect.png"
                        onClicked: droneController.getDroneAction("connect")
                    }

                    DroneButton {
                        Layout.row: 4
                        Layout.column: 1
                        Layout.columnSpan: 3
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                       
                        buttonText: "Down"
                        imagePath: "images/down.png"
                        onClicked: droneController.getDroneAction("down")
                    }

                    // Container for Takeoff and Land buttons
                    RowLayout {
                        Layout.row: 4
                        Layout.column: 4
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        spacing: 5

                        DroneButton {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            Layout.row:4
                            Layout.column:2
                            Layout.preferredWidth:30
                    
                            buttonText: "Takeoff"
                            imagePath: "images/takeoff.png"
                            onClicked: droneController.getDroneAction("takeoff")
                        }

                        DroneButton {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            Layout.row:4
                            Layout.column:3
                            Layout.preferredWidth: 30
                        
                            buttonText: "Land"
                            imagePath: "images/land.png"
                            onClicked: droneController.getDroneAction("land")
                        }
                    }
                }
            }
        }
    }

    // Reusable component for drone control buttons
    component DroneButton: Button {
        property string buttonText: ""
        property string imagePath: ""
        
        background: Rectangle {
            color: parent.hovered ? "white" : "#242c4d"
            border.color: "#1B2631"
        }
        
        contentItem: Item {
            Image {
                source: imagePath
                width: Math.min(parent.width, parent.height) * 0.6
                height: width
                anchors.centerIn: parent
                fillMode: Image.PreserveAspectFit
            }
            
            Text {
                text: buttonText
                color: parent.parent.hovered ? "black" : "white"
                font.pixelSize: 14
                anchors.bottom: parent.bottom
                anchors.bottomMargin: 5
                anchors.horizontalCenter: parent.horizontalCenter
            }
        }
        
        hoverEnabled: true
    }
    
    // DroneController object to handle actions
    QtObject {
        id: droneController
        
        function getDroneAction(action) {
            console.log(action + " triggered")
            // Add to flight log
            flightLogArea.append(action + " command sent")
        }
    }
}
