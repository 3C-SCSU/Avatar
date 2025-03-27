import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs


ApplicationWindow {
    property bool isRandomForestSelected: false
    visible: true
    width: 1200
    height: 800 
    title: "Avatar - Brainwave Reading"

    // Proper connection handling
    Connections {
        target: backend  // Now properly defined through context property
        function onImagesReady(imageData) {
            imageModel.clear();
            for (let item of imageData) {
                imageModel.append(item);
            }
        }
    }

    ListModel {
        id: imageModel
    }


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
            TabButton {
                text: "Brainwave Visualization"
                onClicked: stackLayout.currentIndex = 3
            }
            TabButton {
                text: "File Shuffler"
                onClicked: stackLayout.currentIndex = 4
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
                                source: "brainwave-prediction-app/images/brain.png"
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
                                text: "Not what I was thinking..."
                                Layout.preferredWidth: 160
                                Layout.preferredHeight: 80
                                background: Rectangle {
                                    color: "#1b3a4b"
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
                            ListView {
                                id: flightLogView
                                Layout.preferredWidth: 230
                                Layout.preferredHeight: 170
                                model: ListModel {}
                                delegate: Text {
                                    text: log
                                    color: "white"
                                }
                            }
                        }

                        // Connect Image with Transparent Button
                        RowLayout {
                            Layout.alignment: Qt.AlignHCenter
                            spacing: 20
                            Rectangle {
                                width: 150
                                height: 150
                                color: "#1b3a4b" // Dark blue background

                                Image {
                                    source: "brainwave-prediction-app/images/connect.png"
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
                            ColumnLayout {
                                spacing: 5
                                Layout.alignment: Qt.AlignHCenter
                                
                                // Radio Button
                                RadioButton {
                                    id: randomForestRadio
                                    Layout.alignment: Qt.AlignHCenter
                                    checked: isRandomForestSelected
                                }

                                // Green Box with Text
                                Rectangle {
                                    width: 150
                                    height: 80
                                    color: "#4CAF50"
                                    radius: 5

                                    Text {
                                        text: "Random Forest"
                                        font.bold: isRandomForestSelected
                                        font.pixelSize: 16
                                        color: "white"
                                        anchors.centerIn: parent
                                    }

                                    MouseArea {
                                        anchors.fill: parent
                                        onClicked: {
                                            isRandomForestSelected = true;
                                            backend.selectModel("Random Forest");
                                        }
                                    }
                                }
                            }
                            ColumnLayout {
                                spacing: 5
                                Layout.alignment: Qt.AlignHCenter

                                // Radio Button
                                RadioButton {
                                    id: deepLearningRadio
                                    Layout.alignment: Qt.AlignHCenter
                                    checked: !isRandomForestSelected
                                }
                                // Green Box with Text
                                Rectangle {
                                    width: 150
                                    height: 80
                                    color: "#4CAF50"
                                    radius: 5

                                    Text {
                                        text: "Deep Learning"
                                        font.pixelSize: 16
                                        color: "white"
                                        anchors.centerIn: parent
                                        font.bold: !isRandomForestSelected
                                    }
                                }

                                MouseArea {
                                    anchors.fill: parent
                                    onClicked: {
                                        isRandomForestSelected = false;
                                        backend.selectModel("Deep Learning");
                                    }
                                }
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

            // Manual Drone Control view
            ColumnLayout {
                anchors.fill: parent
                spacing: 10

                // Top Row - Home, Up, Flight Log
                RowLayout {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 150 // Larger height for the buttons


                    // Home Button
                    Button {
                        Layout.preferredWidth: 150 // Set larger width for Home button
                        Layout.preferredHeight: 150 // Set height for Home button

                        // Define the custom background
                        background: Rectangle {
                            id: buttonBackground // Give an ID to reference
                            color: "#242c4d" // Initial background color
                            border.color: "black" // Border color
                        }

                        // Stack Image and Text on top of each other and center them
                        contentItem: Item {
                            anchors.fill: parent

                            Image {
                                source: "brainwave-prediction-app/images/Home.png"
                                width: 150
                                height: 150
                                anchors.centerIn: parent
                            }

                            Text {
                                id: buttonText // Give an ID to reference
                                text: "Home"
                                color: "white" // Initial text color
                                font.pixelSize: 18
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.verticalCenter: parent.verticalCenter
                            }

                            MouseArea {
                                anchors.fill: parent
                                onEntered: {
                                    buttonBackground.color = "white"; // Change background to white on hover
                                    buttonText.color = "black"; // Change text color to black on hover
                                }
                                onExited: {
                                    buttonBackground.color = "#242c4d"; // Revert background color on exit
                                    buttonText.color = "white"; // Revert text color to white on exit
                                }
                                onClicked: {
                                    backend.getDroneAction("home");
                                }
                            }
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 150

                        // Up Button
                        Button {
                            Layout.preferredWidth: 1000
                            Layout.preferredHeight: 150

                            // Define the custom background
                            background: Rectangle {
                                id: upButtonBackground // Unique ID for the Up button background
                                color: "#242c4d" // Initial background color
                                border.color: "black" // Border color
                            }

                            // Stack Image and Text on top of each other and center them
                            contentItem: Item {
                                anchors.fill: parent

                                Image {
                                    source: "brainwave-prediction-app/images/Up.png"
                                    width: 150
                                    height: 150
                                    anchors.centerIn: parent
                                }

                                Text {
                                    id: upButtonText // Unique ID for the Up button text
                                    text: "Up"
                                    color: "white" // Initial text color
                                    font.pixelSize: 18
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    anchors.verticalCenter: parent.verticalCenter
                                }

                                MouseArea {
                                    anchors.fill: parent
                                    onEntered: {
                                        upButtonBackground.color = "white"; // Change background to white on hover
                                        upButtonText.color = "black"; // Change text color to black on hover
                                    }
                                    onExited: {
                                        upButtonBackground.color = "#242c4d"; // Revert background color on exit
                                        upButtonText.color = "white"; // Revert text color to white on exit
                                    }
                                    onClicked: {
                                        backend.getDroneAction("up");
                                    }
                                }
                            }
                        }
                        // Flight Log Label and Space beside Up Button
                        ColumnLayout {
                            spacing: 5

                            // Flight Log Label
                            Text {
                                id: flightlog
                                text: "Flight Log"
                                font.pixelSize: 20
                                color: "white"
                            }

                            // Flight Log TextArea (Box)
                            Rectangle {
                                width: 250
                                height: 100
                                color: "white"
                                border.color: "#2E4053"
                                anchors.leftMargin: 20

                                TextArea {
                                    id: flightLogSpace
                                    width: 400 // Adjust to account for scrollbar width
                                    height: 100
                                    // Ensure vertical scrollbar is always on

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
                    }

                    // Log ListView
                    ListView {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 150
                        model: logModel
                        delegate: Item {
                            Text {
                                text: modelData
                            }
                        }
                    }
                }

                // Forward Button
                RowLayout {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 150

                    // Forward Button
                    Button {
                        Layout.preferredWidth: 1400
                        Layout.preferredHeight: 150

                        // Define the custom background
                        background: Rectangle {
                            id: forwardBackground // Unique ID for the Forward button background
                            color: "#242c4d" // Initial background color
                            border.color: "black" // Border color
                        }

                        // Stack Image and Text on top of each other and center them
                        contentItem: Item {
                            anchors.fill: parent

                            Image {
                                source: "brainwave-prediction-app/images/Forward.png"
                                width: 150
                                height: 150
                                anchors.centerIn: parent
                            }

                            Text {
                                id: forwardText // Unique ID for the Forward button text
                                text: "Forward"
                                color: "white" // Initial text color
                                font.pixelSize: 18
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.verticalCenter: parent.verticalCenter
                            }

                            MouseArea {
                                anchors.fill: parent
                                onEntered: {
                                    forwardBackground.color = "white"; // Change background to white on hover
                                    forwardText.color = "black"; // Change text color to black on hover
                                }
                                onExited: {
                                    forwardBackground.color = "#242c4d"; // Revert background color on exit
                                    forwardText.color = "white"; // Revert text color to white on exit
                                }
                                onClicked: backend.getDroneAction("forward")
                            }
                        }
                    }
                }
                // Turn Left, Left, Stream, Right, Turn Right
                RowLayout {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 150

                    Button {
                        Layout.preferredWidth: 280
                        Layout.preferredHeight: 150

                        // Define the custom background
                        background: Rectangle {
                            id: turnLeftBackground // Unique ID for the Turn Left button background
                            color: "#242c4d" // Initial background color
                            border.color: "black" // Border color
                        }

                        // Stack Image and Text on top of each other and center them
                        contentItem: Item {
                            anchors.fill: parent

                            Image {
                                source: "brainwave-prediction-app/images/turnLeft.png"
                                width: 150
                                height: 150
                                anchors.centerIn: parent
                            }

                            Text {
                                id: turnLeftText // Unique ID for the Turn Left button text
                                text: "Turn Left"
                                color: "white" // Initial text color
                                font.pixelSize: 18
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.verticalCenter: parent.verticalCenter
                            }

                            MouseArea {
                                anchors.fill: parent
                                onEntered: {
                                    turnLeftBackground.color = "white"; // Change background to white on hover
                                    turnLeftText.color = "black"; // Change text color to black on hover
                                }
                                onExited: {
                                    turnLeftBackground.color = "#242c4d"; // Revert background color on exit
                                    turnLeftText.color = "white"; // Revert text color to white on exit
                                }
                                onClicked: {
                                    backend.getDroneAction("turn_left");
                                }
                            }
                        }
                    }

                    Button {
                        Layout.preferredWidth: 280
                        Layout.preferredHeight: 150

                        // Define the custom background
                        background: Rectangle {
                            id: leftBackground // Unique ID for the Left button background
                            color: "#242c4d" // Initial background color
                            border.color: "black" // Border color
                        }

                        // Stack Image and Text on top of each other and center them
                        contentItem: Item {
                            anchors.fill: parent

                            Image {
                                source: "brainwave-prediction-app/images/Left.png"
                                width: 150
                                height: 150
                                anchors.centerIn: parent
                            }

                            Text {
                                id: leftText // Unique ID for the Left button text
                                text: "Left"
                                color: "white" // Initial text color
                                font.pixelSize: 18
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.verticalCenter: parent.verticalCenter
                            }

                            MouseArea {
                                anchors.fill: parent
                                onEntered: {
                                    leftBackground.color = "white"; // Change background to white on hover
                                    leftText.color = "black"; // Change text color to black on hover
                                }
                                onExited: {
                                    leftBackground.color = "#242c4d"; // Revert background color on exit
                                    leftText.color = "white"; // Revert text color to white on exit
                                }
                                onClicked: {
                                    backend.getDroneAction("left");
                                }
                            }
                        }
                    }

                    Button {
                        Layout.preferredWidth: 280
                        Layout.preferredHeight: 150

                        // Define the custom background
                        background: Rectangle {
                            id: streamBackground // Unique ID for the Stream button background
                            color: "#242c4d" // Initial background color
                            border.color: "black" // Border color
                        }

                        // Stack Image and Text on top of each other and center them
                        contentItem: Item {
                            anchors.fill: parent

                            Image {
                                source: "brainwave-prediction-app/images/Stream.png"
                                width: 150
                                height: 150
                                anchors.centerIn: parent
                            }

                            Text {
                                id: streamText // Unique ID for the Stream button text
                                text: "Stream"
                                color: "white" // Initial text color
                                font.pixelSize: 18
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.verticalCenter: parent.verticalCenter
                            }

                            MouseArea {
                                anchors.fill: parent
                                onEntered: {
                                    streamBackground.color = "white"; // Change background to white on hover
                                    streamText.color = "black"; // Change text color to black on hover
                                }
                                onExited: {
                                    streamBackground.color = "#242c4d"; // Revert background color on exit
                                    streamText.color = "white"; // Revert text color to white on exit
                                }
                                onClicked: {
                                    backend.getDroneAction("stream");
                                }
                            }
                        }
                    }

                    Button {
                        Layout.preferredWidth: 280
                        Layout.preferredHeight: 150

                        // Define the custom background
                        background: Rectangle {
                            id: rightBackground // Unique ID for the Right button background
                            color: "#242c4d" // Initial background color
                            border.color: "black" // Border color
                        }

                        // Stack Image and Text on top of each other and center them
                        contentItem: Item {
                            anchors.fill: parent

                            Image {
                                source: "brainwave-prediction-app/images/right.png"
                                width: 150
                                height: 150
                                anchors.centerIn: parent
                            }

                            Text {
                                id: rightText // Unique ID for the Right button text
                                text: "Right"
                                color: "white" // Initial text color
                                font.pixelSize: 18
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.verticalCenter: parent.verticalCenter
                            }

                            MouseArea {
                                anchors.fill: parent
                                onEntered: {
                                    rightBackground.color = "white"; // Change background to white on hover
                                    rightText.color = "black"; // Change text color to black on hover
                                }
                                onExited: {
                                    rightBackground.color = "#242c4d"; // Revert background color on exit
                                    rightText.color = "white"; // Revert text color to white on exit
                                }
                                onClicked: {
                                    backend.getDroneAction("right");
                                }
                            }
                        }
                    }


                    Button {
                        Layout.preferredWidth: 280
                        Layout.preferredHeight: 150

                        // Define the custom background
                        background: Rectangle {
                            id: turnRightBackground // Unique ID for the Turn Right button background
                            color: "#242c4d" // Initial background color
                            border.color: "black" // Border color
                        }

                        // Stack Image and Text on top of each other and center them
                        contentItem: Item {
                            anchors.fill: parent

                            Image {
                                source: "brainwave-prediction-app/images/turnRight.png"
                                width: 150
                                height: 150
                                anchors.centerIn: parent
                            }

                            Text {
                                id: turnRightText // Unique ID for the Turn Right button text
                                text: "Turn Right"
                                color: "white" // Initial text color
                                font.pixelSize: 18
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.verticalCenter: parent.verticalCenter
                            }

                            MouseArea {
                                anchors.fill: parent
                                onEntered: {
                                    turnRightBackground.color = "white"; // Change background to white on hover
                                    turnRightText.color = "black"; // Change text color to black on hover
                                }
                                onExited: {
                                    turnRightBackground.color = "#242c4d"; // Revert background color on exit
                                    turnRightText.color = "white"; // Revert text color to white on exit
                                }
                                onClicked: {
                                    backend.getDroneAction("turn_right");
                                }
                            }
                        }
                    }
                }

                // Back Button
                RowLayout {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 150

                    Button {
                        Layout.preferredWidth: 1400
                        Layout.preferredHeight: 150

                        // Define the custom background
                        background: Rectangle {
                            id: backButtonBackground // Unique ID for the Back button background
                            color: "#242c4d"    // Initial background color
                            border.color: "black" // Border color
                        }

                        // Stack Image and Text on top of each other and center them
                        contentItem: Item {
                            anchors.fill: parent

                            Image {
                                source: "brainwave-prediction-app/images/back.png"
                                width: 150
                                height: 150
                                anchors.centerIn: parent
                            }

                            Text {
                                id: backButtonText // Unique ID for the Back button text
                                text: "Back"
                                color: "white" // Initial text color
                                font.pixelSize: 18
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.verticalCenter: parent.verticalCenter
                            }

                            MouseArea {
                                anchors.fill: parent
                                onEntered: {
                                    backButtonBackground.color = "white"; // Change background to white on hover
                                    backButtonText.color = "black"; // Change text color to black on hover
                                }
                                onExited: {
                                    backButtonBackground.color = "#242c4d"; // Revert background color on exit
                                    backButtonText.color = "white"; // Revert text color to white on exit
                                }
                                onClicked: {
                                    backend.getDroneAction("backward"); // Action for the "Back" button
                                }
                            }
                        }
                    }
                }

                // Connect, Down, Takeoff, Land
                RowLayout {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 150
                    //Connect
                    Button {
                        Layout.preferredWidth: 200
                        Layout.preferredHeight: 150

                        // Define the custom background
                        background: Rectangle {
                            id: connectButtonBackground // Unique ID for the Connect button background
                            color: "#242c4d"    // Initial background color
                            border.color: "black" // Border color
                        }

                        // Stack Image and Text on top of each other and center them
                        contentItem: Item {
                            anchors.fill: parent

                            Image {
                                source: "brainwave-prediction-app/images/connect.png"
                                width: 150
                                height: 150
                                anchors.centerIn: parent
                            }

                            Text {
                                id: connectButtonText // Unique ID for the Connect button text
                                text: "Connect"
                                color: "white" // Initial text color
                                font.pixelSize: 18
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.verticalCenter: parent.verticalCenter
                            }

                            MouseArea {
                                anchors.fill: parent
                                onEntered: {
                                    connectButtonBackground.color = "white"; // Change background to white on hover
                                    connectButtonText.color = "black"; // Change text color to black on hover
                                }
                                onExited: {
                                    connectButtonBackground.color = "#242c4d"; // Revert background color on exit
                                    connectButtonText.color = "white"; // Revert text color to white on exit
                                }
                                onClicked: {
                                backend.getDroneAction("connect"); // Action for the "Connect" button
                                }
                            }
                        }
                    }

                    Button {
                        Layout.preferredWidth: 800
                        Layout.preferredHeight: 150

                        // Define the custom background
                        background: Rectangle {
                            id: downButtonBackground // Unique ID for the Down button background
                            color: "#242c4d"    // Initial background color
                            border.color: "black" // Border color
                        }

                        // Stack Image and Text on top of each other and center them
                        contentItem: Item {
                            anchors.fill: parent

                            Image {
                                source: "brainwave-prediction-app/images/down.png"
                                width: 150
                                height: 150
                                anchors.centerIn: parent
                            }

                            Text {
                                id: downButtonText // Unique ID for the Down button text
                                text: "Down"
                                color: "white" // Initial text color
                                font.pixelSize: 18
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.verticalCenter: parent.verticalCenter
                            }

                            MouseArea {
                                anchors.fill: parent
                                onEntered: {
                                    downButtonBackground.color = "white"; // Change background to white on hover
                                    downButtonText.color = "black"; // Change text color to black on hover
                                }
                                onExited: {
                                    downButtonBackground.color = "#242c4d"; // Revert background color on exit
                                    downButtonText.color = "white"; // Revert text color to white on exit
                                }
                                onClicked: {
                                    backend.getDroneAction("down"); // Action for the "Down" button
                                }
                            }
                        }
                    }

                    Button {
                        Layout.preferredWidth: 200
                        Layout.preferredHeight: 150

                        // Define the custom background
                        background: Rectangle {
                            id: takeoffButtonBackground // Unique ID for the Takeoff button background
                            color: "#242c4d"    // Initial background color
                            border.color: "black" // Border color
                        }

                        // Stack Image and Text on top of each other and center them
                        contentItem: Item {
                            anchors.fill: parent

                            Image {
                                source: "brainwave-prediction-app/images/takeoff.png"
                                width: 150
                                height: 150
                                anchors.centerIn: parent
                            }

                            Text {
                                id: takeoffButtonText // Unique ID for the Takeoff button text
                                text: "Takeoff"
                                color: "white" // Initial text color
                                font.pixelSize: 18
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.verticalCenter: parent.verticalCenter
                            }

                            MouseArea {
                                anchors.fill: parent
                                onEntered: {
                                    takeoffButtonBackground.color = "white"; // Change background to white on hover
                                    takeoffButtonText.color = "black"; // Change text color to black on hover
                                }
                                onExited: {
                                    takeoffButtonBackground.color = "#242c4d"; // Revert background color on exit
                                    takeoffButtonText.color = "white"; // Revert text color to white on exit
                                }
                                onClicked: backend.getDroneAction("takeoff")
                            }
                        }
                    }

                    Button {
                        Layout.preferredWidth: 200
                        Layout.preferredHeight: 150

                        // Define the custom background
                        background: Rectangle {
                            id: landButtonBackground // Unique ID for the Land button background
                            color: "#242c4d"    // Initial background color
                            border.color: "black" // Border color
                        }

                        // Stack Image and Text on top of each other and center them
                        contentItem: Item {
                            anchors.fill: parent

                            Image {
                                source: "brainwave-prediction-app/images/land.png"
                                width: 150
                                height: 150
                                anchors.centerIn: parent
                            }

                            Text {
                                id: landButtonText // Unique ID for the Land button text
                                text: "Land"
                                color: "white" // Initial text color
                                font.pixelSize: 18
                                anchors.horizontalCenter: parent.horizontalCenter
                                anchors.verticalCenter: parent.verticalCenter
                            }

                            MouseArea {
                                anchors.fill: parent
                                onEntered: {
                                    landButtonBackground.color = "white"; // Change background to white on hover
                                    landButtonText.color = "black"; // Change text color to black on hover
                                }
                                onExited: {
                                    landButtonBackground.color = "#242c4d"; // Revert background color on exit
                                    landButtonText.color = "white"; // Revert text color to white on exit
                                }
                                onClicked: {
                                    backend.getDroneAction("land"); // Action for the "Land" button
                                }
                            }
                        }
                    }
                }
            }
            function getDroneAction(action) {
                //logModel.append({ action: action + " button pressed" })
                // Here you would implement the actual drone control logic
                console.log(action + " triggered.")
            }
            // Brainwave Visualization

            Rectangle {
                color: "#2b3a4a"
                Layout.fillWidth: true
                Layout.fillHeight: true

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10

                    Text {
                        text: "Brainwave Visualization"
                        font.bold: true
                        font.pixelSize: 20
                        color: "white"
                        Layout.alignment: Qt.AlignHCenter
                    }

                    // Grid Layout for 6 Graphs (2 Rows x 3 Columns)
                    GridLayout {
                        columns: 3
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        columnSpacing: 10
                        rowSpacing: 10

                        Repeater {
                            model: imageModel  
                            delegate: Rectangle {
                                color: "black"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                border.color: "#3b4b57"

                                ColumnLayout {
                                    width: parent.width
                                    height: parent.height
                                    spacing: 5

                                    // Graph Title
                                    Text {
                                        text: model.graphTitle
                                        color: "white"
                                        font.bold: true
                                        Layout.alignment: Qt.AlignHCenter | Qt.AlignTop
                                        Layout.topMargin: 10
                                    }

                                    // Display Image
                                    Image {
                                        source: model.imagePath
                                        Layout.preferredWidth: parent.width * 0.9
                                        Layout.preferredHeight: parent.height * 0.8
                                        Layout.alignment: Qt.AlignHCenter
                                        fillMode: Image.PreserveAspectFit
                                    }
                                }
                            }
                        }
                    }
                }
            }
            //File shuffler view 
            Rectangle {
                id: fileShufflerView
                color: "#2b3a4a"
                Layout.fillWidth: true
                Layout.fillHeight: true

                property string outputBoxText: ""
                property string selectedDirectory: ""
                property bool ranShuffle: false 

                Column {
                    anchors.fill: parent
                    spacing: 10

                    Text {
                        text: "File Shuffler"
                        color: "white"
                        font.bold: true
                        font.pixelSize: 24
                        horizontalAlignment: Text.AlignHCenter
                        anchors.horizontalCenter: parent.horizontalCenter
                        anchors.top: parent.top
                        anchors.topMargin: 20
                    }

                    Rectangle {
                        width: parent.width * 0.6
                        height: parent.height * 0.6
                        color: "lightgrey"
                        anchors.horizontalCenter: parent.horizontalCenter
                        anchors.verticalCenter: parent.verticalCenter

                        ScrollView {
                            anchors.fill: parent

                            TextArea {
                                id: outputBox
                                text: fileShufflerView.outputBoxText
                                color: "black"
                                readOnly: true
                                width: parent.width
                            }
                        }
                    }

                    Row {
                        id: buttonRow
                        spacing: 20
                        anchors.horizontalCenter: parent.horizontalCenter
                        anchors.top: parent.verticalCenter
                        anchors.topMargin: parent.height * 0.3 + 10

                        Button {
                            id: folderButton
                            text: "Select your Directory"
                            onClicked: folderDialog.open()
                        }

                        Button {
                            id: runButton
                            text: "Run File Shuffler"
                            onClicked: {
                                fileShufflerView.ranShuffle = true; 
                                fileShufflerView.outputBoxText = `Running File Shuffler...\n`;
                                var output = fileShufflerGui.run_file_shuffler_program(fileShufflerView.selectedDirectory);
                                fileShufflerView.outputBoxText += output;
                            }
                        }
                    }

                    Text {
                        id: ranText
                        text: "Shuffle Complete!"
                        color: "yellow"
                        font.bold: true
                        font.pixelSize: 18
                        anchors.horizontalCenter: parent.horizontalCenter
                        anchors.top: buttonRow.bottom 
                        anchors.topMargin: 10 
                        visible: fileShufflerView.ranShuffle 
                    }
                }

                FolderDialog {
                    id: folderDialog
                    title: "Select Your Directory"
                    onAccepted:
                    {
                        let cleanedDirectory = String(folderDialog.selectedFolder);
                        cleanedDirectory = cleanedDirectory.replace("file:///", "");
                        fileShufflerView.selectedDirectory = cleanedDirectory;
                        fileShufflerView.outputBoxText += "Selected directory: " + fileShufflerView.selectedDirectory + "\n";
                    }
                }
            }
        }
    }
}