import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs


ApplicationWindow {
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
            Rectangle {
                color: "lightgrey"
                Layout.fillWidth: true
                Layout.fillHeight: true
                Text {
                    anchors.centerIn: parent
                    text: "Manual Drone Control View"
                }
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
