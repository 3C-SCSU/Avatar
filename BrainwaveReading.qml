import QtQuick.Dialogs
import Qt.labs.platform
import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtQuick3D 6.7

// Brainwave Reading view
Rectangle {
    property string selectedModel: "Random Forest"
    property string currentFramework: "PyTorch"
    color: "#718399"
    Layout.fillWidth: true
    Layout.fillHeight: true

    Row {
        anchors.fill: parent
        spacing: width * 0.02

        // Left Column
        Column {
            width: parent.width * 0.5
            height: parent.height
            spacing: height * 0.02
            anchors.left: parent.left

            // Control Mode
            Row {
                width: parent.width * 0.4
                spacing: parent.width * 0.02
                anchors.horizontalCenter: parent.horizontalCenter

                RadioButton {
                    id: manualControl
                    text: "Manual Control"
                    checked: true
                    font.pixelSize: parent.width * 0.05

                    contentItem: Text {
                        text: manualControl.text
                        color: "white"
                        font.pixelSize: manualControl.font.pixelSize
                        font.bold: true
                        verticalAlignment: Text.AlignVCenter
                        leftPadding: manualControl.indicator.width + manualControl.spacing
                    }
                }

                RadioButton {
                    id: autopilot
                    text: "Autopilot"
                    font.pixelSize: parent.width * 0.05
                    contentItem: Text {
                        text: autopilot.text
                        color: "white"
                        font.pixelSize: autopilot.font.pixelSize
                        font.bold: true
                        verticalAlignment: Text.AlignVCenter
                        leftPadding: autopilot.indicator.width + autopilot.spacing
                    }
                }
            }

            // Brainwave Image with Transparent Button
            Rectangle {
                width: parent.width * 0.25
                height: parent.height * 0.2
                color: "#242c4d"
                anchors.horizontalCenter: parent.horizontalCenter

                Image {
                    source: "GUI_Pics/brain.png"
                    anchors.fill: parent
                    fillMode: Image.PreserveAspectFit
                }

                Button {
                    anchors.fill: parent
                    background: Item {
                    }
                    contentItem: Text {
                        text: "Read my mind..."
                        font.pixelSize: parent.width * 0.1
                        color: "white"
                        anchors.centerIn: parent
                    }
                    onClicked: backend.readMyMind()
                }
            }

            // Model Prediction Section
            Label {
                text: "The model says ..."
                color: "white"
                font.pixelSize: parent.width * 0.03
                anchors.horizontalCenter: parent.horizontalCenter
            }

            GroupBox {
                width: parent.width * 0.4
                height: parent.height * 0.15
                anchors.horizontalCenter: parent.horizontalCenter

                Row {
                    width: parent.width
                    height: parent.height * 0.28
                    spacing: parent.width * 0.01
                    Rectangle {
                        color: "white"
                        width: parent.width * 0.5
                        height: parent.height
                        Text {
                            text: "Count"
                            font.bold: true
                            font.pixelSize: parent.width * 0.09
                            color: "black"
                            anchors.centerIn: parent
                        }
                    }
                    Rectangle {
                        color: "white"
                        width: parent.width * 0.5
                        height: parent.height
                        Text {
                            text: "Label"
                            font.bold: true
                            font.pixelSize: parent.width * 0.09
                            color: "black"
                            anchors.centerIn: parent
                        }
                    }
                }

                ListView {
                    id: predictionListView
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    model: ListModel {
                    }
                    delegate: RowLayout {
                        spacing: 150
                        Text { 
                            text: model.count; font.bold: true; color: "white"; width: 80 
                        }
                        Text { 
                            text: model.label; font.bold: true; color: "white"; width: 80 
                        }
                    }
                }
            }

            // Action Buttons
            Row {
                width: parent.width * 0.6
                height: parent.height * 0.08
                spacing: parent.width * 0.02
                anchors.horizontalCenter: parent.horizontalCenter
                Button {
                    id: notThinking
                    text: "Not what I was thinking..."
                    font.pixelSize: parent.width * 0.03
                    width: (parent.width - parent.spacing) * 0.5
                    height: parent.height
                    background: Rectangle { 
                        color: "#242c4d" 
                    }
                    contentItem: Text {
                        text: notThinking.text
                        color: "white"
                        font.pixelSize: notThinking.font.pixelSize
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                        elide: Text.ElideRight
                    }
                    onClicked: backend.notWhatIWasThinking(manualInput.text)
                }

                Button {
                    id: executeBtn
                    text: "Action"
                    font.pixelSize: parent.width * 0.03
                    width: (parent.width - parent.spacing) * 0.5
                    height: parent.height
                    background: Rectangle { 
                        color: "#242c4d" 
                    }
                    contentItem: Text {
                        text: executeBtn.text
                        color: "white"
                        font.pixelSize: executeBtn.font.pixelSize
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                        elide: Text.ElideRight
                    }
                    onClicked: backend.executeAction()
                }
            }

            // Manual Input and Keep Alive
            Row {
                width: parent.width * .8
                height: parent.height * 0.03
                spacing: parent.width * 0.01
                anchors.horizontalCenter: parent.horizontalCenter
                TextField {
                    id: manualInput
                    placeholderText: "Manual Command"
                    font.pixelSize: parent.width * 0.03
                    width: parent.width * 0.6
                    height: parent.height
                }
                Button {
                    width: parent.width * 0.3
                    height: parent.height
                    background: Rectangle { 
                        color: "#242c4d" 
                    }

                    contentItem: Text {
                        text: qsTr("Run")
                        color: "white"
                        font.pixelSize: parent.height * 0.5
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                        anchors.fill: parent
                    }
                    onClicked: {
                        backend.keepDroneAlive(text=manualInput.text)
                        manualInput.text = ""
                    }
                }
            }

            // Flight Log
            GroupBox {
                title: "Flight Log"
                width: parent.width * 0.5
                height: parent.height * 0.2
                anchors.horizontalCenter: parent.horizontalCenter
                label: Text { 
                    text: qsTr("Flight Log"); font.bold: true; color: "white" 
                }

                Rectangle {
                    anchors.fill: parent
                    color: "white"
                    ListView {
                        id: flightLogView
                        anchors.fill: parent
                        model: ListModel {
                        }
                        delegate: Text {
                            text: log
                            font.pixelSize: parent.width * 0.03
                            font.bold: true
                            color: "black"
                        }
                    }
                }
            }

            Connections {
                target: backend
                function onImagesReady(imageData) {
                    imageModel.clear();
                    for (let item of imageData) {
                        imageModel.append(item);
                    }
                }
                function onLogMessage(message) {
                    var timestamp = new Date().toLocaleString()
                    consoleLog.append(message + " at " + timestamp)
                }
            }
        }

        // Right Column (Prediction Table and Console Log)
        Column {
            width: parent.width * 0.45
            height: parent.height
            spacing: parent.height * 0.02
            anchors.right: parent.right

            // Predictions Table
            GroupBox {
                title: "Predictions Table"
                width: parent.width * 0.9
                height: parent.height * 0.35

                label: Text {
                    text: qsTr("Predictions Table")
                    color: "white"
                    font.pixelSize: parent.width * 0.03
                    font.bold: true
                    anchors.left: parent.left
                    anchors.leftMargin: 10
                    anchors.top: parent.top
                    anchors.topMargin: 5
                }

                Row {
                    width: parent.width
                    height: parent.height * 0.1
                    spacing: parent.width * 0.001
                    Rectangle {
                        color: "white"
                        width: parent.width * 0.33
                        height: parent.height
                        Text {
                            text: "Predictions Count"
                            font.bold: true
                            font.pixelSize: parent.width * 0.09
                            color: "black"
                            anchors.centerIn: parent
                        }
                    }
                    Rectangle {
                        color: "white"
                        width: parent.width * 0.33
                        height: parent.height
                        Text {
                            text: "Server Predictions"
                            font.bold: true
                            font.pixelSize: parent.width * 0.09
                            color: "black"
                            anchors.centerIn: parent
                        }
                    }
                    Rectangle {
                        color: "white"
                        width: parent.width * 0.33
                        height: parent.height
                        Text {
                            text: "Prediction Label"
                            font.bold: true
                            font.pixelSize: parent.width * 0.09
                            color: "black"
                            anchors.centerIn: parent
                        }
                    }
                }

                ListView {
                    Layout.preferredWidth: 700
                    Layout.preferredHeight: 550
                    model: ListModel {
                        ListElement { 
                            count: "1"; server: "Prediction A"; label: "Label A" 
                        }
                        ListElement { 
                            count: "2"; server: "Prediction B"; label: "Label B" 
                        }
                    }

                    delegate: Rectangle {
                        width: parent.width
                        height: 40
                        color: "white"

                        RowLayout {
                            anchors.fill: parent
                            spacing: 50
                            Text { 
                                text: model.count; font.bold: true; color: "black"; width: 120 
                            }
                            Text { 
                                text: model.server; font.bold: true; color: "black"; width: 200 
                            }
                            Text { 
                                text: model.label; font.bold: true; color: "black"; width: 120 
                            }
                        }
                    }
                }
            }

            // Console Log
            GroupBox {
                title: "Console Log"
                width: parent.width * 0.9
                height: parent.height * 0.25
                label: Text { 
                    text: qsTr("Console Log"); 
                    font.bold: true; 
                    color: "white" 
                }

                ScrollView {
                    anchors.fill: parent
                    clip: true

                    TextArea {
                        id: consoleLog
                        wrapMode: Text.WrapAnywhere
                        readOnly: true
                        font.pixelSize: parent.width * 0.025
                        color: "black"
                        background: Rectangle { color: "white" }
                    }
                }
            }

            // Model Selection Buttons in dark box
            Rectangle {
                width: parent.width * 0.9
                height: parent.height * 0.18
                color: "#2c3e50"
                
                Column {
                    anchors.fill: parent
                    anchors.margins: parent.height * 0.05
                    spacing: parent.height * 0.1
                    
                    // Model Selection Row
                    Row {
                        width: parent.width
                        height: (parent.height - parent.spacing) * 0.5
                        spacing: parent.width * 0.02
                        
                        Rectangle {
                            width: (parent.width - parent.spacing * 2) / 3
                            height: parent.height
                            color: "#2d7a4a"
                            radius: 5
                            border.color: selectedModel === "Random Forest" ? "yellow" : "#4a9d6f"
                            border.width: selectedModel === "Random Forest" ? 3 : 1
                            
                            Text {
                                text: "Random Forest"
                                font.pixelSize: parent.height * 0.25
                                font.bold: true
                                color: selectedModel === "Random Forest" ? "yellow" : "white"
                                anchors.centerIn: parent
                                horizontalAlignment: Text.AlignHCenter
                            }
                            
                            MouseArea {
                                anchors.fill: parent
                                onClicked: {
                                    selectedModel = "Random Forest"
                                    backend.selectModel("Random Forest")
                                }
                            }
                        }
                        
                        Rectangle {
                            width: (parent.width - parent.spacing * 2) / 3
                            height: parent.height
                            color: "#2d7a4a"
                            radius: 5
                            border.color: selectedModel === "GaussianNB" ? "yellow" : "#4a9d6f"
                            border.width: selectedModel === "GaussianNB" ? 3 : 1
                            
                            Text {
                                text: "GaussianNB"
                                font.pixelSize: parent.height * 0.25
                                font.bold: true
                                color: selectedModel === "GaussianNB" ? "yellow" : "white"
                                anchors.centerIn: parent
                                horizontalAlignment: Text.AlignHCenter
                            }
                            
                            MouseArea {
                                anchors.fill: parent
                                onClicked: {
                                    selectedModel = "GaussianNB"
                                    backend.selectModel("GaussianNB")
                                }
                            }
                        }
                        
                        Rectangle {
                            width: (parent.width - parent.spacing * 2) / 3
                            height: parent.height
                            color: "#2d7a4a"
                            radius: 5
                            border.color: selectedModel === "Deep Learning" ? "yellow" : "#4a9d6f"
                            border.width: selectedModel === "Deep Learning" ? 3 : 1
                            
                            Text {
                                text: "Deep Learning"
                                font.pixelSize: parent.height * 0.25
                                font.bold: true
                                color: selectedModel === "Deep Learning" ? "yellow" : "white"
                                anchors.centerIn: parent
                                horizontalAlignment: Text.AlignHCenter
                            }
                            
                            MouseArea {
                                anchors.fill: parent
                                onClicked: {
                                    selectedModel = "Deep Learning"
                                    backend.selectModel("Deep Learning")
                                }
                            }
                        }
                    }
                    
                    // Framework Selection Row
                    Row {
                        width: parent.width
                        height: (parent.height - parent.spacing) * 0.5
                        spacing: parent.width * 0.02
                        
                        Rectangle {
                            width: (parent.width - parent.spacing * 2) / 3
                            height: parent.height
                            color: "#2d7a4a"
                            radius: 5
                            border.color: currentFramework === "PyTorch" ? "yellow" : "#4a9d6f"
                            border.width: currentFramework === "PyTorch" ? 3 : 1
                            
                            Text {
                                text: "PyTorch"
                                font.pixelSize: parent.height * 0.3
                                font.bold: true
                                color: currentFramework === "PyTorch" ? "yellow" : "white"
                                anchors.centerIn: parent
                            }

                            MouseArea {
                                anchors.fill: parent
                                onClicked: {
                                    currentFramework = "PyTorch"
                                    backend.selectFramework("PyTorch")
                                }
                            }
                        }

                        Rectangle {
                            width: (parent.width - parent.spacing * 2) / 3
                            height: parent.height
                            color: "#2d7a4a"
                            radius: 5
                            border.color: currentFramework === "TensorFlow" ? "yellow" : "#4a9d6f"
                            border.width: currentFramework === "TensorFlow" ? 3 : 1
                            
                            Text {
                                text: "TensorFlow"
                                font.pixelSize: parent.height * 0.3
                                font.bold: true
                                color: currentFramework === "TensorFlow" ? "yellow" : "white"
                                anchors.centerIn: parent
                            }

                            MouseArea {
                                anchors.fill: parent
                                onClicked: {
                                    currentFramework = "TensorFlow"
                                    backend.selectFramework("TensorFlow")
                                }
                            }
                        }
                        
                        Rectangle {
                            width: (parent.width - parent.spacing * 2) / 3
                            height: parent.height
                            color: "#2d7a4a"
                            radius: 5
                            border.color: currentFramework === "JAX" ? "yellow" : "#4a9d6f"
                            border.width: currentFramework === "JAX" ? 3 : 1
                            
                            Text {
                                text: "JAX"
                                font.pixelSize: parent.height * 0.3
                                font.bold: true
                                color: currentFramework === "JAX" ? "yellow" : "white"
                                anchors.centerIn: parent
                            }

                            MouseArea {
                                anchors.fill: parent
                                onClicked: {
                                    currentFramework = "JAX"
                                    backend.selectFramework("JAX")
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Connect Button - Fixed to bottom left
    Rectangle {
        width: parent.width * 0.08
        height: parent.height * 0.12
        color: "#242c4d"
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.leftMargin: parent.width * 0.01
        anchors.bottomMargin: parent.height * 0.01

        Image {
            source: "GUI_Pics/connect.png"
            anchors.fill: parent
            fillMode: Image.PreserveAspectFit
        }

        Button {
            anchors.fill: parent
            background: Item {
            }
            contentItem: Text {
                text: "Connect"
                font.pixelSize: parent.width * 0.15
                color: "white"
                anchors.centerIn: parent
            }
            onClicked: backend.connectDrone()
        }
    }

    // Radio Buttons - Fixed to bottom left, next to connect
    Row {
        width: parent.width * 0.25
        height: parent.height * 0.06
        spacing: width * 0.02
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.leftMargin: parent.width * 0.1
        anchors.bottomMargin: parent.height * 0.03

        RadioButton {
            id: liveRadio
            text: "Live Data"
            font.pixelSize: parent.width * 0.04
            font.bold: true
            checked: true
            contentItem: Text {
                text: liveRadio.text
                color: "white"
                font.pixelSize: liveRadio.font.pixelSize
                font.bold: liveRadio.font.bold
                verticalAlignment: Text.AlignVCenter
                leftPadding: liveRadio.indicator.width + liveRadio.spacing
            }
            onClicked: backend.setDataMode("live")
        }

        RadioButton {
            id: syntheticRadio
            text: "Synthetic Data"
            font.pixelSize: parent.width * 0.04
            font.bold: true
            checked: false
            contentItem: Text {
                text: syntheticRadio.text
                color: "white"
                font.pixelSize: syntheticRadio.font.pixelSize
                font.bold: syntheticRadio.font.bold
                verticalAlignment: Text.AlignVCenter
                leftPadding: syntheticRadio.indicator.width + syntheticRadio.spacing
            }
            onClicked: backend.setDataMode("synthetic")
        }
    }
}
