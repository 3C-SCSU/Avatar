import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15

Rectangle {
    id: root
    color: "#718399"
    Layout.fillWidth: true
    Layout.fillHeight: true
    // Function to log messages to console
    function logToConsole(message) {
        var timestamp = Qt.formatDateTime(new Date(), "dddd, MMMM d, yyyy h:mm:ss AP")
        var logMessage = message + " at " + timestamp
        if (consoleLog.text === "Console output will appear here...") {
            consoleLog.text = logMessage
        } else {
            consoleLog.text = consoleLog.text + "\n" + logMessage
        }
    }
    // Mode and model selection
    property string mode: "Deploy"            // "Deploy" (default) | "Train"
    property string currentModel: "Random Forest"   // "Random Forest" | "Deep Learning"

    // Deep Learning params (UI state only for now)
    property real   learningRate: 0.001
    property int    batchSize: 64
    property int    epochs: 10
    property string optimizer: "Adam"
    property string activation: "ReLU"
    property real   dropout: 0.2
    property string regChoice: "L2"
    property real   momentum: 0.9

    // Random Forest params (UI state only for now)
    property int    rfNEstimators: 100
    property string rfMaxDepth: ""            // empty means None
    property int    rfMinSamplesSplit: 2
    property int    rfMinSamplesLeaf: 1
    property string rfMaxFeatures: "auto"     // auto|sqrt|log2
    property string rfCriterion: "gini"       // gini|entropy

    // Derived
    property bool paramsEnabled: mode === "Train"
    // Framework selection state
    property string currentFramework: "PyTorch"

    RowLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 16

        // LEFT CARD --------------------------------------------------------
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 12

            // MACHINE LEARNING BOX
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: Math.min(root.height * 0.25, 220)
                color: "#242c4d"
                radius: 6
                border.color: "#1a2035"

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 16

                    // Title
                    Text {
                        text: "Machine Learning"
                        color: "white"
                        font.bold: true
                        font.pixelSize: Math.max(32, root.height * 0.06)
                        horizontalAlignment: Text.AlignHCenter
                        Layout.alignment: Qt.AlignHCenter
                    }

                    // Model row
                    RowLayout {
                        Layout.alignment: Qt.AlignHCenter
                        spacing: 16

                        // Random Forest
                        Rectangle {
                            width: Math.max(200, root.width * 0.22)
                            height: Math.max(70, root.height * 0.09)
                            radius: 8
                            color: "#6eb109"
                            border.color: currentModel === "Random Forest" ? "yellow" : "#5a8c2b"
                            border.width: currentModel === "Random Forest" ? 3 : 1
                            Text {
                                anchors.centerIn: parent
                                text: "Random Forest"
                                color: currentModel === "Random Forest" ? "yellow" : "white"
                                font.bold: true
                                font.pixelSize: Math.max(18, parent.height * 0.28)
                            }
                            MouseArea {
                                anchors.fill: parent
                                onClicked: {
                                    currentModel = "Random Forest"
                                    backend.selectModel("Random Forest")
                                    logToConsole("Model selected: Random Forest")
                                }
                            }
                        }

                        // Deep Learning
                        Rectangle {
                            width: Math.max(200, root.width * 0.22)
                            height: Math.max(70, root.height * 0.09)
                            radius: 8
                            color: "#6eb109"
                            border.color: currentModel === "Deep Learning" ? "yellow" : "#5a8c2b"
                            border.width: currentModel === "Deep Learning" ? 3 : 1
                            Text {
                                anchors.centerIn: parent
                                text: "Deep Learning"
                                color: currentModel === "Deep Learning" ? "yellow" : "white"
                                font.bold: true
                                font.pixelSize: Math.max(18, parent.height * 0.28)
                            }
                            MouseArea {
                                anchors.fill: parent
                                onClicked: {
                                    currentModel = "Deep Learning"
                                    backend.selectModel("Deep Learning")
                                    logToConsole("Model selected: Deep Learning")
                                }
                            }
                        }
                    }
                }
            }

            // FRAMEWORK BOX
            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: Math.min(root.height * 0.25, 220)
                color: "#242c4d"
                radius: 6
                border.color: "#1a2035"

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 16

                    // Framework title
                    Text {
                        text: "Framework"
                        color: "white"
                        font.bold: true
                        font.pixelSize: Math.max(32, root.height * 0.06)
                        horizontalAlignment: Text.AlignHCenter
                        Layout.alignment: Qt.AlignHCenter
                    }

                    // Framework row
                    RowLayout {
                        Layout.alignment: Qt.AlignHCenter
                        spacing: 16

                        // PyTorch
                        Rectangle {
                            width: Math.max(170, root.width * 0.18)
                            height: Math.max(70, root.height * 0.09)
                            radius: 8
                            color: "#6eb109"
                            border.color: currentFramework === "PyTorch" ? "yellow" : "#5a8c2b"
                            border.width: currentFramework === "PyTorch" ? 3 : 1
                            Text {
                                anchors.centerIn: parent
                                text: "PyTorch"
                                color: currentFramework === "PyTorch" ? "yellow" : "white"
                                font.bold: true
                                font.pixelSize: Math.max(18, parent.height * 0.28)
                            }
                            MouseArea {
                                anchors.fill: parent
                                onClicked: {
                                    currentFramework = "PyTorch"
                                    backend.selectFramework("PyTorch")
                                    logToConsole("Model selected: PyTorch")
                                }
                            }
                        }

                        // TensorFlow
                        Rectangle {
                            width: Math.max(170, root.width * 0.18)
                            height: Math.max(70, root.height * 0.09)
                            radius: 8
                            color: "#6eb109"
                            border.color: currentFramework === "TensorFlow" ? "yellow" : "#5a8c2b"
                            border.width: currentFramework === "TensorFlow" ? 3 : 1
                            Text {
                                anchors.centerIn: parent
                                text: "TensorFlow"
                                color: currentFramework === "TensorFlow" ? "yellow" : "white"
                                font.bold: true
                                font.pixelSize: Math.max(18, parent.height * 0.28)
                            }
                            MouseArea {
                                anchors.fill: parent
                                onClicked: {
                                    currentFramework = "TensorFlow"
                                    backend.selectFramework("TensorFlow")
                                    logToConsole("Model selected: TensorFlow")
                                }
                            }
                        }

                        // JAX (UI only; backend logs selection)
                        Rectangle {
                            width: Math.max(170, root.width * 0.18)
                            height: Math.max(70, root.height * 0.09)
                            radius: 8
                            color: "#6eb109"
                            border.color: currentFramework === "JAX" ? "yellow" : "#5a8c2b"
                            border.width: currentFramework === "JAX" ? 3 : 1
                            Text {
                                anchors.centerIn: parent
                                text: "JAX"
                                color: currentFramework === "JAX" ? "yellow" : "white"
                                font.bold: true
                                font.pixelSize: Math.max(18, parent.height * 0.28)
                            }
                            MouseArea {
                                anchors.fill: parent
                                onClicked: {
                                    currentFramework = "JAX"
                                    backend.selectFramework("JAX")
                                    logToConsole("Model selected: JAX")
                                }
                            }
                        }
                    }
                }
            }
                        
            RowLayout {
            Layout.fillWidth: true
            spacing: 24
            // Success Rate section
            ColumnLayout {
                Layout.fillWidth: true
                spacing: 8
            
            
            // Success Rate label
            Text {
                text: "Success Rate"
                color: "white"
                font.bold: true
                font.pixelSize: Math.max(18, root.height * 0.03)
            }

                // Results table (unchanged content)
                Rectangle {
                    id: resultsTable
                    Layout.preferredWidth: Math.min(root.width * 0.55, 520)
                    Layout.preferredHeight: Math.min(root.height * 0.32, 260) // slightly taller
                    color: "#5f6b7a"
                    radius: 6
                    border.color: "#d0d6df"
                    property int contentWidth: width - 32 // account for ColumnLayout margins (16 each side)

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 16 // larger inner padding so grey shows around white cells
                        spacing: 6

                        Row {
                            id: headerRow
                            width: resultsTable.contentWidth
                            spacing: 1
                            Rectangle {
                                color: "white"; height: 28; width: (resultsTable.contentWidth - headerRow.spacing) / 2
                                Text { anchors.centerIn: parent; text: "Class"; color: "black"; font.bold: true }
                            }
                            Rectangle {
                                color: "white"; height: 28; width: (resultsTable.contentWidth - headerRow.spacing) / 2
                                Text { anchors.centerIn: parent; text: "Precision"; color: "black"; font.bold: true }
                            }
                        }

                        ListModel {
                            id: resultsModel
                            ListElement { klass: "Backward"; precision: "1" }
                            ListElement { klass: "Forward"; precision: "0.97" }
                            ListElement { klass: "Left"; precision: "1" }
                            ListElement { klass: "Right"; precision: "0.99" }
                            ListElement { klass: "Land"; precision: "1" }
                            ListElement { klass: "Takeoff"; precision: "1" }
                        }

                        ListView {
                            id: resultsList
                            Layout.fillWidth: false
                            width: resultsTable.contentWidth
                            Layout.fillHeight: true
                            clip: true
                            model: resultsModel
                            delegate: Row {
                                id: resultRow
                                width: resultsTable.contentWidth
                                spacing: 1
                                Rectangle {
                                    color: "white"; height: 28; width: (resultsTable.contentWidth - resultRow.spacing) / 2
                                    Text { anchors.centerIn: parent; text: model.klass; color: "black" }
                                }
                                Rectangle {
                                    color: "white"; height: 28; width: (resultsTable.contentWidth - resultRow.spacing) / 2
                                    Text { anchors.centerIn: parent; text: model.precision; color: "black" }
                                }
                            }
                        }
                    }
                }
            }
                // Console Log
                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    
                    Text {
                        text: "Console Log"
                        color: "white"
                        font.bold: true
                        font.pixelSize: Math.max(18, root.height * 0.03)
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.preferredHeight: Math.min(root.height * 0.25, 200)
                        color: "white"
                        radius: 6
                        border.color: "#d0d6df"

                        ScrollView {
                            anchors.fill: parent
                            anchors.margins: 8
                            ScrollBar.vertical.policy: ScrollBar.AlwaysOn

                            TextArea {
                                id: consoleLog
                                readOnly: true
                                wrapMode: Text.Wrap
                                text: "Console output here..."
                                color: "black"
                                font.pixelSize: 10
                                background: Rectangle {
                                    color: "white"
                                }
                            }
                        }
                    }
                }
                // Vertical buttons column: Deploy (top), Train (below)
                ColumnLayout {
                    spacing: 24
                    Layout.preferredWidth: 160
                    Layout.alignment: Qt.AlignTop
                    Layout.leftMargin: 28 // shift a little more right from the results table

                    // DEPLOY (default)
                    Button {
                        id: deployBtn
                        checkable: true
                        checked: mode === "Deploy"
                        onClicked: {
                          mode = "Deploy"
                          logToConsole("Mode changed: Deploy")
                        }
                        Layout.preferredWidth: 110
                        Layout.preferredHeight: 110
                        padding: 0
                        background: Rectangle {
                            radius: width/2
                            gradient: Gradient {
                                GradientStop { position: 0; color: "#6aa5ff" }
                                GradientStop { position: 1; color: "#2d53cc" }
                            }
                            border.color: deployBtn.checked ? "yellow" : "#102a6b"
                            border.width: 2
                        }
                        contentItem: Text {
                            anchors.centerIn: parent
                            text: "Deploy"
                            color: "yellow"
                            font.bold: true
                            font.pixelSize: 22
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                    }

                    // TRAIN (selects training mode)
                    Button {
                        id: trainBtn
                        checkable: true
                        checked: mode === "Train"
                        onClicked: {
                          mode = "Train"
                          logToConsole("Mode changed: Train")
                        }
                        Layout.preferredWidth: 110
                        Layout.preferredHeight: 110
                        padding: 0
                        background: Rectangle {
                            radius: width/2
                            gradient: Gradient {
                                GradientStop { position: 0; color: "#6aa5ff" }
                                GradientStop { position: 1; color: "#2d53cc" }
                            }
                            border.color: trainBtn.checked ? "yellow" : "#102a6b"
                            border.width: 2
                        }
                        contentItem: Text {
                            anchors.centerIn: parent
                            text: "Train"
                            color: "white"
                            font.bold: true
                            font.pixelSize: 22
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                    }
                }
            }
        }
          
        // RIGHT PANE (Parameters – navy; visible and enabled only in Train) --
        Rectangle {
            Layout.preferredWidth: Math.max(380, root.width * 0.34)
            Layout.fillHeight: true
            radius: 8
            color: "#242c4d"
            border.color: "#1a2035"
            opacity: paramsEnabled ? 1.0 : 0.55
            enabled: paramsEnabled

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 10

                // ON/OFF indicator (auto from mode)
                Rectangle {
                    Layout.alignment: Qt.AlignRight
                    width: 160; height: 60; radius: 8
                    color: paramsEnabled ? "#2f8f2f" : "#c13a2a"
                    border.color: "#202020"
                    Text {
                        anchors.centerIn: parent
                        text: paramsEnabled ? "ON" : "OFF"
                        color: "white"
                        font.bold: true
                        font.pixelSize: 28
                    }
                    // Let user toggle quickly, flipping mode
                    MouseArea { anchors.fill: parent; onClicked: mode = paramsEnabled ? "Deploy" : "Train" }
                }

                Text {
                    text: "Parameters"
                    color: "white"
                    font.bold: true
                    font.pixelSize: Math.max(26, root.height * 0.04)
                }

                // RANDOM FOREST PARAMS --------------------------------------
                ColumnLayout {
                    visible: currentModel === "Random Forest"
                    spacing: 10
                    Layout.fillWidth: true

                    // n_estimators
                    RowLayout {
                        Layout.fillWidth: true; spacing: 8
                        Text { text: "n_estimators:"; color: "white"; Layout.preferredWidth: 160 }
                        TextField {
                            text: rfNEstimators.toString()
                            inputMethodHints: Qt.ImhDigitsOnly
                            onEditingFinished: {
                                var v = parseInt(text); rfNEstimators = isNaN(v) ? rfNEstimators : Math.max(1, v)
                                text = rfNEstimators.toString()
                            }
                            Layout.fillWidth: true
                        }
                    }

                    // max_depth
                    RowLayout {
                        Layout.fillWidth: true; spacing: 8
                        Text { text: "max_depth:"; color: "white"; Layout.preferredWidth: 160 }
                        TextField {
                            placeholderText: "None"
                            text: rfMaxDepth
                            onEditingFinished: rfMaxDepth = text.trim()
                            Layout.fillWidth: true
                        }
                    }

                    // max_features
                    RowLayout {
                        Layout.fillWidth: true; spacing: 8
                        Text { text: "max_features:"; color: "white"; Layout.preferredWidth: 160 }
                        ComboBox {
                            model: ["auto","sqrt","log2"]
                            currentIndex: Math.max(0, model.indexOf(rfMaxFeatures))
                            onCurrentTextChanged: rfMaxFeatures = currentText
                            Layout.fillWidth: true
                        }
                    }

                    // min_samples_split
                    RowLayout {
                        Layout.fillWidth: true; spacing: 8
                        Text { text: "min_samples_split:"; color: "white"; Layout.preferredWidth: 160 }
                        TextField {
                            text: rfMinSamplesSplit.toString()
                            inputMethodHints: Qt.ImhDigitsOnly
                            onEditingFinished: {
                                var v = parseInt(text); rfMinSamplesSplit = isNaN(v) ? rfMinSamplesSplit : Math.max(2, v)
                                text = rfMinSamplesSplit.toString()
                            }
                            Layout.fillWidth: true
                        }
                    }

                    // min_samples_leaf
                    RowLayout {
                        Layout.fillWidth: true; spacing: 8
                        Text { text: "min_samples_leaf:"; color: "white"; Layout.preferredWidth: 160 }
                        TextField {
                            text: rfMinSamplesLeaf.toString()
                            inputMethodHints: Qt.ImhDigitsOnly
                            onEditingFinished: {
                                var v = parseInt(text); rfMinSamplesLeaf = isNaN(v) ? rfMinSamplesLeaf : Math.max(1, v)
                                text = rfMinSamplesLeaf.toString()
                            }
                            Layout.fillWidth: true
                        }
                    }

                    // criterion
                    RowLayout {
                        Layout.fillWidth: true; spacing: 8
                        Text { text: "criterion:"; color: "white"; Layout.preferredWidth: 160 }
                        ComboBox {
                            model: ["gini","entropy"]
                            currentIndex: Math.max(0, model.indexOf(rfCriterion))
                            onCurrentTextChanged: rfCriterion = currentText
                            Layout.fillWidth: true
                        }
                    }
                }

                // DEEP LEARNING PARAMS --------------------------------------
                ColumnLayout {
                    visible: currentModel === "Deep Learning"
                    spacing: 10
                    Layout.fillWidth: true

                    // Learning rate
                    RowLayout {
                        Layout.fillWidth: true; spacing: 8
                        Text { text: "Learning rate:"; color: "white"; Layout.preferredWidth: 150 }
                        Slider {
                            Layout.fillWidth: true
                            from: 0.0001; to: 0.1; stepSize: 0.0001
                            value: learningRate
                            onValueChanged: learningRate = value
                        }
                        TextField {
                            text: Number(learningRate).toFixed(4)
                            onEditingFinished: {
                                var v = parseFloat(text); if (!isNaN(v)) learningRate = Math.max(0.0001, Math.min(0.1, v))
                                text = Number(learningRate).toFixed(4)
                            }
                            Layout.preferredWidth: 90
                        }
                    }

                    // Batch size
                    RowLayout {
                        Layout.fillWidth: true; spacing: 8
                        Text { text: "Batch size:"; color: "white"; Layout.preferredWidth: 150 }
                        Slider {
                            Layout.fillWidth: true
                            from: 8; to: 256; stepSize: 8
                            value: batchSize
                            onValueChanged: batchSize = Math.round(value)
                        }
                        TextField {
                            text: batchSize.toString()
                            inputMethodHints: Qt.ImhDigitsOnly
                            onEditingFinished: {
                                var v = parseInt(text); if (!isNaN(v)) batchSize = Math.max(8, Math.min(256, v))
                                text = batchSize.toString()
                            }
                            Layout.preferredWidth: 90
                        }
                    }

                    // Epoch #
                    RowLayout {
                        Layout.fillWidth: true; spacing: 8
                        Text { text: "Epoch #:"; color: "white"; Layout.preferredWidth: 150 }
                        Slider {
                            Layout.fillWidth: true
                            from: 1; to: 200; stepSize: 1
                            value: epochs
                            onValueChanged: epochs = Math.round(value)
                        }
                        TextField {
                            text: epochs.toString()
                            inputMethodHints: Qt.ImhDigitsOnly
                            onEditingFinished: {
                                var v = parseInt(text); if (!isNaN(v)) epochs = Math.max(1, Math.min(200, v))
                                text = epochs.toString()
                            }
                            Layout.preferredWidth: 90
                        }
                    }

                    // Optimizer
                    RowLayout {
                        Layout.fillWidth: true; spacing: 8
                        Text { text: "Optimizer:"; color: "white"; Layout.preferredWidth: 150 }
                        ComboBox {
                            Layout.fillWidth: true
                            model: ["Adam", "SGD", "RMSProp", "AdamW"]
                            currentIndex: Math.max(0, model.indexOf(optimizer))
                            onCurrentTextChanged: optimizer = currentText
                        }
                    }

                    // Activation
                    RowLayout {
                        Layout.fillWidth: true; spacing: 8
                        Text { text: "Activation fn:"; color: "white"; Layout.preferredWidth: 150 }
                        ComboBox {
                            Layout.fillWidth: true
                            model: ["ReLU", "LeakyReLU", "Tanh", "Sigmoid", "GELU"]
                            currentIndex: Math.max(0, model.indexOf(activation))
                            onCurrentTextChanged: activation = currentText
                        }
                    }

                    // Dropout
                    RowLayout {
                        Layout.fillWidth: true; spacing: 8
                        Text { text: "Drop out rate:"; color: "white"; Layout.preferredWidth: 150 }
                        Slider {
                            Layout.fillWidth: true
                            from: 0; to: 0.9; stepSize: 0.01
                            value: dropout
                            onValueChanged: dropout = value
                        }
                        TextField {
                            text: Number(dropout).toFixed(2)
                            onEditingFinished: {
                                var v = parseFloat(text); if (!isNaN(v)) dropout = Math.max(0, Math.min(0.9, v))
                                text = Number(dropout).toFixed(2)
                            }
                            Layout.preferredWidth: 90
                        }
                    }

                    // L1/L2
                    RowLayout {
                        Layout.fillWidth: true; spacing: 8
                        Text { text: "L1/L2 choice:"; color: "white"; Layout.preferredWidth: 150 }
                        ComboBox {
                            Layout.preferredWidth: 100
                            model: ["L1","L2"]
                            currentIndex: regChoice === "L1" ? 0 : 1
                            onCurrentTextChanged: regChoice = currentText
                        }
                    }

                    // Momentum
                    RowLayout {
                        Layout.fillWidth: true; spacing: 8
                        Text { text: "Momentum:"; color: "white"; Layout.preferredWidth: 150 }
                        Slider {
                            Layout.fillWidth: true
                            from: 0; to: 0.99; stepSize: 0.01
                            value: momentum
                            onValueChanged: momentum = value
                        }
                        TextField {
                            text: Number(momentum).toFixed(2)
                            onEditingFinished: {
                                var v = parseFloat(text); if (!isNaN(v)) momentum = Math.max(0, Math.min(0.99, v))
                                text = Number(momentum).toFixed(2)
                            }
                            Layout.preferredWidth: 90
                        }
                    }
                }

                Item { Layout.fillHeight: true } // spacer
            }
        }
    }
}
