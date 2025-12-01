import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15

Rectangle {
    id: root
    color: "#718399"
    Layout.fillWidth: true
    Layout.fillHeight: true

    // Global minima so UI can collapse uniformly
    property int minControlSize: 1

    // Function to log messages to console
    function logToConsole(message) {
        var timestamp = Qt.formatDateTime(new Date(), "dddd, MMMM d, yyyy h:mm:ss AP")
        var logMessage = message + " at " + timestamp
        if (consoleLog.text === "Console output will appear here..." ||
            consoleLog.text === "Console output here...") {
            consoleLog.text = logMessage
        } else {
            consoleLog.text = consoleLog.text + "\n" + logMessage
        }
    }

    // Mode and model selection
    property string mode: "Deploy"                 // "Deploy" (default) | "Train"
    property string currentModel: "Random Forest"  // "Random Forest" | "GaussianNB" | "Deep Learning"

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

    // TOP-LEVEL LAYOUT
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 16

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 16

            // LEFT SIDE ----------------------------------------------------
            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 12

                // MACHINE LEARNING BOX -------------------------------------
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: Math.min(root.height * 0.25, 220)
                    Layout.minimumHeight: root.minControlSize
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
                            font.pixelSize: Math.max(10, Math.min(32, root.height * 0.05))
                            horizontalAlignment: Text.AlignHCenter
                            Layout.alignment: Qt.AlignHCenter
                        }

                        // Model row (buttons shrink with row)
                        RowLayout {
                            Layout.fillWidth: true
                            Layout.alignment: Qt.AlignHCenter
                            spacing: 16

                            // Random Forest
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.minimumWidth: root.minControlSize
                                Layout.minimumHeight: root.minControlSize
                                Layout.maximumWidth: 260
                                implicitWidth: Math.max(root.minControlSize,
                                                        Math.min(root.width * 0.16, 260))
                                implicitHeight: Math.max(root.minControlSize,
                                                         Math.min(root.height * 0.08, 90))
                                radius: 8
                                color: "#2d7a4a"
                                border.color: currentModel === "Random Forest" ? "yellow" : "#2d7a4a"
                                border.width: currentModel === "Random Forest" ? 3 : 1

                                Text {
                                    anchors.centerIn: parent
                                    text: "Random Forest"
                                    color: currentModel === "Random Forest" ? "yellow" : "white"
                                    font.bold: true
                                    font.pixelSize: Math.max(10,
                                                             Math.min(20, parent.height * 0.35))
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                    wrapMode: Text.NoWrap
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
                            // GaussianNB
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.minimumWidth: root.minControlSize
                                Layout.minimumHeight: root.minControlSize
                                Layout.maximumWidth: 260
                                implicitWidth: Math.max(root.minControlSize,
                                                        Math.min(root.width * 0.16, 260))
                                implicitHeight: Math.max(root.minControlSize,
                                                         Math.min(root.height * 0.08, 90))
                                radius: 8
                                color: "#2d7a4a"
                                border.color: currentModel === "GaussianNB" ? "#439566" : "#2d7a4a"
                                border.width: currentModel === "GaussianNB" ? 3 : 1

                                Text {
                                    anchors.centerIn: parent
                                    text: "GaussianNB"
                                    color: currentModel === "GaussianNB" ? "yellow" : "white"
                                    font.bold: true
                                    font.pixelSize: Math.max(10,
                                                             Math.min(20, parent.height * 0.35))
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                    wrapMode: Text.NoWrap
                                }

                                MouseArea {
                                    anchors.fill: parent
                                    onClicked: {
                                        currentModel = "GaussianNB"
                                        backend.selectModel("GaussianNB")
                                        logToConsole("Model selected: GaussianNB")
                                    }
                                }
                            }

                            // Deep Learning
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.minimumWidth: root.minControlSize
                                Layout.minimumHeight: root.minControlSize
                                Layout.maximumWidth: 260
                                implicitWidth: Math.max(root.minControlSize,
                                                        Math.min(root.width * 0.16, 260))
                                implicitHeight: Math.max(root.minControlSize,
                                                         Math.min(root.height * 0.08, 90))
                                radius: 8
                                color: "#2d7a4a"
                                border.color: currentModel === "Deep Learning" ? "yellow" : "#2d7a4a"
                                border.width: currentModel === "Deep Learning" ? 3 : 1

                                Text {
                                    anchors.centerIn: parent
                                    text: "Deep Learning"
                                    color: currentModel === "Deep Learning" ? "yellow" : "white"
                                    font.bold: true
                                    font.pixelSize: Math.max(10,
                                                             Math.min(20, parent.height * 0.35))
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                    wrapMode: Text.NoWrap
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

                // FRAMEWORK BOX ---------------------------------------------
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: Math.min(root.height * 0.25, 220)
                    Layout.minimumHeight: root.minControlSize
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
                            font.pixelSize: Math.max(10, Math.min(32, root.height * 0.05))
                            horizontalAlignment: Text.AlignHCenter
                            Layout.alignment: Qt.AlignHCenter
                        }

                        // Framework row
                        RowLayout {
                            Layout.fillWidth: true
                            Layout.alignment: Qt.AlignHCenter
                            spacing: 16

                            // PyTorch
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.minimumWidth: root.minControlSize
                                Layout.minimumHeight: root.minControlSize
                                Layout.maximumWidth: 260
                                implicitWidth: Math.max(root.minControlSize,
                                                        Math.min(root.width * 0.12, 220))
                                implicitHeight: Math.max(root.minControlSize,
                                                         Math.min(root.height * 0.07, 80))
                                radius: 8
                                color: "#2d7a4a"
                                border.color: currentFramework === "PyTorch" ? "yellow" : "#2d7a4a"
                                border.width: currentFramework === "PyTorch" ? 3 : 1

                                Text {
                                    anchors.centerIn: parent
                                    text: "PyTorch"
                                    color: currentFramework === "PyTorch" ? "yellow" : "white"
                                    font.bold: true
                                    font.pixelSize: Math.max(10,
                                                             Math.min(18, parent.height * 0.35))
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                    wrapMode: Text.NoWrap
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
                                Layout.fillWidth: true
                                Layout.minimumWidth: root.minControlSize
                                Layout.minimumHeight: root.minControlSize
                                Layout.maximumWidth: 260
                                implicitWidth: Math.max(root.minControlSize,
                                                        Math.min(root.width * 0.12, 220))
                                implicitHeight: Math.max(root.minControlSize,
                                                         Math.min(root.height * 0.07, 80))
                                radius: 8
                                color: "#2d7a4a"
                                border.color: currentFramework === "TensorFlow" ? "yellow" : "#2d7a4a"
                                border.width: currentFramework === "TensorFlow" ? 3 : 1

                                Text {
                                    anchors.centerIn: parent
                                    text: "TensorFlow"
                                    color: currentFramework === "TensorFlow" ? "yellow" : "white"
                                    font.bold: true
                                    font.pixelSize: Math.max(10,
                                                             Math.min(18, parent.height * 0.35))
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                    wrapMode: Text.NoWrap
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

                            // JAX
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.minimumWidth: root.minControlSize
                                Layout.minimumHeight: root.minControlSize
                                Layout.maximumWidth: 260
                                implicitWidth: Math.max(root.minControlSize,
                                                        Math.min(root.width * 0.10, 180))
                                implicitHeight: Math.max(root.minControlSize,
                                                         Math.min(root.height * 0.07, 80))
                                radius: 8
                                color: "#2d7a4a"
                                border.color: currentFramework === "JAX" ? "yellow" : "#2d7a4a"
                                border.width: currentFramework === "JAX" ? 3 : 1

                                Text {
                                    anchors.centerIn: parent
                                    text: "JAX"
                                    color: currentFramework === "JAX" ? "yellow" : "white"
                                    font.bold: true
                                    font.pixelSize: Math.max(10,
                                                             Math.min(18, parent.height * 0.35))
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                    wrapMode: Text.NoWrap
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

                // METRICS + CONSOLE + MODE BUTTONS --------------------------
                RowLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    spacing: 6

                    // Success Rate section
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        spacing: 6

                        Text {
                            text: "Success Rate"
                            color: "white"
                            font.bold: true
                            font.pixelSize: Math.max(10, Math.min(22, root.height * 0.03))
                        }

                        // Results table
                        Rectangle {
                            id: resultsTable
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            Layout.minimumWidth: root.minControlSize
                            Layout.minimumHeight: root.minControlSize
                            implicitWidth: Math.max(root.minControlSize,
                                                    Math.min(root.width * 0.35, 420))
                            implicitHeight: Math.max(root.minControlSize,
                                                    Math.min(root.height * 0.25, 220))
                            color: "#5f6b7a"
                            radius: 6
                            border.color: "#d0d6df"
                            property int contentWidth: width - 32

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 16
                                spacing: 4

                                // ---------- HEADER ROW ----------
                                Row {
                                    id: headerRow
                                    width: resultsTable.contentWidth
                                    spacing: 1

                                    Rectangle {
                                        // height scales with table; never smaller than 18
                                        height: Math.max(18, resultsTable.height * 0.08)
                                        width: (resultsTable.contentWidth - headerRow.spacing) / 2
                                        color: "white"

                                        Text {
                                            anchors.centerIn: parent
                                            text: "Class"
                                            color: "black"
                                            font.bold: true
                                            // ~45% of cell height, min 10px
                                            font.pixelSize: Math.max(10, parent.height * 0.45)
                                        }
                                    }

                                    Rectangle {
                                        height: Math.max(18, resultsTable.height * 0.08)
                                        width: (resultsTable.contentWidth - headerRow.spacing) / 2
                                        color: "white"

                                        Text {
                                            anchors.centerIn: parent
                                            text: "Precision"
                                            color: "black"
                                            font.bold: true
                                            font.pixelSize: Math.max(10, parent.height * 0.45)
                                        }
                                    }
                                }

                                // ---------- DATA MODEL ----------
                                ListModel {
                                    id: resultsModel
                                    ListElement { klass: "Backward"; precision: "1" }
                                    ListElement { klass: "Forward"; precision: "0.97" }
                                    ListElement { klass: "Left"; precision: "1" }
                                    ListElement { klass: "Right"; precision: "0.99" }
                                    ListElement { klass: "Land"; precision: "1" }
                                    ListElement { klass: "Takeoff"; precision: "1" }
                                }

                                // ---------- DATA ROWS ----------
                                ListView {
                                    id: resultsList
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    width: resultsTable.contentWidth
                                    clip: true
                                    model: resultsModel

                                    delegate: Row {
                                        id: resultRow
                                        width: resultsTable.contentWidth
                                        spacing: 1

                                        Rectangle {
                                            // scale row height with table; never smaller than 16
                                            height: Math.max(16, resultsTable.height * 0.06)
                                            width: (resultsTable.contentWidth - resultRow.spacing) / 2
                                            color: "white"

                                            Text {
                                                anchors.centerIn: parent
                                                text: model.klass
                                                color: "black"
                                                // ~50% of cell height, min 10px
                                                font.pixelSize: Math.max(10, parent.height * 0.5)
                                            }
                                        }

                                        Rectangle {
                                            height: Math.max(16, resultsTable.height * 0.06)
                                            width: (resultsTable.contentWidth - resultRow.spacing) / 2
                                            color: "white"

                                            Text {
                                                anchors.centerIn: parent
                                                text: model.precision
                                                color: "black"
                                                font.pixelSize: Math.max(10, parent.height * 0.5)
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // Buttons under Success Rate
                        RowLayout {
                                Layout.alignment: Qt.AlignHCenter
                                spacing: 24

                            // DEPLOY
                            Button {
                                id: deployBtn
                                property real circleSize: Math.max(root.minControlSize,
                                                                Math.min(root.height * 0.18,
                                                                            root.width * 0.10,
                                                                            130))
                                checkable: true
                                checked: mode === "Deploy"
                                onClicked: {
                                    mode = "Deploy"
                                    logToConsole("Mode changed: Deploy")
                                }
                                implicitWidth: circleSize
                                implicitHeight: circleSize
                                Layout.minimumWidth: root.minControlSize
                                Layout.minimumHeight: root.minControlSize
                                padding: 0

                                background: Rectangle {
                                    radius: Math.min(width, height) / 2
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
                                    font.pixelSize: Math.max(10,
                                                            Math.min(22, parent.height * 0.25))
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                    wrapMode: Text.NoWrap
                                }
                            }

                            // TRAIN
                            Button {
                                id: trainBtn
                                property real circleSize: Math.max(root.minControlSize,
                                                                Math.min(root.height * 0.18,
                                                                            root.width * 0.10,
                                                                            130))
                                checkable: true
                                checked: mode === "Train"
                                onClicked: {
                                    mode = "Train"
                                    logToConsole("Mode changed: Train")
                                }
                                implicitWidth: circleSize
                                implicitHeight: circleSize
                                Layout.minimumWidth: root.minControlSize
                                Layout.minimumHeight: root.minControlSize
                                padding: 0

                                background: Rectangle {
                                    radius: Math.min(width, height) / 2
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
                                    font.pixelSize: Math.max(10,
                                                            Math.min(22, parent.height * 0.25))
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                    wrapMode: Text.NoWrap
                                }
                            }
                        }
                    }

                    // Console Log
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        spacing: 6

                        Text {
                            text: "Console Log"
                            color: "white"
                            font.bold: true
                            font.pixelSize: Math.max(10, Math.min(22, root.height * 0.03))
                        }

                        // Outer frame (now matches Success Rate table)
                        Rectangle {
                            id: consoleFrame
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            Layout.minimumWidth: root.minControlSize
                            Layout.minimumHeight: root.minControlSize
                            implicitWidth: Math.max(root.minControlSize,
                                                    Math.min(root.width * 0.35, 420))
                            implicitHeight: Math.max(root.minControlSize,
                                                    Math.min(root.height * 0.25, 220))

                            color: "#5f6b7a"           // SAME as Success Rate box
                            radius: 6
                            border.color: "#d0d6df"    // SAME border color
                            border.width: 1

                            // Inner scrollable area
                            Rectangle {
                                anchors.fill: parent
                                anchors.margins: 8
                                color: "white"         // White interior like the table cells
                                radius: 4
                                border.color: "transparent"

                                ScrollView {
                                    anchors.fill: parent
                                    clip: true
                                    ScrollBar.vertical.policy: ScrollBar.AsNeeded

                                    TextArea {
                                        id: consoleLog
                                        readOnly: true
                                        wrapMode: Text.WrapAtWordBoundaryOrAnywhere
                                        text: "Console output here..."
                                        color: "black"
                                        background: Rectangle { color: "transparent" }
                                        font.pixelSize: Math.max(10,
                                                                Math.min(14, height * 0.12))
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // RIGHT PANE: PARAMETERS ---------------------------------------
            Rectangle {
                Layout.fillHeight: true
                Layout.minimumWidth: root.minControlSize
                Layout.minimumHeight: root.minControlSize
                implicitWidth: Math.max(root.minControlSize,
                                        Math.min(root.width * 0.30, 420))
                color: "#242c4d"
                radius: 8
                border.color: "#1a2035"
                opacity: paramsEnabled ? 1.0 : 0.55
                enabled: paramsEnabled

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 8

                    // ON/OFF indicator
                    Rectangle {
                        Layout.alignment: Qt.AlignRight
                        Layout.minimumWidth: root.minControlSize
                        Layout.minimumHeight: root.minControlSize
                        implicitWidth: Math.max(root.minControlSize,
                                                Math.min(root.width * 0.12, 180))
                        implicitHeight: Math.max(root.minControlSize,
                                                 Math.min(root.height * 0.10, 80))
                        radius: 8
                        color: paramsEnabled ? "#2f8f2f" : "#c13a2a"
                        border.color: "#202020"

                        Text {
                            anchors.centerIn: parent
                            text: paramsEnabled ? "ON" : "OFF"
                            color: "white"
                            font.bold: true
                            font.pixelSize: Math.max(10,
                                                     Math.min(28, parent.height * 0.6))
                        }

                        MouseArea {
                            anchors.fill: parent
                            onClicked: mode = paramsEnabled ? "Deploy" : "Train"
                        }
                    }

                    Text {
                        text: "Parameters"
                        color: "white"
                        font.bold: true
                        font.pixelSize: Math.max(10, Math.min(26, root.height * 0.04))
                    }

                    // RANDOM FOREST PARAMS ----------------------------------
                    ColumnLayout {
                        visible: currentModel === "Random Forest"
                        spacing: 6
                        Layout.fillWidth: true

                        RowLayout {
                            Layout.fillWidth: true; spacing: 4
                            Text {
                                text: "n_estimators:"
                                color: "white"
                                Layout.preferredWidth: 120
                                font.pixelSize: Math.max(10, Math.min(16, root.height * 0.02))
                            }
                            TextField {
                                text: rfNEstimators.toString()
                                inputMethodHints: Qt.ImhDigitsOnly
                                onEditingFinished: {
                                    var v = parseInt(text)
                                    rfNEstimators = isNaN(v) ? rfNEstimators : Math.max(1, v)
                                    text = rfNEstimators.toString()
                                }
                                Layout.fillWidth: true
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true; spacing: 4
                            Text {
                                text: "max_depth:"
                                color: "white"
                                Layout.preferredWidth: 120
                                font.pixelSize: Math.max(10, Math.min(16, root.height * 0.02))
                            }
                            TextField {
                                placeholderText: "None"
                                text: rfMaxDepth
                                onEditingFinished: rfMaxDepth = text.trim()
                                Layout.fillWidth: true
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true; spacing: 4
                            Text {
                                text: "max_features:"
                                color: "white"
                                Layout.preferredWidth: 120
                                font.pixelSize: Math.max(10, Math.min(16, root.height * 0.02))
                            }
                            ComboBox {
                                model: ["auto", "sqrt", "log2"]
                                currentIndex: Math.max(0, model.indexOf(rfMaxFeatures))
                                onCurrentTextChanged: rfMaxFeatures = currentText
                                Layout.fillWidth: true
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true; spacing: 4
                            Text {
                                text: "min_samples_split:"
                                color: "white"
                                Layout.preferredWidth: 120
                                font.pixelSize: Math.max(10, Math.min(16, root.height * 0.02))
                            }
                            TextField {
                                text: rfMinSamplesSplit.toString()
                                inputMethodHints: Qt.ImhDigitsOnly
                                onEditingFinished: {
                                    var v = parseInt(text)
                                    rfMinSamplesSplit = isNaN(v) ? rfMinSamplesSplit : Math.max(2, v)
                                    text = rfMinSamplesSplit.toString()
                                }
                                Layout.fillWidth: true
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true; spacing: 4
                            Text {
                                text: "min_samples_leaf:"
                                color: "white"
                                Layout.preferredWidth: 120
                                font.pixelSize: Math.max(10, Math.min(16, root.height * 0.02))
                            }
                            TextField {
                                text: rfMinSamplesLeaf.toString()
                                inputMethodHints: Qt.ImhDigitsOnly
                                onEditingFinished: {
                                    var v = parseInt(text)
                                    rfMinSamplesLeaf = isNaN(v) ? rfMinSamplesLeaf : Math.max(1, v)
                                    text = rfMinSamplesLeaf.toString()
                                }
                                Layout.fillWidth: true
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true; spacing: 4
                            Text {
                                text: "criterion:"
                                color: "white"
                                Layout.preferredWidth: 120
                                font.pixelSize: Math.max(10, Math.min(16, root.height * 0.02))
                            }
                            ComboBox {
                                model: ["gini", "entropy"]
                                currentIndex: Math.max(0, model.indexOf(rfCriterion))
                                onCurrentTextChanged: rfCriterion = currentText
                                Layout.fillWidth: true
                            }
                        }
                    }

                    // DEEP LEARNING PARAMS ----------------------------------
                    ColumnLayout {
                        visible: currentModel === "Deep Learning"
                        spacing: 6
                        Layout.fillWidth: true

                        RowLayout {
                            Layout.fillWidth: true; spacing: 4
                            Text {
                                text: "Learning rate:"
                                color: "white"
                                Layout.preferredWidth: 110
                                font.pixelSize: Math.max(10, Math.min(16, root.height * 0.02))
                            }
                            Slider {
                                Layout.fillWidth: true
                                from: 0.0001; to: 0.1; stepSize: 0.0001
                                value: learningRate
                                onValueChanged: learningRate = value
                            }
                            TextField {
                                text: Number(learningRate).toFixed(4)
                                onEditingFinished: {
                                    var v = parseFloat(text)
                                    if (!isNaN(v))
                                        learningRate = Math.max(0.0001, Math.min(0.1, v))
                                    text = Number(learningRate).toFixed(4)
                                }
                                Layout.preferredWidth: 80
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true; spacing: 4
                            Text {
                                text: "Batch size:"
                                color: "white"
                                Layout.preferredWidth: 110
                                font.pixelSize: Math.max(10, Math.min(16, root.height * 0.02))
                            }
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
                                    var v = parseInt(text)
                                    if (!isNaN(v))
                                        batchSize = Math.max(8, Math.min(256, v))
                                    text = batchSize.toString()
                                }
                                Layout.preferredWidth: 80
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true; spacing: 4
                            Text {
                                text: "Epoch #:"
                                color: "white"
                                Layout.preferredWidth: 110
                                font.pixelSize: Math.max(10, Math.min(16, root.height * 0.02))
                            }
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
                                    var v = parseInt(text)
                                    if (!isNaN(v))
                                        epochs = Math.max(1, Math.min(200, v))
                                    text = epochs.toString()
                                }
                                Layout.preferredWidth: 80
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true; spacing: 4
                            Text {
                                text: "Optimizer:"
                                color: "white"
                                Layout.preferredWidth: 110
                                font.pixelSize: Math.max(10, Math.min(16, root.height * 0.02))
                            }
                            ComboBox {
                                Layout.fillWidth: true
                                model: ["Adam", "SGD", "RMSProp", "AdamW"]
                                currentIndex: Math.max(0, model.indexOf(optimizer))
                                onCurrentTextChanged: optimizer = currentText
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true; spacing: 4
                            Text {
                                text: "Activation fn:"
                                color: "white"
                                Layout.preferredWidth: 110
                                font.pixelSize: Math.max(10, Math.min(16, root.height * 0.02))
                            }
                            ComboBox {
                                Layout.fillWidth: true
                                model: ["ReLU", "LeakyReLU", "Tanh", "Sigmoid", "GELU"]
                                currentIndex: Math.max(0, model.indexOf(activation))
                                onCurrentTextChanged: activation = currentText
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true; spacing: 4
                            Text {
                                text: "Drop out rate:"
                                color: "white"
                                Layout.preferredWidth: 110
                                font.pixelSize: Math.max(10, Math.min(16, root.height * 0.02))
                            }
                            Slider {
                                Layout.fillWidth: true
                                from: 0; to: 0.9; stepSize: 0.01
                                value: dropout
                                onValueChanged: dropout = value
                            }
                            TextField {
                                text: Number(dropout).toFixed(2)
                                onEditingFinished: {
                                    var v = parseFloat(text)
                                    if (!isNaN(v))
                                        dropout = Math.max(0, Math.min(0.9, v))
                                    text = Number(dropout).toFixed(2)
                                }
                                Layout.preferredWidth: 80
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true; spacing: 4
                            Text {
                                text: "L1/L2 choice:"
                                color: "white"
                                Layout.preferredWidth: 110
                                font.pixelSize: Math.max(10, Math.min(16, root.height * 0.02))
                            }
                            ComboBox {
                                Layout.preferredWidth: 80
                                model: ["L1", "L2"]
                                currentIndex: regChoice === "L1" ? 0 : 1
                                onCurrentTextChanged: regChoice = currentText
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true; spacing: 4
                            Text {
                                text: "Momentum:"
                                color: "white"
                                Layout.preferredWidth: 110
                                font.pixelSize: Math.max(10, Math.min(16, root.height * 0.02))
                            }
                            Slider {
                                Layout.fillWidth: true
                                from: 0; to: 0.99; stepSize: 0.01
                                value: momentum
                                onValueChanged: momentum = value
                            }
                            TextField {
                                text: Number(momentum).toFixed(2)
                                onEditingFinished: {
                                    var v = parseFloat(text)
                                    if (!isNaN(v))
                                        momentum = Math.max(0, Math.min(0.99, v))
                                    text = Number(momentum).toFixed(2)
                                }
                                Layout.preferredWidth: 80
                            }
                        }
                    }

                    Item { Layout.fillHeight: true } // spacer
                }
            }
        }
    }
}
