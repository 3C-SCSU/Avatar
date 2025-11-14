import QtQuick 6.5 
import QtQuick.Controls 6.4 
import QtQuick.Layouts 1.15 

Rectangle { 
    id: root color: "#718399" 
    Layout.fillWidth: true 
    Layout.fillHeight: true 

    // Base design size (size designed for the tab) 
    property int  baseWidth: 1280 
    property int  baseHeight: 720 
    property int  outerMargin: 80 
    
    // margins
    property int marginSide: 60 
    property int marginTop: 40 
    property int marginBottom: 80   // ensures the bottom never gets too close 
    
    // Scale factor: how much to scale the design to fit current window 
    property real scaleFactor: Math.min( 
    (width  - 2 * marginSide) / baseWidth, 
    (height - marginTop - marginBottom) / baseHeight 
    ) 
    
    
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
    
    Item { 
        id: contentRoot 
        width: baseWidth 
        height: baseHeight 
        
        // horizontally centered, vertically pinned to a fixed top margin 
        anchors.horizontalCenter: parent.horizontalCenter 
        anchors.top: parent.top 
        anchors.topMargin: marginTop 
        
        scale: root.scaleFactor 
        transformOrigin: Item.Top 
    
        RowLayout { 
            id: mainLayout 
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
                    Layout.preferredHeight: 220 
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
                            font.pixelSize: 32 
                            horizontalAlignment: Text.AlignHCenter 
                            Layout.alignment: Qt.AlignHCenter 
                        } 
        
                        // Model row 
                        RowLayout { 
                            Layout.alignment: Qt.AlignHCenter 
                            spacing: 16 
        
                            // Random Forest 
                            Rectangle { 
                                Layout.preferredWidth: 200      // <-- fixed design size 
                                Layout.preferredHeight: 60 
                                radius: 8 
                                color: "#6eb109" 
                                border.color: currentModel === "Random Forest" ? "yellow" : "#5a8c2b" 
                                border.width: currentModel === "Random Forest" ? 3 : 1 
                                Text { 
                                    anchors.centerIn: parent 
                                    text: "Random Forest" 
                                    color: "white" 
                                    font.bold: true 
                                    font.pixelSize: 18 
                                } 
                                MouseArea { 
                                    anchors.fill: parent 
                                    onClicked: { 
                                        currentModel = "Random Forest" 
                                        backend.selectModel("Random Forest") 
                                    } 
                                } 
                            } 
        
                            // Deep Learning 
                            Rectangle { 
                                Layout.preferredWidth: 200      // <-- fixed design size 
                                Layout.preferredHeight: 60 
                                radius: 8 
                                color: "#6eb109" 
                                border.color: currentModel === "Deep Learning" ? "yellow" : "#5a8c2b" 
                                border.width: currentModel === "Deep Learning" ? 3 : 1 
                                Text { 
                                    anchors.centerIn: parent 
                                    text: "Deep Learning" 
                                    color: "white" 
                                    font.bold: true 
                                    font.pixelSize: 18 
                                } 
                                MouseArea { 
                                    anchors.fill: parent 
                                    onClicked: { 
                                        currentModel = "Deep Learning" 
                                        backend.selectModel("Deep Learning") 
                                    } 
                                } 
                            } 
                        } 
                    } 
                } 
        
                // FRAMEWORK BOX 
                Rectangle { 
                    Layout.fillWidth: true 
                    Layout.preferredHeight: 220 
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
                            font.pixelSize: 32 
                            horizontalAlignment: Text.AlignHCenter 
                            Layout.alignment: Qt.AlignHCenter 
                        } 
        
                        // Framework row 
                        RowLayout { 
                            Layout.alignment: Qt.AlignHCenter 
                            spacing: 16 
        
                            // PyTorch 
                            Rectangle { 
                                Layout.preferredWidth: 160      // smaller looks nice here 
                                Layout.preferredHeight: 60 
                                radius: 8 
                                color: "#6eb109" 
                                border.color: currentFramework === "PyTorch" ? "yellow" : "#5a8c2b" 
                                border.width: currentFramework === "PyTorch" ? 3 : 1 
                                Text { 
                                    anchors.centerIn: parent 
                                    text: "PyTorch" 
                                    color: "white" 
                                    font.bold: true 
                                    font.pixelSize: 18 
                                } 
                                MouseArea { 
                                    anchors.fill: parent 
                                    onClicked: { 
                                        currentFramework = "PyTorch" 
                                        backend.selectFramework("PyTorch") 
                                    } 
                                } 
                            } 
        
                            // TensorFlow 
                            Rectangle { 
                                Layout.preferredWidth: 160 
                                Layout.preferredHeight: 60 
                                radius: 8 
                                color: "#6eb109" 
                                border.color: currentFramework === "TensorFlow" ? "yellow" : "#5a8c2b" 
                                border.width: currentFramework === "TensorFlow" ? 3 : 1 
                                Text { 
                                    anchors.centerIn: parent 
                                    text: "TensorFlow" 
                                    color: "white" 
                                    font.bold: true 
                                    font.pixelSize: 18 
                                } 
                                MouseArea { 
                                    anchors.fill: parent 
                                    onClicked: { 
                                        currentFramework = "TensorFlow" 
                                        backend.selectFramework("TensorFlow") 
                                    } 
                                } 
                            } 
        
                            // JAX (UI only; backend logs selection) 
                            Rectangle { 
                                Layout.preferredWidth: 120 
                                Layout.preferredHeight: 60 
                                radius: 8 
                                color: "#6eb109" 
                                border.color: currentFramework === "JAX" ? "yellow" : "#5a8c2b" 
                                border.width: currentFramework === "JAX" ? 3 : 1 
                                Text { 
                                    anchors.centerIn: parent 
                                    text: "JAX" 
                                    color: "white" 
                                    font.bold: true 
                                    font.pixelSize: 18 
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
        
                // Results Log label 
                Text { 
                    text: "Results Log:" 
                    color: "white" 
                    font.bold: true 
                    font.pixelSize: 18 
                } 
        
                // Results table with side buttons (Deploy above Train) 
                RowLayout { 
                    Layout.fillWidth: true 
                    spacing: 24 
        
                    // Results table 
                    Rectangle { 
                        id: resultsTable 
                        Layout.preferredWidth: 520 
                        Layout.preferredHeight: 260 
                        color: "#5f6b7a" 
                        radius: 6 
                        border.color: "#d0d6df" 
                        property int contentWidth: width - 32 
        
                        ColumnLayout { 
                            anchors.fill: parent 
                            anchors.margins: 16 
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
                                Layout.fillWidth: true 
                                Layout.fillHeight: true 
                                clip: true 
                                model: resultsModel 
                                delegate: Row { 
                                    id: resultRow 
                                    width: ListView.view.width 
                                    spacing: 1 
                                    Rectangle { 
                                        color: "white"; height: 28; width: (resultRow.width - resultRow.spacing) / 2 
                                        Text { anchors.centerIn: parent; text: model.klass; color: "black" } 
                                    } 
                                    Rectangle { 
                                        color: "white"; height: 28; width: (resultRow.width - resultRow.spacing) / 2 
                                        Text { anchors.centerIn: parent; text: model.precision; color: "black" } 
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
                        Layout.leftMargin: 28 
        
                        // DEPLOY (default) 
                        Button { 
                            id: deployBtn 
                            checkable: true 
                            checked: mode === "Deploy" 
                            onClicked: mode = "Deploy" 
                            Layout.preferredWidth: 110 
                            Layout.preferredHeight: 110 
                            padding: 0 
                            background: Rectangle { 
                                radius: width / 2 
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
                            onClicked: mode = "Train" 
                            Layout.preferredWidth: 110 
                            Layout.preferredHeight: 110 
                            padding: 0 
                            background: Rectangle { 
                                radius: width / 2 
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
        
            // RIGHT PANE (Parameters) 
            Rectangle { 
                Layout.preferredWidth: 380      // fixed design width; scales with contentRoot 
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
        
                    // ON/OFF indicator 
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
                        MouseArea { anchors.fill: parent; onClicked: mode = paramsEnabled ? "Deploy" : "Train" } 
                    } 
        
                    Text { 
                        text: "Parameters" 
                        color: "white" 
                        font.bold: true 
                        font.pixelSize: 26      // fixed; scales with contentRoot 
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
}  

    
