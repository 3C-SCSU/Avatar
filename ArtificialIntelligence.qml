import QtQuick.Dialogs
import Qt.labs.platform
import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtQuick3D 6.7

// Artificial Intelligence Tab view
Rectangle {
    id: artificialIntelligenceView
    
    // State properties
    property bool isTrainMode: false
    property bool isDeployMode: true
    property bool isRandomForestSelected: false
    property bool isDeepLearningSelected: true
    property bool isPyTorchSelected: true
    property bool isTensorFlowSelected: false
    property bool isJAXSelected: false
    property string trainingStatus: ""
    property string deploymentStatus: ""
    property var precisionData: {
        "Backward": "0.98",
        "Forward": "1.00",
        "Left": "0.97",
        "Right": "0.98",
        "Land": "0.96",
        "Takeoff": "0.98"
    }
    
    color: "#718399"
    Layout.fillWidth: true
    Layout.fillHeight: true

    // Main layout: RowLayout with left (buttons) and right (parameters) columns
    RowLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 20

        // ================= LEFT COLUMN: Buttons and Controls =================
        ColumnLayout {
            Layout.preferredWidth: parent.width * 0.45
            Layout.fillHeight: true
            spacing: 20

            // Tab title header
            Text {
                text: "Machine Learning"
                color: "white"
                font.bold: true
                font.pixelSize: 32
                Layout.alignment: Qt.AlignHCenter
            }

            // Model Selection Buttons: Random Forest and Deep Learning
            // These buttons allow users to choose which ML model to train/deploy
            // Same style as Brainwave Reading tab
            Row {
                Layout.alignment: Qt.AlignHCenter
                spacing: 20
            
            // Random Forest Button
            Rectangle {
                width: 150
                height: 50
                color: "#6eb109"
                radius: 5
                
                Text {
                    text: "Random Forest"
                    font.pixelSize: 16
                    font.bold: true
                    // Text color changes to yellow when selected, white when not
                    color: isRandomForestSelected ? "yellow" : "white"
                    anchors.centerIn: parent
                }
                
                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        isRandomForestSelected = true
                        isDeepLearningSelected = false
                        backend.selectModel("Random Forest")
                    }
                }
            }
            
            // Deep Learning Button
            Rectangle {
                width: 150
                height: 50
                color: "#6eb109"
                radius: 5
                
                Text {
                    text: "Deep Learning"
                    font.pixelSize: 16
                    font.bold: true
                    // Text color changes to yellow when selected, white when not
                    color: isDeepLearningSelected ? "yellow" : "white"
                    anchors.centerIn: parent
                }
                
                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        isDeepLearningSelected = true
                        isRandomForestSelected = false
                        backend.selectModel("Deep Learning")
                    }
                }
            }
        }

            // Framework title header (same style as Machine Learning title)
            Text {
                text: "Framework"
                color: "white"
                font.bold: true
                font.pixelSize: 32
                Layout.alignment: Qt.AlignHCenter
                Layout.topMargin: 10
            }

            // Framework Selection Buttons: PyTorch, TensorFlow, and JAX
            // These buttons allow users to choose which ML framework to use
            // JAX is found in the codebase (prediction-random-forest/JAX and prediction-deep-learning/JAX)
            Row {
                Layout.alignment: Qt.AlignHCenter
                spacing: 20
            
            // PyTorch Button
            Rectangle {
                width: 150
                height: 50
                color: "#6eb109"
                radius: 5
                
                Text {
                    text: "PyTorch"
                    font.pixelSize: 16
                    font.bold: true
                    // Text color changes to yellow when selected, white when not
                    color: isPyTorchSelected ? "yellow" : "white"
                    anchors.centerIn: parent
                }
                
                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        isPyTorchSelected = true
                        isTensorFlowSelected = false
                        isJAXSelected = false
                        backend.selectFramework("PyTorch")
                    }
                }
            }
            
            // TensorFlow Button
            Rectangle {
                width: 150
                height: 50
                color: "#6eb109"
                radius: 5
                
                Text {
                    text: "TensorFlow"
                    font.pixelSize: 16
                    font.bold: true
                    // Text color changes to yellow when selected, white when not
                    color: isTensorFlowSelected ? "yellow" : "white"
                    anchors.centerIn: parent
                }
                
                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        isTensorFlowSelected = true
                        isPyTorchSelected = false
                        isJAXSelected = false
                        backend.selectFramework("TensorFlow")
                    }
                }
            }
            
            // JAX Button (found in codebase: prediction-random-forest/JAX and prediction-deep-learning/JAX)
            Rectangle {
                width: 150
                height: 50
                color: "#6eb109"
                radius: 5
                
                Text {
                    text: "JAX"
                    font.pixelSize: 16
                    font.bold: true
                    // Text color changes to yellow when selected, white when not
                    color: isJAXSelected ? "yellow" : "white"
                    anchors.centerIn: parent
                }
                
                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        isJAXSelected = true
                        isPyTorchSelected = false
                        isTensorFlowSelected = false
                        backend.selectFramework("JAX")
                    }
                }
            }
        }

            // Toggle Buttons: Train and Deploy
            // These buttons allow users to switch between training and deployment modes
            // Only one mode can be active at a time
            Row {
                Layout.alignment: Qt.AlignHCenter
                Layout.topMargin: 10
                spacing: 20

            // Train Button - Activates training parameter configuration interface
            Rectangle {
                width: 150
                height: 50
                // Color changes to green when active, dark blue-gray when inactive
                color: isTrainMode ? "#6eb109" : "#64778d"
                radius: 5
                border.color: "#4a5d6e"
                border.width: 2

                Text {
                    text: "Train"
                    font.pixelSize: 18
                    font.bold: true
                    // Text color changes to yellow when active
                    color: isTrainMode ? "yellow" : "white"
                    anchors.centerIn: parent
                }

                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        isTrainMode = true
                        isDeployMode = false
                    }
                }
            }

            // Deploy Button - Activates model deployment interface (default mode)
            Rectangle {
                width: 150
                height: 50
                // Color changes to green when active, dark blue-gray when inactive
                color: isDeployMode ? "#6eb109" : "#64778d"
                radius: 5
                border.color: "#4a5d6e"
                border.width: 2

                Text {
                    text: "Deploy"
                    font.pixelSize: 18
                    font.bold: true
                    // Text color changes to yellow when active
                    color: isDeployMode ? "yellow" : "white"
                    anchors.centerIn: parent
                }

                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        isDeployMode = true
                        isTrainMode = false
                    }
                }
            }
            }
            
            // Result Log Table - Shows precision metrics for each class
            // Sized to fit all classes and precision values
            Rectangle {
                Layout.alignment: Qt.AlignHCenter
                Layout.topMargin: 20
                width: Math.max(160, tableContent.implicitWidth + 30)
                height: tableContent.implicitHeight + 30
                color: "#64778d"
                radius: 8
                border.color: "#4a5d6e"
                border.width: 2
                
                Column {
                    id: tableContent
                    anchors.fill: parent
                    anchors.margins: 15
                    spacing: 10
                    
                    // Table Title
                    Text {
                        text: "Result Log"
                        color: "white"
                        font.bold: true
                        font.pixelSize: 14
                        Layout.alignment: Qt.AlignHCenter
                    }
                    
                    // Table Headers
                    Row {
                        width: parent.width - 30
                        spacing: 0
                        
                        // Class Header
                        Rectangle {
                            width: Math.max(80, classHeaderText.implicitWidth + 16)
                            height: 25
                            color: "#5a6d7d"
                            border.color: "#4a5d6e"
                            border.width: 1
                            
                            Text {
                                id: classHeaderText
                                text: "Class"
                                color: "white"
                                font.bold: true
                                font.pixelSize: 12
                                anchors.centerIn: parent
                            }
                        }
                        
                        // Precision Header
                        Rectangle {
                            width: Math.max(80, precisionHeaderText.implicitWidth + 16)
                            height: 25
                            color: "#5a6d7d"
                            border.color: "#4a5d6e"
                            border.width: 1
                            
                            Text {
                                id: precisionHeaderText
                                text: "Precision"
                                color: "white"
                                font.bold: true
                                font.pixelSize: 12
                                anchors.centerIn: parent
                            }
                        }
                    }
                    
                    // Table Rows - Classes and their precision values
                    // Data pulled from codebase classification reports
                    Repeater {
                        model: [
                            { class: "Backward", precision: precisionData["Backward"] },
                            { class: "Forward", precision: precisionData["Forward"] },
                            { class: "Left", precision: precisionData["Left"] },
                            { class: "Right", precision: precisionData["Right"] },
                            { class: "Land", precision: precisionData["Land"] },
                            { class: "Takeoff", precision: precisionData["Takeoff"] }
                        ]
                        
                        Row {
                            spacing: 0
                            
                            // Class Name Cell
                            Rectangle {
                                width: Math.max(80, classHeaderText.width)
                                height: 28
                                color: index % 2 === 0 ? "#64778d" : "#5a6d7d"
                                border.color: "#4a5d6e"
                                border.width: 1
                                
                                Text {
                                    text: modelData.class
                                    color: "white"
                                    font.pixelSize: 11
                                    anchors.left: parent.left
                                    anchors.leftMargin: 8
                                    anchors.verticalCenter: parent.verticalCenter
                                }
                            }
                            
                            // Precision Value Cell
                            Rectangle {
                                width: Math.max(80, precisionHeaderText.width)
                                height: 28
                                color: index % 2 === 0 ? "#64778d" : "#5a6d7d"
                                border.color: "#4a5d6e"
                                border.width: 1
                                
                                Text {
                                    text: modelData.precision
                                    color: "white"
                                    font.pixelSize: 11
                                    anchors.right: parent.right
                                    anchors.rightMargin: 8
                                    anchors.verticalCenter: parent.verticalCenter
                                }
                            }
                        }
                    }
                }
            }
            
            // Connection to backend signals
            Connections {
                target: backend
                function onTrainingStatusUpdated(message) {
                    trainingStatus = message
                }
                function onDeploymentStatusUpdated(message) {
                    deploymentStatus = message
                }
                function onPrecisionMetricsUpdated(metrics) {
                    precisionData = metrics
                }
            }
            
            Component.onCompleted: {
                if (typeof backend !== 'undefined' && backend.getPrecisionMetrics) {
                    var metrics = backend.getPrecisionMetrics()
                    if (metrics) {
                        precisionData = metrics
                    }
                }
            }
        }

        // ================= RIGHT COLUMN: Parameters and Deployment Interface =================
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: "#64778d"
            radius: 8
            border.color: "#4a5d6e"
            border.width: 2

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 20
                spacing: 15

                // TRAIN MODE: Parameter Fields Section
                // This section is visible when Train mode is active AND a model (Random Forest or Deep Learning) is selected
                // Contains input fields for configuring model training hyperparameters
                // Based on codebase logic: parameters only active when Train is selected
                ColumnLayout {
                    id: parameterFields
                    visible: isTrainMode && (isRandomForestSelected || isDeepLearningSelected)
                    Layout.fillWidth: true
                    spacing: 15

                    Text {
                        text: "Training Parameters"
                        color: "white"
                        font.bold: true
                        font.pixelSize: 20
                        Layout.alignment: Qt.AlignHCenter
                    }

                    // DEEP LEARNING PARAMETERS
                    // These parameters are shown when Deep Learning model is selected
                    // Only using parameters found in the codebase
                    ColumnLayout {
                        id: deepLearningParams
                        visible: isDeepLearningSelected
                        Layout.fillWidth: true
                        spacing: 10

                        // Learning Rate: Controls how fast the model learns during training (found in codebase)
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "Learning rate:"
                                color: "white"
                                font.pixelSize: 14
                                Layout.preferredWidth: 150
                            }

                            TextField {
                                id: learningRateField
                                Layout.fillWidth: true
                                placeholderText: "e.g., 0.001"
                            }
                        }

                        // Batch Size: Number of samples processed before updating model weights (found in codebase)
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "Batch size:"
                                color: "white"
                                font.pixelSize: 14
                                Layout.preferredWidth: 150
                            }

                            TextField {
                                id: batchSizeField
                                Layout.fillWidth: true
                                placeholderText: "e.g., 32"
                            }
                        }

                        // Epochs: Number of complete passes through the training dataset (found in codebase)
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "Epoch #:"
                                color: "white"
                                font.pixelSize: 14
                                Layout.preferredWidth: 150
                            }

                            TextField {
                                id: epochsField
                                Layout.fillWidth: true
                                placeholderText: "e.g., 100"
                            }
                        }

                        // Optimizer: Algorithm used to update model weights (found in codebase: Adam, SGD)
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "Optimizer:"
                                color: "white"
                                font.pixelSize: 14
                                Layout.preferredWidth: 150
                            }

                            TextField {
                                id: optimizerField
                                Layout.fillWidth: true
                                placeholderText: "e.g., Adam, SGD"
                            }
                        }

                        // Activation Function: Non-linearity applied to neurons (found in codebase: ReLU)
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "Activation fn:"
                                color: "white"
                                font.pixelSize: 14
                                Layout.preferredWidth: 150
                            }

                            TextField {
                                id: activationField
                                Layout.fillWidth: true
                                placeholderText: "e.g., ReLU, Tanh"
                            }
                        }

                        // Dropout Rate: Fraction of neurons randomly deactivated to prevent overfitting (found in codebase)
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "Drop out rate:"
                                color: "white"
                                font.pixelSize: 14
                                Layout.preferredWidth: 150
                            }

                            TextField {
                                id: dropoutRateField
                                Layout.fillWidth: true
                                placeholderText: "e.g., 0.3"
                            }
                        }

                        // L1/L2 Choice: Regularization method (L1 or L2)
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "L1/L2 choice:"
                                color: "white"
                                font.pixelSize: 14
                                Layout.preferredWidth: 150
                            }

                            TextField {
                                id: l1l2ChoiceField
                                Layout.fillWidth: true
                                placeholderText: "e.g., L1, L2, or None"
                            }
                        }

                        // Momentum: Momentum factor for optimizers (found in codebase: momentum=0.9 in BatchNorm)
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "Momentum:"
                                color: "white"
                                font.pixelSize: 14
                                Layout.preferredWidth: 150
                            }

                            TextField {
                                id: momentumField
                                Layout.fillWidth: true
                                placeholderText: "e.g., 0.9"
                            }
                        }
                    }

                    // RANDOM FOREST PARAMETERS
                    // These parameters are shown when Random Forest model is selected
                    // Only using parameters found in the codebase (JAX and PyTorch implementations)
                    ColumnLayout {
                        id: randomForestParams
                        visible: isRandomForestSelected
                        Layout.fillWidth: true
                        spacing: 10

                        // n_estimators: Number of trees in the forest (found in codebase: n_estimators=100/300, n_trees=40)
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "n_estimators:"
                                color: "white"
                                font.pixelSize: 14
                                Layout.preferredWidth: 150
                            }

                            TextField {
                                id: nEstimatorsField
                                Layout.fillWidth: true
                                placeholderText: "e.g., 100"
                            }
                        }

                        // max_depth: Maximum depth of the trees (found in codebase: max_depth=8/12/20)
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "max_depth:"
                                color: "white"
                                font.pixelSize: 14
                                Layout.preferredWidth: 150
                            }

                            TextField {
                                id: maxDepthField
                                Layout.fillWidth: true
                                placeholderText: "e.g., 20"
                            }
                        }

                        // min_samples_split: Minimum samples required to split a node (found in codebase: min_samples_split=50/100)
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "min_samples_split:"
                                color: "white"
                                font.pixelSize: 14
                                Layout.preferredWidth: 150
                            }

                            TextField {
                                id: minSamplesSplitField
                                Layout.fillWidth: true
                                placeholderText: "e.g., 50"
                            }
                        }

                        // min_samples_leaf: Minimum samples required in a leaf node (found in codebase: MIN_SAMPLES_LEAF=50)
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "min_samples_leaf:"
                                color: "white"
                                font.pixelSize: 14
                                Layout.preferredWidth: 150
                            }

                            TextField {
                                id: minSamplesLeafField
                                Layout.fillWidth: true
                                placeholderText: "e.g., 50"
                            }
                        }

                        // max_features: Number of features to consider for best split (found in codebase: nb_features=None in PyTorch, FEATURE_SUBSAMPLE_RATIO=0.2 in JAX)
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "max_features:"
                                color: "white"
                                font.pixelSize: 14
                                Layout.preferredWidth: 150
                            }

                            TextField {
                                id: maxFeaturesField
                                Layout.fillWidth: true
                                placeholderText: "e.g., sqrt, log2, or number"
                            }
                        }

                        // bootstrap: Whether to use bootstrap sampling when building trees (found in codebase: bootstrap=True, bootstrap_ratio=0.6)
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "bootstrap:"
                                color: "white"
                                font.pixelSize: 14
                                Layout.preferredWidth: 150
                            }

                            TextField {
                                id: bootstrapField
                                Layout.fillWidth: true
                                placeholderText: "true or false"
                            }
                        }

                        // criterion: Function to measure split quality (found in codebase: Gini impurity hardcoded, sklearn supports "gini" or "entropy")
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "criterion:"
                                color: "white"
                                font.pixelSize: 14
                                Layout.preferredWidth: 150
                            }

                            TextField {
                                id: criterionField
                                Layout.fillWidth: true
                                placeholderText: "e.g., gini, entropy"
                            }
                        }

                        // random_state: Seed for random number generation (found in codebase: seed=42, random_state=42)
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "random_state:"
                                color: "white"
                                font.pixelSize: 14
                                Layout.preferredWidth: 150
                            }

                            TextField {
                                id: randomStateField
                                Layout.fillWidth: true
                                placeholderText: "e.g., 42"
                            }
                        }

                        // n_jobs: Number of parallel jobs to run (found in codebase: n_jobs=-1 in sklearn RandomForestClassifier in GUI5.py)
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "n_jobs:"
                                color: "white"
                                font.pixelSize: 14
                                Layout.preferredWidth: 150
                            }

                            TextField {
                                id: nJobsField
                                Layout.fillWidth: true
                                placeholderText: "e.g., -1 (all cores)"
                            }
                        }
                    }

                    // Training Data Path: Location of the dataset for training
                    // Supports file browsing via the Browse button
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 10

                        Text {
                            text: "Training Data Path:"
                            color: "white"
                            font.pixelSize: 14
                            Layout.preferredWidth: 150
                        }

                        TextField {
                            id: dataPathField
                            Layout.fillWidth: true
                            placeholderText: "/path/to/training/data"
                        }

                        Button {
                            text: "Browse"
                            onClicked: {
                                // TODO: Implement file dialog to select training data directory
                                console.log("Browse for training data")
                            }
                        }
                    }

                    // Start Training Button - Initiates the training process with configured parameters
                    // Passes different parameters based on selected model type (Deep Learning or Random Forest)
                    Button {
                        Layout.alignment: Qt.AlignHCenter
                        Layout.preferredWidth: 200
                        Layout.preferredHeight: 40
                        text: "Start Training"
                        font.bold: true
                        onClicked: {
                            // Pass parameters based on selected model type
                            if (isDeepLearningSelected) {
                                // Deep Learning parameters (only parameters found in codebase)
                                // Architecture defaults to "MLP" if not specified (as per codebase pattern)
                                backend.startTraining(
                                    learningRateField.text,
                                    epochsField.text,
                                    batchSizeField.text,
                                    "MLP",  // Default architecture (found in codebase)
                                    dataPathField.text
                                )
                            } else if (isRandomForestSelected) {
                                // Random Forest parameters (only parameters found in codebase)
                                // Pass them in a format the backend can parse
                                backend.startTraining(
                                    nEstimatorsField.text,
                                    maxDepthField.text,
                                    minSamplesSplitField.text,  // Using min_samples_split instead of max_features
                                    "Random Forest",  // Architecture type
                                    dataPathField.text
                                )
                            }
                        }
                    }
                    
                    // Training Status Display
                    Text {
                        visible: trainingStatus !== ""
                        text: "Status: " + trainingStatus
                        color: "white"
                        font.pixelSize: 14
                        Layout.alignment: Qt.AlignHCenter
                        Layout.topMargin: 10
                    }
                }

                // DEPLOY MODE: Model Deployment Section
                // This section is only visible when Deploy mode is active (default)
                // Allows users to select and deploy a pre-trained model for inference
                ColumnLayout {
                    id: deployContent
                    visible: isDeployMode
                    Layout.fillWidth: true
                    Layout.fillHeight: true

                    Text {
                        text: "Deploy Model"
                        color: "white"
                        font.bold: true
                        font.pixelSize: 20
                        Layout.alignment: Qt.AlignHCenter
                    }

                    Text {
                        text: "Select a trained model to deploy for inference."
                        color: "#b0c4de"
                        font.pixelSize: 16
                        Layout.alignment: Qt.AlignHCenter
                        wrapMode: Text.WordWrap
                        Layout.maximumWidth: parent.width - 40
                        horizontalAlignment: Text.AlignHCenter
                    }

                    // Model file selection - supports various model formats (.pkl, .h5, etc.)
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 10
                        Layout.topMargin: 20

                        Text {
                            text: "Model File:"
                            color: "white"
                            font.pixelSize: 14
                            Layout.preferredWidth: 150
                        }

                        TextField {
                            id: modelFileField
                            Layout.fillWidth: true
                            placeholderText: "/path/to/model.pkl or model.h5"
                        }

                        Button {
                            text: "Browse"
                            onClicked: {
                                // TODO: Implement file dialog to select model file
                                console.log("Browse for model file")
                            }
                        }
                    }

                    // Deploy Button - Loads the selected model and prepares it for inference
                    Button {
                        Layout.alignment: Qt.AlignHCenter
                        Layout.preferredWidth: 200
                        Layout.preferredHeight: 40
                        Layout.topMargin: 20
                        text: "Deploy Model"
                        font.bold: true
                        onClicked: {
                            // Call backend to deploy the selected model
                            backend.deployModel(modelFileField.text)
                        }
                    }
                    
                    // Deployment Status Display
                    Text {
                        visible: deploymentStatus !== ""
                        text: "Status: " + deploymentStatus
                        color: "white"
                        font.pixelSize: 14
                        Layout.alignment: Qt.AlignHCenter
                        Layout.topMargin: 10
                    }
                }
            }
        }
    }
}
