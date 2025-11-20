import QtQuick.Dialogs
import Qt.labs.platform
import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtQuick3D 6.7

// Artificial Intelligence Tab view 🤡
// The new Tab: [ Artificial Intelligence ] will have all functionality related to A.I./M.L. in one tab. 
//The tab will allow the user to choose to deploy or train the model. 
//The [ Deploy ] option will be the default. And according to the Machine Learning model selected, 
//if the user chose to [ Train ], then the Parameters will with TextInput field become active [ ON ],
// with the list of parameter applicable to that type of model.
Rectangle {
    id: artificialIntelligenceView
    
    // Connection to backend signals for status updates
    Connections {
        target: backend
        function onTrainingStatusUpdated(message) {
            trainingStatus = message
        }
        function onDeploymentStatusUpdated(message) {
            deploymentStatus = message
        }
    }
    
    // State properties to track which mode is active
    // Deploy mode is the default when the tab loads
    property bool isTrainMode: false
    property bool isDeployMode: true
    
    // Model selection properties (matching Brainwave Reading tab style)
    property bool isRandomForestSelected: false
    property bool isDeepLearningSelected: true  // Default to Deep Learning
    
    // Framework selection properties (matching Brainwave Reading tab style)
    property bool isPyTorchSelected: true  // Default to PyTorch
    property bool isTensorFlowSelected: false
    
    // Status messages from backend
    property string trainingStatus: ""
    property string deploymentStatus: ""
    
    color: "#718399"
    Layout.fillWidth: true
    Layout.fillHeight: true

    // Main layout container for all tab content
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
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
                        // Call backend to notify model selection
                        if (typeof backend !== 'undefined' && backend.selectModel) {
                            backend.selectModel("Random Forest")
                        }
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
                        // Call backend to notify model selection
                        if (typeof backend !== 'undefined' && backend.selectModel) {
                            backend.selectModel("Deep Learning")
                        }
                    }
                }
            }
        }

        // Framework Selection Buttons: PyTorch and TensorFlow
        // These buttons allow users to choose which ML framework to use
        // Same style as Brainwave Reading tab
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
                        // Call backend to notify framework selection
                        if (typeof backend !== 'undefined' && backend.selectFramework) {
                            backend.selectFramework("PyTorch")
                        }
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
                        // Call backend to notify framework selection
                        if (typeof backend !== 'undefined' && backend.selectFramework) {
                            backend.selectFramework("TensorFlow")
                        }
                    }
                }
            }
        }

        // Toggle Buttons: Train and Deploy
        // These buttons allow users to switch between training and deployment modes
        // Only one mode can be active at a time
        Row {
            Layout.alignment: Qt.AlignHCenter
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

        // Main content container
        // This rectangle dynamically shows either training parameters or deployment interface
        // based on which mode is currently active
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
                // This section is only visible when Train mode is active
                // Contains input fields for configuring model training hyperparameters
                ColumnLayout {
                    id: parameterFields
                    visible: isTrainMode
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
                    ColumnLayout {
                        id: deepLearningParams
                        visible: isDeepLearningSelected
                        Layout.fillWidth: true
                        spacing: 10

                        // Learning Rate: Controls how fast the model learns during training
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "Learning Rate:"
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

                        // Epochs: Number of complete passes through the training dataset
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "Epochs:"
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

                        // Batch Size: Number of samples processed before updating model weights
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "Batch Size:"
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

                        // Model Architecture: Type of neural network to train
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "Model Architecture:"
                                color: "white"
                                font.pixelSize: 14
                                Layout.preferredWidth: 150
                            }

                            TextField {
                                id: modelArchField
                                Layout.fillWidth: true
                                placeholderText: "e.g., CNN, LSTM, MLP"
                            }
                        }

                        // Optimizer: Algorithm used to update model weights
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

                        // Activation Function: Non-linearity applied to neurons
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "Activation Function:"
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

                        // Dropout Rate: Fraction of neurons randomly deactivated to prevent overfitting
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 10

                            Text {
                                text: "Dropout Rate:"
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
                    }

                    // RANDOM FOREST PARAMETERS
                    // These parameters are shown when Random Forest model is selected
                    ColumnLayout {
                        id: randomForestParams
                        visible: isRandomForestSelected
                        Layout.fillWidth: true
                        spacing: 10

                        // n_estimators: Number of trees in the forest
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

                        // max_depth: Maximum depth of the trees
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

                        // max_features: Number of features to consider for best split
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

                        // min_samples_split: Minimum samples required to split a node
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
                                placeholderText: "e.g., 2"
                            }
                        }

                        // min_samples_leaf: Minimum samples required in a leaf node
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
                                placeholderText: "e.g., 1"
                            }
                        }

                        // bootstrap: Whether to use bootstrap sampling when building trees
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

                        // random_state: Seed for random number generation
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

                        // n_jobs: Number of parallel jobs to run
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
                                // Deep Learning parameters
                                backend.startTraining(
                                    learningRateField.text,
                                    epochsField.text,
                                    batchSizeField.text,
                                    modelArchField.text,
                                    dataPathField.text
                                )
                            } else if (isRandomForestSelected) {
                                // Random Forest parameters
                                // Note: Backend may need to be updated to handle RF-specific parameters
                                // For now, pass them in a format the backend can parse
                                backend.startTraining(
                                    nEstimatorsField.text,
                                    maxDepthField.text,
                                    maxFeaturesField.text,
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
