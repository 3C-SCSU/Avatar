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

                    // Learning Rate: Controls how fast the model learns during training
                    // Lower values = slower but more stable learning
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
                    // More epochs = longer training time, potentially better results
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
                    // Larger batches = more memory usage but potentially more stable gradients
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
                    // Examples: CNN for images, LSTM for sequences, Transformer for NLP
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
                            placeholderText: "e.g., CNN, LSTM, Transformer"
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
                    Button {
                        Layout.alignment: Qt.AlignHCenter
                        Layout.preferredWidth: 200
                        Layout.preferredHeight: 40
                        text: "Start Training"
                        font.bold: true
                        onClicked: {
                            // Call backend to start training with all parameters
                            backend.startTraining(
                                learningRateField.text,
                                epochsField.text,
                                batchSizeField.text,
                                modelArchField.text,
                                dataPathField.text
                            )
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
