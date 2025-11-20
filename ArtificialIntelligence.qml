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
                    