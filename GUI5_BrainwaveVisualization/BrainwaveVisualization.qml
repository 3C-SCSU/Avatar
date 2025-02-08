import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

ApplicationWindow {
    visible: true
    width: 1200
    height: 800 
    title: "Avatar - Brainwave Reading"

    Connections {
        target: backend
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
        spacing: 0  // Remove spacing between components

        // Tab bar
        TabBar {
            id: tabBar
            Layout.fillWidth: true
            currentIndex: 0  // Default to first tab
            height: 40

            TabButton {
                text: "Brainwave Visualization"
            }
        }

        // Stack layout for tab content
        StackLayout {
            id: stackLayout
            Layout.fillWidth: true
            Layout.fillHeight: true  // Critical: Fill remaining space
            currentIndex: tabBar.currentIndex  // Sync with TabBar

            // Tab 1: Brainwave Visualization
            Rectangle {
                color: "#2b3a4a"
                
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

                    // Grid Layout for Graphs
                    GridLayout {
                        columns: 3
                        Layout.fillWidth: true
                        Layout.fillHeight: true  // Ensure grid fills space
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
                                    anchors.fill: parent
                                    spacing: 5

                                    Text {
                                        text: model.graphTitle
                                        color: "white"
                                        font.bold: true
                                        Layout.alignment: Qt.AlignHCenter
                                        Layout.topMargin: 10
                                    }

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
        }
    }
}