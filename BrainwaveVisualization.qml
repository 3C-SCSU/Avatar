import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtQuick3D 6.7
import QtQuick.Dialogs
import Qt.labs.platform

// Brainwave Visualization
Rectangle {
    color: "#718399"
    Layout.fillWidth: true
    Layout.fillHeight: true

    ColumnLayout {
        anchors.fill: parent
        spacing: 10

        // Grid Layout for 6 Graphs (2 Rows × 3 Columns)
        GridLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            columns: 3
            Layout.margins: 10
            columnSpacing: 10
            rowSpacing: 10

            Repeater {
                model: imageModel
                delegate: Rectangle {
                    color: "#e6e6f0"
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    border.color: "#d0d0d8"
                    border.width: 1
                    radius: 4

                    Column {
                        width: parent.width
                        height: parent.height
                        spacing: 0

                        Rectangle {
                            width: parent.width
                            height: 30
                            color: "#242c4d"

                            Text {
                                // Extract just the first word from the title
                                text: {
                                    var parts = model.graphTitle.split(" ");
                                    return parts[0];
                                }
                                color: "white"
                                font.bold: true
                                font.pixelSize: 14
                                x: (parent.width - width) / 2
                                y: (parent.height - height) / 2
                            }
                        }

                        Rectangle {
                            width: parent.width
                            height: parent.height - 30
                            color: "white"

                            //Display Image
                            Image {
                                x: 8
                                y: 8
                                width: parent.width - 16
                                height: parent.height - 16
                                source: model.imagePath
                                fillMode: Image.PreserveAspectFit
                            }
                        }
                    }
                }
            }
        }

        // Refresh and Rollback buttons after graphs
        RowLayout {
            Layout.alignment: Qt.AlignHCenter | Qt.AlignBottom
            Layout.bottomMargin: 20
            spacing: 15

            // Refresh Button
            Button {
                text: "Refresh"
                font.bold: true
                implicitWidth: 120
                implicitHeight: 40
                property bool isHovering: false

                HoverHandler { onHoveredChanged: parent.isHovering = hovered }

                background: Rectangle {
                    // Use the isHovering property to change color
                    color: parent.isHovering ? "#3e4e7a" : "#2e3a5c"
                    radius: 4

                    // Add a smooth color transition
                    Behavior on color { 
                        ColorAnimation { 
                            duration: 150 
                        } 
                    }
                }

                contentItem: Text {
                    text: parent.text
                    font.pixelSize: 14
                    font.bold: true
                    color: "white"
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }

                onClicked: {
                    backend.setDataset("refresh")
                }
            }

            // Rollback Button
            Button {
                text: "Rollback"
                font.bold: true
                implicitWidth: 120
                implicitHeight: 40
                
                // This property allows us to track the hover state
                property bool isHovering: false

                // Define the hover handler
                HoverHandler { 
                    onHoveredChanged: parent.isHovering = hovered 
                }

                background: Rectangle {
                    // Use the isHovering property to change color
                    color: parent.isHovering ? "#3e4e7a" : "#2e3a5c"
                    radius: 4
                    
                    // Add a smooth color transition
                    Behavior on color { 
                        ColorAnimation { 
                            duration: 150 
                        } 
                    }
                }

                contentItem: Text {
                    text: parent.text
                    font.pixelSize: 14
                    font.bold: true
                    color: "white"
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }

                onClicked: {
                    // Display rollback plots
                    backend.setDataset("rollback")
                }
            }
        }
    }
}
