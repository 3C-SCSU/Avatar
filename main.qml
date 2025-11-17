import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtQuick3D 6.7
import QtQuick.Dialogs
import Qt.labs.platform

ApplicationWindow {
    visible: true
    width: 1200
    height: 800
    title: "Avatar - BCI"

    ListModel { id: imageModel }

    ColumnLayout {
        anchors.fill: parent
        spacing: 10

        // ===== TOP TAB BAR =====
        TabBar {
            id: topTabBar
            Layout.fillWidth: true
            height: 40

            TabButton {
                // this tab shows page index 0
                property int targetIndex: 0

                text: "Read Brain"
                font.bold: true
                onClicked: stackLayout.currentIndex = targetIndex

                background: Rectangle {
                    anchors.fill: parent
                    color: stackLayout.currentIndex === parent.targetIndex ? "green" : "#444444"
                }
                contentItem: Text {
                    text: parent.text
                    anchors.centerIn: parent
                    color: stackLayout.currentIndex === parent.targetIndex ? "yellow" : "white"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideRight
                }
            }

            TabButton {
                property int targetIndex: 2

                text: "Manual Drone Control"
                font.bold: true
                onClicked: stackLayout.currentIndex = targetIndex

                background: Rectangle {
                    anchors.fill: parent
                    color: stackLayout.currentIndex === parent.targetIndex ? "green" : "#444444"
                }
                contentItem: Text {
                    text: parent.text
                    anchors.centerIn: parent
                    color: stackLayout.currentIndex === parent.targetIndex ? "yellow" : "white"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideRight
                }
            }

            TabButton {
                property int targetIndex: 3

                text: "Manual NAO Control"
                font.bold: true
                onClicked: {
                    stackLayout.currentIndex = targetIndex
                    console.log("Manual Controller tab clicked")
                    tabController.startNaoViewer()
                }

                background: Rectangle {
                    anchors.fill: parent
                    color: stackLayout.currentIndex === parent.targetIndex ? "green" : "#444444"
                }
                contentItem: Text {
                    text: parent.text
                    anchors.centerIn: parent
                    color: stackLayout.currentIndex === parent.targetIndex ? "yellow" : "white"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideRight
                }
            }

            TabButton {
                property int targetIndex: 7

                text: "Artificial Intelligence"
                font.bold: true
                onClicked: {
                    stackLayout.currentIndex = targetIndex
                    console.log("Artificial Intelligence tab clicked")
                }

                background: Rectangle {
                    anchors.fill: parent
                    color: stackLayout.currentIndex === parent.targetIndex ? "green" : "#444444"
                }
                contentItem: Text {
                    text: parent.text
                    anchors.centerIn: parent
                    color: stackLayout.currentIndex === parent.targetIndex ? "yellow" : "white"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideRight
                }
            }
        }

        // ===== MAIN STACK LAYOUT =====
        StackLayout {
            id: stackLayout
            Layout.fillWidth: true
            Layout.fillHeight: true

            BrainwaveReading        { Layout.fillWidth: true; Layout.fillHeight: true } // index 0
            BrainwaveVisualization  { Layout.fillWidth: true; Layout.fillHeight: true } // index 1
            ManualDroneControl      { Layout.fillWidth: true; Layout.fillHeight: true } // index 2
            ManualNaoControl        { Layout.fillWidth: true; Layout.fillHeight: true } // index 3
            FileShuffler            { Layout.fillWidth: true; Layout.fillHeight: true } // index 4
            TransferData            { Layout.fillWidth: true; Layout.fillHeight: true } // index 5
            Developers              { Layout.fillWidth: true; Layout.fillHeight: true } // index 6
            ArtificialIntelligence  { Layout.fillWidth: true; Layout.fillHeight: true } // index 7
        }

        // ===== BOTTOM TAB BAR =====
        TabBar {
            id: bottomTabBar
            Layout.fillWidth: true
            height: 40
            position: TabBar.Footer

            TabButton {
                property int targetIndex: 1

                text: "Brainwave Visualization"
                font.bold: true
                onClicked: stackLayout.currentIndex = targetIndex

                background: Rectangle {
                    anchors.fill: parent
                    color: stackLayout.currentIndex === parent.targetIndex ? "green" : "#444444"
                }
                contentItem: Text {
                    text: parent.text
                    anchors.centerIn: parent
                    color: stackLayout.currentIndex === parent.targetIndex ? "yellow" : "white"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideRight
                }
            }

            TabButton {
                property int targetIndex: 4

                text: "File Shuffler"
                font.bold: true
                onClicked: stackLayout.currentIndex = targetIndex

                background: Rectangle {
                    anchors.fill: parent
                    color: stackLayout.currentIndex === parent.targetIndex ? "green" : "#444444"
                }
                contentItem: Text {
                    text: parent.text
                    anchors.centerIn: parent
                    color: stackLayout.currentIndex === parent.targetIndex ? "yellow" : "white"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideRight
                }
            }

            TabButton {
                property int targetIndex: 5

                text: "Transfer Data"
                font.bold: true
                onClicked: stackLayout.currentIndex = targetIndex

                background: Rectangle {
                    anchors.fill: parent
                    color: stackLayout.currentIndex === parent.targetIndex ? "green" : "#444444"
                }
                contentItem: Text {
                    text: parent.text
                    anchors.centerIn: parent
                    color: stackLayout.currentIndex === parent.targetIndex ? "yellow" : "white"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideRight
                }
            }

            TabButton {
                property int targetIndex: 6

                text: "Developers"
                font.bold: true
                onClicked: stackLayout.currentIndex = targetIndex

                background: Rectangle {
                    anchors.fill: parent
                    color: stackLayout.currentIndex === parent.targetIndex ? "green" : "#444444"
                }
                contentItem: Text {
                    text: parent.text
                    anchors.centerIn: parent
                    color: stackLayout.currentIndex === parent.targetIndex ? "yellow" : "white"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideRight
                }
            }
        }
    }
}
