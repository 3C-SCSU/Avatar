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
    title: "Avatar - Brainwave Reading"

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
                text: "Brainwave Reading"
                font.bold: true
                onClicked: stackLayout.currentIndex = 0
            }
            TabButton {
                text: "Manual Drone Control"
                font.bold: true
                onClicked: stackLayout.currentIndex = 2
            }
            TabButton {
                text: "Manual NAO Control"
                font.bold: true
                onClicked: {
                    stackLayout.currentIndex = 3
                    console.log("Manual Controller tab clicked")
                    tabController.startNaoViewer()
                }
            }
        }

        // ===== MAIN STACK LAYOUT =====
        StackLayout {
            id: stackLayout
            Layout.fillWidth: true
            Layout.fillHeight: true

            BrainwaveReading { Layout.fillWidth: true; Layout.fillHeight: true }
            BrainwaveVisualization { Layout.fillWidth: true; Layout.fillHeight: true }
            ManualDroneControl { Layout.fillWidth: true; Layout.fillHeight: true }
            ManualNaoControl { Layout.fillWidth: true; Layout.fillHeight: true }
            FileShuffler { Layout.fillWidth: true; Layout.fillHeight: true }
            TransferData { Layout.fillWidth: true; Layout.fillHeight: true }
            Team { Layout.fillWidth: true; Layout.fillHeight: true }
        }

        // ===== BOTTOM TAB BAR =====
        TabBar {
            id: bottomTabBar
            Layout.fillWidth: true
            height: 40
            position: TabBar.Footer

            TabButton {
                text: "Brainwave Visualization"
                font.bold: true
                onClicked: stackLayout.currentIndex = 1
            }
            TabButton {
                text: "File Shuffler"
                font.bold: true
                onClicked: stackLayout.currentIndex = 4
            }
            TabButton {
                text: "Transfer Data"
                font.bold: true
                onClicked: stackLayout.currentIndex = 5
            }
            TabButton {
                text: "Team"
                font.bold: true
                onClicked: stackLayout.currentIndex = 6
            }
        }
    }
}
