import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtQuick3D 6.7
import QtQuick.Dialogs
import Qt.labs.platform
import "GUI-Components"

ApplicationWindow {
    visible: true
    width: 1200
    height: 800
    title: "Avatar - BCI"

    ListModel { id: imageModel }

    ColumnLayout {
        anchors.fill: parent
        spacing: 5

        // ===== TOP TAB BAR =====
        TabBar {
            id: topTabBar
            Layout.fillWidth: true
            height: 40

            TabButton {
                text: "Read Brain"
                targetIndex: 0
                currentIndex: stackLayout.currentIndex
                onTabClicked: (index) => stackLayout.currentIndex = index
            }

            TabButton {
                text: "Manual Drone Control"
                targetIndex: 2
                currentIndex: stackLayout.currentIndex
                onTabClicked: (index) => stackLayout.currentIndex = index
            }

            TabButton {
                text: "Manual NAO Control"
                targetIndex: 3
                currentIndex: stackLayout.currentIndex
                onTabClicked: (index) => {
                    stackLayout.currentIndex = index
                    console.log("Manual Controller tab clicked")
                    tabController.startNaoViewer()
                }
            }

            TabButton {
                text: "Artificial Intelligence"
                targetIndex: 7
                currentIndex: stackLayout.currentIndex
                onTabClicked: (index) => {
                    stackLayout.currentIndex = index
                    console.log("Artificial Intelligence tab clicked")
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
            CloudComputing { Layout.fillWidth: true; Layout.fillHeight: true } // Renamed Transfer Data to Cloud Computing 
            Developers { Layout.fillWidth: true; Layout.fillHeight: true }
            ArtificialIntelligence { Layout.fillWidth: true; Layout.fillHeight: true }
        }

        // ===== BOTTOM TAB BAR =====
        TabBar {
            id: bottomTabBar
            Layout.fillWidth: true
            position: TabBar.Footer

            TabButton {
                text: "Brainwave Visualization"
                targetIndex: 1
                currentIndex: stackLayout.currentIndex
                onTabClicked: (index) => stackLayout.currentIndex = index
            }

            TabButton {
                text: "File Shuffler"
                targetIndex: 4
                currentIndex: stackLayout.currentIndex
                onTabClicked: (index) => stackLayout.currentIndex = index
            }

            TabButton {
                text: "Cloud Computing" // Renamed Transfer Data to Cloud Computing 
                targetIndex: 5
                currentIndex: stackLayout.currentIndex
                onTabClicked: (index) => stackLayout.currentIndex = index
            }

            TabButton {
                text: "Developers"
                targetIndex: 6
                currentIndex: stackLayout.currentIndex
                onTabClicked: (index) => stackLayout.currentIndex = index
            }
        }
    }
}
