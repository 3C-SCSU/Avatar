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

            NavigationTabButton {
                targetIndex: 0
                text: "Read Brain"
                stackLayout: stackLayout
            }

            NavigationTabButton {
                targetIndex: 2
                text: "Manual Drone Control"
                stackLayout: stackLayout
            }

            NavigationTabButton {
                targetIndex: 3
                text: "Manual NAO Control"
                stackLayout: stackLayout
                onClicked: {
                    console.log("Manual Controller tab clicked")
                    tabController.startNaoViewer()
                }
            }

            NavigationTabButton {
                targetIndex: 7
                text: "Artificial Intelligence"
                stackLayout: stackLayout
                onClicked: {
                    console.log("Artificial Intelligence tab clicked")
                }
            }
        }

        // ===== MAIN STACK LAYOUT =====
        StackLayout {
            id: stackLayout
            Layout.fillWidth: true
            Layout.fillHeight: true

            ReadBrain { Layout.fillWidth: true; Layout.fillHeight: true }
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
            height: 40
            position: TabBar.Footer

            NavigationTabButton {
                targetIndex: 1
                text: "Brainwave Visualization"
                stackLayout: stackLayout
            }

            NavigationTabButton {
                targetIndex: 4
                text: "Shuffler"
                stackLayout: stackLayout
            }

            NavigationTabButton {
                targetIndex: 5
                text: "Cloud Computing"
                stackLayout: stackLayout
            }

            NavigationTabButton {
                targetIndex: 6
                text: "Developers"
                stackLayout: stackLayout
            }
        }
    }
}

