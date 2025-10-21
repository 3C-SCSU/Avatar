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
    title: "AVATAR - BCI"   // ✅ Updated title
    title: "Avatar - Brainwave Reading"

    ListModel { id: imageModel }

    ColumnLayout {
        anchors.fill: parent
        spacing: 10

        // ===== SINGLE TOP TAB BAR =====
        // ===== TOP TAB BAR =====
        TabBar {
            id: topTabBar
            Layout.fillWidth: true
            height: 45
            position: TabBar.HeaderPosition    // ✅ Tabs always on top
            background: Rectangle { color: "#1E1E1E" }

            // ---- TAB 1 ----
            TabButton {
                text: "Brainwave Reading"
                onClicked: stackLayout.currentIndex = 0

                background: Rectangle {
                    color: stackLayout.currentIndex === 0 ? "green" : "#2E2E2E"
                    radius: 6
                }
                contentItem: Text {
                    text: parent.text
                    color: stackLayout.currentIndex === 0 ? "yellow" : "white"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
            }

            // ---- TAB 2 ----
            TabButton {
                text: "Brainwave Visualization"
                onClicked: stackLayout.currentIndex = 1

                background: Rectangle {
                    color: stackLayout.currentIndex === 1 ? "green" : "#2E2E2E"
                    radius: 6
                }
                contentItem: Text {
                    text: parent.text
                    color: stackLayout.currentIndex === 1 ? "yellow" : "white"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
            }

            // ---- TAB 3 ----
            TabButton {
                text: "Manual Drone Control"
                onClicked: stackLayout.currentIndex = 2

                background: Rectangle {
                    color: stackLayout.currentIndex === 2 ? "green" : "#2E2E2E"
                    radius: 6
                }
                contentItem: Text {
                    text: parent.text
                    color: stackLayout.currentIndex === 2 ? "yellow" : "white"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
            }

            // ---- TAB 4 ----
            TabButton {
                text: "Manual NAO Control"
                onClicked: {
                    stackLayout.currentIndex = 3
                    console.log("Manual Controller tab clicked")
                    tabController.startNaoViewer()
                }

                background: Rectangle {
                    color: stackLayout.currentIndex === 3 ? "green" : "#2E2E2E"
                    radius: 6
                }
                contentItem: Text {
                    text: parent.text
                    color: stackLayout.currentIndex === 3 ? "yellow" : "white"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
            }

            // ---- TAB 5 ----
            TabButton {
                text: "File Shuffler"
                onClicked: stackLayout.currentIndex = 4

                background: Rectangle {
                    color: stackLayout.currentIndex === 4 ? "green" : "#2E2E2E"
                    radius: 6
                }
                contentItem: Text {
                    text: parent.text
                    color: stackLayout.currentIndex === 4 ? "yellow" : "white"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
            }

            // ---- TAB 6 ----
            TabButton {
                text: "Transfer Data"
                onClicked: stackLayout.currentIndex = 5

                background: Rectangle {
                    color: stackLayout.currentIndex === 5 ? "green" : "#2E2E2E"
                    radius: 6
                }
                contentItem: Text {
                    text: parent.text
                    color: stackLayout.currentIndex === 5 ? "yellow" : "white"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
            }

            // ---- TAB 7 ----
            TabButton {
                text: "Developers"
                onClicked: stackLayout.currentIndex = 6

                background: Rectangle {
                    color: stackLayout.currentIndex === 6 ? "green" : "#2E2E2E"
                    radius: 6
                }
                contentItem: Text {
                    text: parent.text
                    color: stackLayout.currentIndex === 6 ? "yellow" : "white"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
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
            Developers { Layout.fillWidth: true; Layout.fillHeight: true }
        }
    }
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
            Developers { Layout.fillWidth: true; Layout.fillHeight: true }
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
                text: "Developers"
                font.bold: true
                onClicked: stackLayout.currentIndex = 6
            }
        }
    }
}
