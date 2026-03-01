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
                id: readBrainTab
                // this tab shows page index 0
                property int targetIndex: 0

                text: "Read Brain"
                font.bold: true
                onClicked: stackLayout.currentIndex = targetIndex

                background: Rectangle {
                  anchors.fill: parent
                  gradient: Gradient {
                    GradientStop {
                      position: 0.0
                      color: stackLayout.currentIndex === readBrainTab.targetIndex ? "#00cc44" : "#2a3456"
                    }
                    GradientStop {
                      position: 1.0
                      color: stackLayout.currentIndex === readBrainTab.targetIndex ? "#008833" : "#1a2440"
                    }
                  }
                  border.color: stackLayout.currentIndex === readBrainTab.targetIndex ? "yellow" : "transparent"
                  border.width: stackLayout.currentIndex === readBrainTab.targetIndex ? 3 : 1
                  radius: 5
                }
                contentItem: Text {
                    text: parent.text
                    anchors.centerIn: parent
                    color: stackLayout.currentIndex === readBrainTab.targetIndex ? "yellow" : "white"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideRight
                }
            }

            TabButton {
                id: manualDroneTab
                property int targetIndex: 2

                text: "Manual Drone Control"
                font.bold: true
                onClicked: stackLayout.currentIndex = targetIndex

                background: Rectangle {
                  anchors.fill: parent
                  gradient: Gradient {
                    GradientStop {
                      position: 0.0
                      color: stackLayout.currentIndex === manualDroneTab.targetIndex ? "#00cc44" : "#2a3456"
                    }
                    GradientStop {
                      position: 1.0
                      color: stackLayout.currentIndex === manualDroneTab.targetIndex ? "#008833" : "#1a2440"
                    }
                  }
                  border.color: stackLayout.currentIndex === manualDroneTab.targetIndex ? "yellow" : "transparent"
                  border.width: stackLayout.currentIndex === manualDroneTab.targetIndex ? 3 : 1
                  radius: 5
                }
                contentItem: Text {
                    text: parent.text
                    anchors.centerIn: parent
                    color: stackLayout.currentIndex === manualDroneTab.targetIndex ? "yellow" : "white"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideRight
                }
            }

            TabButton {
                id: manualNaoTab
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
                  gradient: Gradient {
                    GradientStop {
                      position: 0.0
                      color: stackLayout.currentIndex === manualNaoTab.targetIndex ? "#00cc44" : "#2a3456"
                    }
                    GradientStop {
                      position: 1.0
                      color: stackLayout.currentIndex === manualNaoTab.targetIndex ? "#008833" : "#1a2440"
                    }
                  }
                  border.color: stackLayout.currentIndex === manualNaoTab.targetIndex ? "yellow" : "transparent"
                  border.width: stackLayout.currentIndex === manualNaoTab.targetIndex ? 3 : 1
                  radius: 5
                }
                contentItem: Text {
                    text: parent.text
                    anchors.centerIn: parent
                    color: stackLayout.currentIndex === manualNaoTab.targetIndex ? "yellow" : "white"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideRight
                }
            }

            TabButton {
                id: aiTab
                property int targetIndex: 7

                text: "Artificial Intelligence"
                font.bold: true
                onClicked: {
                    stackLayout.currentIndex = targetIndex
                    console.log("Artificial Intelligence tab clicked")
                }

                background: Rectangle {
                  anchors.fill: parent
                  gradient: Gradient {
                    GradientStop {
                      position: 0.0
                      color: stackLayout.currentIndex === aiTab.targetIndex ? "#00cc44" : "#2a3456"
                    }
                    GradientStop {
                      position: 1.0
                      color: stackLayout.currentIndex === aiTab.targetIndex ? "#008833" : "#1a2440"
                    }
                  }
                  border.color: stackLayout.currentIndex === aiTab.targetIndex ? "yellow" : "transparent"
                  border.width: stackLayout.currentIndex === aiTab.targetIndex ? 3 : 1
                  radius: 5
                }
                contentItem: Text {
                    text: parent.text
                    anchors.centerIn: parent
                    color: stackLayout.currentIndex === aiTab.targetIndex ? "yellow" : "white"
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

            TabButton {
                id: brainwaveVizTab
                property int targetIndex: 1

                text: "Brainwave Visualization"
                font.bold: true
                onClicked: stackLayout.currentIndex = targetIndex

                background: Rectangle {
                  anchors.fill: parent
                  gradient: Gradient {
                    GradientStop {
                      position: 0.0
                      color: stackLayout.currentIndex === brainwaveVizTab.targetIndex ? "#00cc44" : "#2a3456"
                    }
                    GradientStop {
                      position: 1.0
                      color: stackLayout.currentIndex === brainwaveVizTab.targetIndex ? "#008833" : "#1a2440"
                    }
                  }
                  border.color: stackLayout.currentIndex === brainwaveVizTab.targetIndex ? "yellow" : "transparent"
                  border.width: stackLayout.currentIndex === brainwaveVizTab.targetIndex ? 3 : 1
                  radius: 5
                }
                contentItem: Text {
                    text: parent.text
                    anchors.centerIn: parent
                    color: stackLayout.currentIndex === brainwaveVizTab.targetIndex ? "yellow" : "white"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideRight
                }
            }

            TabButton {
                id: shufflerTab
                property int targetIndex: 4

                text: "Shuffler"
                font.bold: true
                onClicked: stackLayout.currentIndex = targetIndex

                background: Rectangle {
                  anchors.fill: parent
                  gradient: Gradient {
                    GradientStop {
                      position: 0.0
                      color: stackLayout.currentIndex === shufflerTab.targetIndex ? "#00cc44" : "#2a3456"
                    }
                    GradientStop {
                      position: 1.0
                      color: stackLayout.currentIndex === shufflerTab.targetIndex ? "#008833" : "#1a2440"
                    }
                  }
                  border.color: stackLayout.currentIndex === shufflerTab.targetIndex ? "yellow" : "transparent"
                  border.width: stackLayout.currentIndex === shufflerTab.targetIndex ? 3 : 1
                  radius: 5
                }
                contentItem: Text {
                    text: parent.text
                    anchors.centerIn: parent
                    color: stackLayout.currentIndex === shufflerTab.targetIndex ? "yellow" : "white"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideRight
                }
            }

            TabButton {
                id: cloudTab
                property int targetIndex: 5

                text: "Cloud Computing" // Renamed Transfer Data to Cloud Computing 
                font.bold: true
                onClicked: stackLayout.currentIndex = targetIndex

                background: Rectangle {
                  anchors.fill: parent
                  gradient: Gradient {
                    GradientStop {
                      position: 0.0
                      color: stackLayout.currentIndex === cloudTab.targetIndex ? "#00cc44" : "#2a3456"
                    }
                    GradientStop {
                      position: 1.0
                      color: stackLayout.currentIndex === cloudTab.targetIndex ? "#008833" : "#1a2440"
                    }
                  }
                  border.color: stackLayout.currentIndex === cloudTab.targetIndex ? "yellow" : "transparent"
                  border.width: stackLayout.currentIndex === cloudTab.targetIndex ? 3 : 1
                  radius: 5
                }
                contentItem: Text {
                    text: parent.text
                    anchors.centerIn: parent
                    color: stackLayout.currentIndex === cloudTab.targetIndex ? "yellow" : "white"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideRight
                }
            }

            TabButton {
                id: developersTab
                property int targetIndex: 6

                text: "Developers"
                font.bold: true
                onClicked: stackLayout.currentIndex = targetIndex

                background: Rectangle {
                  anchors.fill: parent
                  gradient: Gradient {
                    GradientStop {
                      position: 0.0
                      color: stackLayout.currentIndex === developersTab.targetIndex ? "#00cc44" : "#2a3456"
                    }
                    GradientStop {
                      position: 1.0
                      color: stackLayout.currentIndex === developersTab.targetIndex ? "#008833" : "#1a2440"
                    }
                  }
                  border.color: stackLayout.currentIndex === developersTab.targetIndex ? "yellow" : "transparent"
                  border.width: stackLayout.currentIndex === developersTab.targetIndex ? 3 : 1
                  radius: 5
                }
                contentItem: Text {
                    text: parent.text
                    anchors.centerIn: parent
                    color: stackLayout.currentIndex === developersTab.targetIndex ? "yellow" : "white"
                    font.bold: true
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideRight
                }
            }
        }
    }
}

