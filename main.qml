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
                  id: tabRect
                  anchors.fill: parent
                  radius: 4
                  border.color: stackLayout.currentIndex === parent.targetIndex ? "#1E90FF" : "#898989"
                  border.width: 1

                  // Defines diagonal gradient (Darker Upper-Left -> Lighter Lower-Right)
                  gradient: Gradient {
                      // Gradient starts at the upper-left corner
                      start: Qt.point(0, 0)
                      // Gradient ends at the lower-right corner
                      end: Qt.point(tabRect.width, tabRect.height)

                      // Gradient Stop 1 (Upper-Left - position 0.0)
                      GradientStop {
                          position: 0.0
                          // Darker color on the start corner
                          color: stackLayout.currentIndex === parent.targetIndex ? "#4682B4" : "#F0F8FF" 
                      }
                      // Gradient Stop 2 (Lower-Right - position 1.0)
                      GradientStop {
                          position: 1.0
                          // Lighter color on the end corner
                          color: stackLayout.currentIndex === parent.targetIndex ? "#6699CC" : "#F8F8FF" 
                      }
                  }
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
                  id: tabRect
                  anchors.fill: parent
                  radius: 4
                  border.color: stackLayout.currentIndex === parent.targetIndex ? "#1E90FF" : "#898989"
                  border.width: 1

                  // Defines diagonal gradient (Darker Upper-Left -> Lighter Lower-Right)
                  gradient: Gradient {
                      // Gradient starts at the upper-left corner
                      start: Qt.point(0, 0)
                      // Gradient ends at the lower-right corner
                      end: Qt.point(tabRect.width, tabRect.height)

                      // Gradient Stop 1 (Upper-Left - position 0.0)
                      GradientStop {
                          position: 0.0
                          // Darker color on the start corner
                          color: stackLayout.currentIndex === parent.targetIndex ? "#4682B4" : "#F0F8FF" 
                      }
                      // Gradient Stop 2 (Lower-Right - position 1.0)
                      GradientStop {
                          position: 1.0
                          // Lighter color on the end corner
                          color: stackLayout.currentIndex === parent.targetIndex ? "#6699CC" : "#F8F8FF" 
                      }
                  }
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
                  id: tabRect
                  anchors.fill: parent
                  radius: 4
                  border.color: stackLayout.currentIndex === parent.targetIndex ? "#1E90FF" : "#898989"
                  border.width: 1

                  // Defines diagonal gradient (Darker Upper-Left -> Lighter Lower-Right)
                  gradient: Gradient {
                      // Gradient starts at the upper-left corner
                      start: Qt.point(0, 0)
                      // Gradient ends at the lower-right corner
                      end: Qt.point(tabRect.width, tabRect.height)

                      // Gradient Stop 1 (Upper-Left - position 0.0)
                      GradientStop {
                          position: 0.0
                          // Darker color on the start corner
                          color: stackLayout.currentIndex === parent.targetIndex ? "#4682B4" : "#F0F8FF" 
                      }
                      // Gradient Stop 2 (Lower-Right - position 1.0)
                      GradientStop {
                          position: 1.0
                          // Lighter color on the end corner
                          color: stackLayout.currentIndex === parent.targetIndex ? "#6699CC" : "#F8F8FF" 
                      }
                  }
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
                  id: tabRect
                  anchors.fill: parent
                  radius: 4
                  border.color: stackLayout.currentIndex === parent.targetIndex ? "#1E90FF" : "#898989"
                  border.width: 1

                  // Defines diagonal gradient (Darker Upper-Left -> Lighter Lower-Right)
                  gradient: Gradient {
                      // Gradient starts at the upper-left corner
                      start: Qt.point(0, 0)
                      // Gradient ends at the lower-right corner
                      end: Qt.point(tabRect.width, tabRect.height)

                      // Gradient Stop 1 (Upper-Left - position 0.0)
                      GradientStop {
                          position: 0.0
                          // Darker color on the start corner
                          color: stackLayout.currentIndex === parent.targetIndex ? "#4682B4" : "#F0F8FF" 
                      }
                      // Gradient Stop 2 (Lower-Right - position 1.0)
                      GradientStop {
                          position: 1.0
                          // Lighter color on the end corner
                          color: stackLayout.currentIndex === parent.targetIndex ? "#6699CC" : "#F8F8FF" 
                      }
                  }
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
                property int targetIndex: 1

                text: "Brainwave Visualization"
                font.bold: true
                onClicked: stackLayout.currentIndex = targetIndex

                background: Rectangle {
                  id: tabRect
                  anchors.fill: parent
                  radius: 4
                  border.color: stackLayout.currentIndex === parent.targetIndex ? "#1E90FF" : "#898989"
                  border.width: 1

                  // Defines diagonal gradient (Darker Upper-Left -> Lighter Lower-Right)
                  gradient: Gradient {
                      // Gradient starts at the upper-left corner
                      start: Qt.point(0, 0)
                      // Gradient ends at the lower-right corner
                      end: Qt.point(tabRect.width, tabRect.height)

                      // Gradient Stop 1 (Upper-Left - position 0.0)
                      GradientStop {
                          position: 0.0
                          // Darker color on the start corner
                          color: stackLayout.currentIndex === parent.targetIndex ? "#4682B4" : "#F0F8FF" 
                      }
                      // Gradient Stop 2 (Lower-Right - position 1.0)
                      GradientStop {
                          position: 1.0
                          // Lighter color on the end corner
                          color: stackLayout.currentIndex === parent.targetIndex ? "#6699CC" : "#F8F8FF" 
                      }
                  }
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

                text: "Shuffler"
                font.bold: true
                onClicked: stackLayout.currentIndex = targetIndex

                background: Rectangle {
                  id: tabRect
                  anchors.fill: parent
                  radius: 4
                  border.color: stackLayout.currentIndex === parent.targetIndex ? "#1E90FF" : "#898989"
                  border.width: 1

                  // Defines diagonal gradient (Darker Upper-Left -> Lighter Lower-Right)
                  gradient: Gradient {
                      // Gradient starts at the upper-left corner
                      start: Qt.point(0, 0)
                      // Gradient ends at the lower-right corner
                      end: Qt.point(tabRect.width, tabRect.height)

                      // Gradient Stop 1 (Upper-Left - position 0.0)
                      GradientStop {
                          position: 0.0
                          // Darker color on the start corner
                          color: stackLayout.currentIndex === parent.targetIndex ? "#4682B4" : "#F0F8FF" 
                      }
                      // Gradient Stop 2 (Lower-Right - position 1.0)
                      GradientStop {
                          position: 1.0
                          // Lighter color on the end corner
                          color: stackLayout.currentIndex === parent.targetIndex ? "#6699CC" : "#F8F8FF" 
                      }
                  }
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

                text: "Cloud Computing" // Renamed Transfer Data to Cloud Computing 
                font.bold: true
                onClicked: stackLayout.currentIndex = targetIndex

                background: Rectangle {
                  id: cloudTabRect
                  anchors.fill: parent
                  radius: 4
                  border.color: stackLayout.currentIndex === parent.targetIndex ? "#1E90FF" : "#898989"
                  border.width: 1

                  // Defines diagonal gradient (Darker Upper-Left -> Lighter Lower-Right)
                  gradient: Gradient {
                      // Gradient starts at the upper-left corner
                      start: Qt.point(0, 0)
                      // Gradient ends at the lower-right corner
                      end: Qt.point(cloudTabRect.width, cloudTabRect.height)

                      // Gradient Stop 1 (Upper-Left - position 0.0)
                      GradientStop {
                          position: 0.0
                          // Darker color on the start corner
                          color: stackLayout.currentIndex === parent.targetIndex ? "#4682B4" : "#F0F8FF" 
                      }
                      // Gradient Stop 2 (Lower-Right - position 1.0)
                      GradientStop {
                          position: 1.0
                          // Lighter color on the end corner
                          color: stackLayout.currentIndex === parent.targetIndex ? "#6699CC" : "#F8F8FF" 
                      }
                  }
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
                  id: tabRect
                  anchors.fill: parent
                  radius: 4
                  border.color: stackLayout.currentIndex === parent.targetIndex ? "#1E90FF" : "#898989"
                  border.width: 1

                  // Defines diagonal gradient (Darker Upper-Left -> Lighter Lower-Right)
                  gradient: Gradient {
                      // Gradient starts at the upper-left corner
                      start: Qt.point(0, 0)
                      // Gradient ends at the lower-right corner
                      end: Qt.point(tabRect.width, tabRect.height)

                      // Gradient Stop 1 (Upper-Left - position 0.0)
                      GradientStop {
                          position: 0.0
                          // Darker color on the start corner
                          color: stackLayout.currentIndex === parent.targetIndex ? "#4682B4" : "#F0F8FF" 
                      }
                      // Gradient Stop 2 (Lower-Right - position 1.0)
                      GradientStop {
                          position: 1.0
                          // Lighter color on the end corner
                          color: stackLayout.currentIndex === parent.targetIndex ? "#6699CC" : "#F8F8FF" 
                      }
                  }
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

