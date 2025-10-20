import QtQuick.Dialogs
import Qt.labs.platform
import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtQuick3D 6.7

// Team
Rectangle {
    color: "#718399"
    width: 800
    height: 600

    ColumnLayout {
        anchors.fill: parent
        spacing: 20
        anchors.margins: 20

        // Main contributor tier row
        ColumnLayout {
            spacing: 10

            // Titles + Bar Graphs in Columns
            RowLayout {
                Layout.fillWidth: true
                spacing: 10

                // Gold Section
                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.preferredWidth: 1
                    Layout.alignment: Qt.AlignHCenter

                    Text {
                        text: "Gold"
                        color: "yellow"
                        font.bold: true
                        font.pixelSize: 35
                        horizontalAlignment: Text.AlignHCenter
                        Layout.alignment: Qt.AlignHCenter
                    }

                    Rectangle {
                        color: "white"
                        border.color: "#d0d0d8"
                        border.width: 1
                        radius: 4
                        Layout.fillWidth: true
                        Layout.preferredHeight: 300

                        Image {
                            id: goldImage
                            anchors.fill: parent
                            anchors.margins: 10
                            fillMode: Image.PreserveAspectFit
                            source: "HallofFame/commit_tiers_output/gold_contributors.png"
                        }
                    }
                }

                // Silver Section
                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.preferredWidth: 1
                    Layout.alignment: Qt.AlignHCenter

                    Text {
                        text: "Silver"
                        color: "white"
                        font.bold: true
                        font.pixelSize: 35
                        horizontalAlignment: Text.AlignHCenter
                        Layout.alignment: Qt.AlignHCenter
                    }

                    Rectangle {
                        color: "white"
                        border.color: "#d0d0d8"
                        border.width: 1
                        radius: 4
                        Layout.fillWidth: true
                        Layout.preferredHeight: 300

                        Image {
                            id: silverImage
                            anchors.fill: parent
                            anchors.margins: 10
                            fillMode: Image.PreserveAspectFit
                            source: "HallofFame/commit_tiers_output/silver_contributors.png"
                        }
                    }
                }

                // Bronze Section
                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.preferredWidth: 1
                    Layout.alignment: Qt.AlignHCenter

                    Text {
                        text: "Bronze"
                        color: "brown"
                        font.bold: true
                        font.pixelSize: 35
                        horizontalAlignment: Text.AlignHCenter
                        Layout.alignment: Qt.AlignHCenter
                    }

                    Rectangle {
                        color: "white"
                        border.color: "#d0d0d8"
                        border.width: 1
                        radius: 4
                        Layout.fillWidth: true
                        Layout.preferredHeight: 300

                        Image {
                            id: bronzeImage
                            anchors.fill: parent
                            anchors.margins: 10
                            fillMode: Image.PreserveAspectFit
                            source: "HallofFame/commit_tiers_output/bronze_contributors.png"
                        }
                    }
                }
            }

            // Additional container row below bar charts
            RowLayout {
                Layout.fillWidth: true
                spacing: 10

                // Developer List (below Gold)
                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.preferredWidth: 1

                    Text {
                        text: "Developer List"
                        color: "white"
                        font.bold: true
                        font.pixelSize: 24
                        horizontalAlignment: Text.AlignHCenter
                        Layout.alignment: Qt.AlignHCenter
                    }

                    ScrollView {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 150
                        clip: true
                        ScrollBar.vertical.policy: ScrollBar.AlwaysOn

                        TextArea {
                            id: devText
                            text: backend.getDevList()
                            readOnly: true
                            wrapMode: TextArea.Wrap
                            font.pixelSize: 12
                            color: "#000"
                            background: Rectangle {
                                color: "white"
                                radius: 4
                            }
                        }
                    }
                }

                // Tickets by Developer (below Bronze)
                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.preferredWidth: 1

                    Text {
                        text: "Tickets By Developer"
                        color: "white"
                        font.bold: true
                        font.pixelSize: 24
                        horizontalAlignment: Text.AlignHCenter
                        Layout.alignment: Qt.AlignHCenter
                    }

                    ScrollView {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 150
                        clip: true
                        ScrollBar.vertical.policy: ScrollBar.AlwaysOn

                        TextArea {
                            id: ticketText
                            text: backend.getTicketsByDev()
                            readOnly: true
                            wrapMode: TextArea.Wrap
                            font.pixelSize: 12
                            color: "#000"
                            background: Rectangle {
                                color: "white"
                                radius: 4
                            }
                        }
                    }
                }

                // Spacer (keeps layout balanced)
                Item { Layout.fillHeight: true }

                // Refresh Button
                Button {
                    text: "Refresh"
                    font.bold: true
                    implicitWidth: 120
                    implicitHeight: 40
                    Layout.alignment: Qt.AlignHCenter
                    onClicked: {
                        devText.text = backend.getDevList()
                        ticketText.text = backend.getTicketsByDev()
                    }
                }
            }
        }
    }
}
