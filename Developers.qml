import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15

Rectangle {
    color: "#718399"
    width: 1100
    height: 700

    ColumnLayout {
        anchors.fill: parent
        spacing: 20         
        anchors.margins: 10

        // ===================== TOP SECTION =====================
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.preferredHeight: parent.height / 2   // 🔹 Top half of the window
            spacing: 15

            // ---------- GOLD PANEL ----------
            Rectangle {
                color: "#1A2B48"
                radius: 10
                border.color: "white"
                Layout.fillWidth: true
                Layout.fillHeight: true

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 8

                    Text {
                        text: "Golden"
                        color: "white"
                        font.bold: true
                        font.pixelSize: 26
                        horizontalAlignment: Text.AlignHCenter
                        Layout.alignment: Qt.AlignHCenter
                    }

                    Rectangle {
                        color: "white"
                        radius: 8
                        Layout.fillWidth: true
                        Layout.fillHeight: true

                        Image {
                            anchors.fill: parent
                            anchors.margins: 10
                            fillMode: Image.PreserveAspectFit
                            source: "Avatar/plotsDevelopers/gold_contributors.png"
                        }

                        Text {
                            text: "Top developer"
                            color: "red"
                            font.bold: true
                            font.pixelSize: 18
                            anchors.horizontalCenter: parent.horizontalCenter
                            anchors.top: parent.top
                            anchors.topMargin: 8
                        }
                    }
                }
            }

            // ---------- SILVER PANEL ----------
            Rectangle {
                color: "#1A2B48"
                radius: 10
                border.color: "white"
                Layout.fillWidth: true
                Layout.fillHeight: true

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 8

                    Text {
                        text: "Silver"
                        color: "white"
                        font.bold: true
                        font.pixelSize: 26
                        horizontalAlignment: Text.AlignHCenter
                        Layout.alignment: Qt.AlignHCenter
                    }

                    Rectangle {
                        color: "white"
                        radius: 8
                        Layout.fillWidth: true
                        Layout.fillHeight: true

                        Image {
                            anchors.fill: parent
                            anchors.margins: 10
                            fillMode: Image.PreserveAspectFit
                            source: "Avatar/plotsDevelopers/silver_contributors.png"
                        }

                        Text {
                            text: "Top developer"
                            color: "red"
                            font.bold: true
                            font.pixelSize: 18
                            anchors.horizontalCenter: parent.horizontalCenter
                            anchors.top: parent.top
                            anchors.topMargin: 8
                        }
                    }
                }
            }

            // ---------- BRONZE PANEL ----------
            Rectangle {
                color: "#1A2B48"
                radius: 10
                border.color: "white"
                Layout.fillWidth: true
                Layout.fillHeight: true

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 8

                    Text {
                        text: "Bronze"
                        color: "white"
                        font.bold: true
                        font.pixelSize: 26
                        horizontalAlignment: Text.AlignHCenter
                        Layout.alignment: Qt.AlignHCenter
                    }

                    Rectangle {
                        color: "white"
                        radius: 8
                        Layout.fillWidth: true
                        Layout.fillHeight: true

                        Image {
                            anchors.fill: parent
                            anchors.margins: 10
                            fillMode: Image.PreserveAspectFit
                            source: "Avatar/plotsDevelopers/bronze_contributors.png"
                        }

                        Text {
                            text: "Top developer"
                            color: "red"
                            font.bold: true
                            font.pixelSize: 18
                            anchors.horizontalCenter: parent.horizontalCenter
                            anchors.top: parent.top
                            anchors.topMargin: 8
                        }
                    }
                }
            }
        }

        // ===================== BOTTOM SECTION =====================
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.preferredHeight: parent.height / 2    // Bottom half of the window
            spacing: 10

            // ---------- DEVELOPERS LOG ----------
            Rectangle {
                color: "#1A2B48"
                radius: 10
                border.color: "white"
                Layout.fillWidth: true
                Layout.fillHeight: true

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 8

                    Text {
                        text: "Developers List"
                        color: "white"
                        font.bold: true
                        font.pixelSize: 22
                        horizontalAlignment: Text.AlignHCenter
                        Layout.alignment: Qt.AlignHCenter
                    }

                    ScrollView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        ScrollBar.vertical.policy: ScrollBar.AlwaysOn

                        TextArea {
                            id: devText
                            text: "Console output here..."
                            readOnly: true
                            wrapMode: TextArea.Wrap
                            font.pixelSize: 12
                            color: "black"
                            background: Rectangle { color: "white"; radius: 6 }
                        }
                    }
                }
            }

            // ---------- MEDAL + REFRESH ----------
            Rectangle {
                color: "#718399"
                Layout.preferredWidth: 230
                Layout.fillHeight: true
                radius: 10
                border.color: "transparent"

                Column {
                    anchors.centerIn: parent
                    spacing: 1   // minimal space between image and button

                    // Medal Image
                    Image {
                        source: "Avatar/plotsDevelopers/Medal.png"
                        width: 225
                        height: 325
                        fillMode: Image.PreserveAspectFit
                        smooth: true
                        antialiasing: true
                        anchors.horizontalCenter: parent.horizontalCenter
                    }

                    Button {
                        text: "Refresh"
                        font.bold: true
                        width: 200
                        height: 50
                        anchors.horizontalCenter: parent.horizontalCenter

                        background: Rectangle {
                            color: "#003366"
                            radius: 5
                            border.color: "#001F3F"
                        }

                        contentItem: Text {
                            text: qsTr("Refresh")
                            font.bold: true
                            color: "white"
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }

                        onClicked: {
                            devText.text = backend.getDevList()
                            ticketText.text = backend.getTicketsByDev()
                        }
                    }
                }
            }


            // ---------- TICKET BY DEVELOPER LOG ----------
            Rectangle {
                color: "#1A2B48"
                radius: 10
                border.color: "white"
                Layout.fillWidth: true
                Layout.fillHeight: true

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 8

                    Text {
                        text: "Ticket by Developer Log"
                        color: "white"
                        font.bold: true
                        font.pixelSize: 22
                        horizontalAlignment: Text.AlignHCenter
                        Layout.alignment: Qt.AlignHCenter
                    }

                    ScrollView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        ScrollBar.vertical.policy: ScrollBar.AlwaysOn

                        TextArea {
                            id: ticketText
                            text: "Console output here..."
                            readOnly: true
                            wrapMode: TextArea.Wrap
                            font.pixelSize: 12
                            color: "black"
                            background: Rectangle { color: "white"; radius: 6 }
                        }
                    }
                }
            }
        }
    }
}
