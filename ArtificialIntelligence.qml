import QtQuick.Dialogs
import Qt.labs.platform
import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtQuick3D 6.7

// Artificial Intelligence Tab view 🤡
Rectangle {
    id: artificialIntelligenceView
    color: "#718399"
    Layout.fillWidth: true
    Layout.fillHeight: true

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 20

        Text {
            text: "Machine Learning"
            color: "white"
            font.bold: true
            font.pixelSize: 32
            Layout.alignment: Qt.AlignHCenter
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: "#64778d"
            radius: 8
            border.color: "#4a5d6e"
            border.width: 2

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 20
                spacing: 15

                Text {
                    text: "AI Features Coming Soon"
                    color: "white"
                    font.pixelSize: 20
                    Layout.alignment: Qt.AlignHCenter
                }

                Text {
                    text: "This tab will contain artificial intelligence functionality."
                    color: "#b0c4de"
                    font.pixelSize: 16
                    Layout.alignment: Qt.AlignHCenter
                    wrapMode: Text.WordWrap
                    Layout.maximumWidth: parent.width - 40
                    horizontalAlignment: Text.AlignHCenter
                }
            }
        }
    }
}
