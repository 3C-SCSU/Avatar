import QtQuick.Dialogs
import Qt.labs.platform
import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtQuick3D 6.7

// Transfer Data view
Rectangle {
    color: "#718399"

    ScrollView {
        anchors.centerIn: parent
        width: Math.min(parent.width * 0.9, 600)
        height: Math.min(parent.height * 0.9, contentHeight)
        clip: true

        ColumnLayout {
            id: contentLayout
            width: parent.width
            spacing: 10

            Label {
                text: "Target IP"
                color: "white"
                font.bold: true
            }

            TextField {
                Layout.fillWidth: true
            }

            Label {
                text: "Target Username"
                color: "white"
                font.bold: true
            }

            TextField {
                Layout.fillWidth: true
            }

            Label {
                text: "Target Password"
                color: "white"
                font.bold: true
            }

            TextField {
                Layout.fillWidth: true
                echoMode: TextInput.Password
            }

            Label {
                text: "Private Key Directory:"
                color: "white"
                font.bold: true
            }

            RowLayout {
                Layout.fillWidth: true

                TextField {
                    id: privateKeyDirInput
                    Layout.fillWidth: true
                }

                Button {
                    text: "Browse"
                    font.bold: true
                    onClicked: console.log("Browse for Private Key Directory")
                }
            }

            CheckBox {
                text: "Ignore Host Key"
                font.bold: true
                checked: true
                contentItem: Text {
                    text: parent.text
                    font.bold: true
                    color: "white"
                    leftPadding: parent.indicator.width + parent.spacing
                }
            }

            Label {
                text: "Source Directory:"
                color: "white"
                font.bold: true
            }

            RowLayout {
                Layout.fillWidth: true

                TextField {
                    id: sourceDirInput
                    Layout.fillWidth: true
                }

                Button {
                    text: "Browse"
                    font.bold: true
                    onClicked: console.log("Browse for Source Directory")
                }
            }

            Label {
                text: "Target Directory:"
                color: "white"
                font.bold: true
            }

            TextField {
                Layout.fillWidth: true
                text: "/home/"
                placeholderText: "/home/"
            }

            RowLayout {
                Layout.fillWidth: true

                Button {
                    text: "Save Config"
                    font.bold: true
                    onClicked: console.log("Save Config clicked")
                }

                Button {
                    text: "Load Config"
                    font.bold: true
                    onClicked: console.log("Load Config clicked")
                }

                Button {
                    text: "Clear Config"
                    font.bold: true
                    onClicked: console.log("Clear Config clicked")
                }

                Button {
                    text: "Upload"
                    font.bold: true
                    onClicked: console.log("Upload clicked")
                }
            }
        }
    }
}
