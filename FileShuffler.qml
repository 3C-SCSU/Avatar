import QtQuick.Dialogs
import Qt.labs.platform
import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtQuick3D 6.7

// File shuffler view
Rectangle {
    id: fileShufflerView
    color: "#718399"
    Layout.fillWidth: true
    Layout.fillHeight: true

    property string outputBoxText: ""
    property string selectedDirectory: ""
    property bool ranShuffle: false
    property bool unifiedThoughts: false

    Column {
        anchors.fill: parent
        spacing: 10

        Text {
            text: "File Shuffler"
            color: "white"
            font.bold: true
            font.pixelSize: 24
            horizontalAlignment: Text.AlignHCenter
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.top: parent.top
            anchors.topMargin: 20
        }

        Rectangle {
            width: parent.width * 0.6
            height: parent.height * 0.6
            color: "lightgrey"
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.verticalCenter: parent.verticalCenter

            ScrollView {
                anchors.fill: parent

                TextArea {
                    id: outputBox
                    text: fileShufflerView.outputBoxText
                    color: "black"
                    readOnly: true
                    width: parent.width
                }
            }
        }

        Row {
            id: buttonRow
            spacing: 20
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.top: parent.verticalCenter
            anchors.topMargin: parent.height * 0.3 + 10

            Button {
                text: "Unify Thoughts"
                onClicked: {
                    unifyThoughts.open()
                    fileShufflerView.outputBoxText = `Running Thoughts Unifier...\n`
                    fileShufflerView.unifiedThoughts = false
                }
            }

            Button {
                text: "Remove 8 Channel Data"
                onClicked: {
                    remove8channelDialog.open()
                    fileShufflerView.outputBoxText = `Running 8 Channel Data Remover...\n`
                    fileShufflerView.unifiedThoughts = false
                }
            }

            Button {
                text: "Run File Shuffler"
                onClicked: {
                    fileShuffler.open()
                    fileShufflerView.outputBoxText = `Running File Shuffler...\n`
                    fileShufflerView.ranShuffle = false
                }
            }
        }

        FolderDialog {
            id: fileShuffler
            folder: "file:///"
            visible: false

            onAccepted: {
                console.log("Selected folder:", fileShuffler.folder)
                fileShufflerGui.run_file_shuffler_program(fileShuffler.folder)
                fileShufflerView.ranShuffle = true
                var output = fileShufflerGui.run_file_shuffler_program(fileShufflerView.folder)
                fileShufflerView.outputBoxText += output
            }

            onRejected: {
                console.log("Folder dialog canceled")
            }
        }

        Text {
            id: ranText
            text: "Shuffle Complete!"
            color: "yellow"
            font.bold: true
            font.pixelSize: 18
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.top: buttonRow.bottom
            anchors.topMargin: 10
            visible: fileShufflerView.ranShuffle
        }

        Text {
            id: unifiedThoughts
            text: "Thoughts Unified!"
            color: "lightgreen"
            font.bold: true
            font.pixelSize: 18
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.top: ranText.bottom
            anchors.topMargin: 10
            visible: fileShufflerView.unifiedThoughts
        }
    }

    FolderDialog {
        id: unifyThoughts
        folder: "file:///"

        onAccepted: {
            console.log("Selected folder:", unifyThoughts.folder)
            fileShufflerGui.unify_thoughts(unifyThoughts.folder)
            fileShufflerView.unifiedThoughts = true
            var outputt = fileShufflerGui.unify_thoughts(unifyThoughts.folder)
            fileShufflerView.outputBoxText += outputt
            fileShufflerView.outputBoxText += "\nThoughts Unified!\n"
        }

        onRejected: {
            console.log("Folder dialog canceled")
        }
    }

    FolderDialog {
        id: remove8channelDialog
        folder: "file:///"

        onAccepted: {
            console.log("Selected folder:", remove8channelDialog.folder)
            fileShufflerGui.outputBoxText = "Running 8 Channel Data Remover...\n"
            var output = fileShufflerGui.remove_8_channel(remove8channelDialog.folder)
            fileShufflerView.outputBoxText += output
            fileShufflerView.unifiedThoughts = false
            fileShufflerView.ranShuffle = false
            fileShufflerView.outputBoxText += "\n8 Channel Data Files Removed!\n"
        }

        onRejected: {
            console.log("Folder dialog canceled")
        }
    }
}
