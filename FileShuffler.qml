import QtQuick.Dialogs
import Qt.labs.platform
import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtQuick3D 6.7

// File shuffler Tab view
Rectangle {
    id: fileShufflerView
    color: "#718399"
    Layout.fillWidth: true
    Layout.fillHeight: true

    // Separate logs for each action
    property string unifyLog: ""
    property string removeLog: ""
    property string runLog: ""

    // Status flags
    property bool ranShuffle: false
    property bool unifiedThoughts: false

    // Which button is currently active ("unify", "remove", "run", or "")
    property string activeButton: ""

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 16

        // HEADER + BUTTON BAR
        Rectangle {
            Layout.alignment: Qt.AlignHCenter
            Layout.preferredWidth: fileShufflerView.width * 0.7
            Layout.preferredHeight: fileShufflerView.height * 0.2
            radius: 10
            color: "#202846"    // dark blue header

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 14
                spacing: 6

                Text {
                    text: "FILE SHUFFLER"
                    color: "white"
                    font.family: "serif"              //serif look
                    font.letterSpacing: 1.0           //
                    font.bold: true
                    font.pixelSize: 24
                    horizontalAlignment: Text.AlignHCenter
                    Layout.alignment: Qt.AlignHCenter
                }

                RowLayout {
                    id: buttonRow
                    Layout.alignment: Qt.AlignHCenter
                    spacing: 14

                    function isActive(name) {
                        return fileShufflerView.activeButton === name;
                    }

                    Button {
                        id: unifyButton
                        Layout.preferredWidth: 180
                        Layout.preferredHeight: 60

                        background: Rectangle {
                            radius: 5
                            color: "#2d7a4a"
                            border.color: buttonRow.isActive("unify") ? "yellow" : "#4a9d6f"
                            border.width: buttonRow.isActive("unify") ? 3 : 1
                        }

                        contentItem: Text {
                            anchors.centerIn: parent
                            text: "Unify Thoughts"
                            font.pixelSize: 18
                            font.bold: true
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                            color: buttonRow.isActive("unify") ? "yellow" : "white"
                        }

                        onClicked: {
                            fileShufflerView.activeButton = "unify"
                            unifyLog = "Running Thoughts Unifier...\n"
                            unifiedThoughts = false
                            ranShuffle = false
                            unifyThoughtsDialog.open()
                        }
                    }

                    Button {
                        id: removeButton
                        Layout.preferredWidth: 180
                        Layout.preferredHeight: 60

                        background: Rectangle {
                            radius: 5
                            color: "#2d7a4a"
                            border.color: buttonRow.isActive("remove") ? "yellow" : "#4a9d6f"
                            border.width: buttonRow.isActive("remove") ? 3 : 1
                        }

                        contentItem: Text {
                            anchors.centerIn: parent
                            text: "Remove 8 Channel"
                            font.pixelSize: 18
                            font.bold: true
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                            color: buttonRow.isActive("remove") ? "yellow" : "white"
                        }

                        onClicked: {
                            fileShufflerView.activeButton = "remove"
                            removeLog = "Running 8 Channel Data Remover...\n"
                            unifiedThoughts = false
                            ranShuffle = false
                            remove8channelDialog.open()
                        }
                    }

                    Button {
                        id: runButton
                        Layout.preferredWidth: 180
                        Layout.preferredHeight: 60

                        background: Rectangle {
                            radius: 5
                            color: "#2d7a4a"
                            border.color: buttonRow.isActive("run") ? "yellow" : "#4a9d6f"
                            border.width: buttonRow.isActive("run") ? 3 : 1
                        }

                        contentItem: Text {
                            anchors.centerIn: parent
                            text: "Anonymize" //Renamed "Run File Shuffler" to "Anonymize"
                            font.pixelSize: 18
                            font.bold: true
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                            color: buttonRow.isActive("run") ? "yellow" : "white"
                        }

                        onClicked: {
                            fileShufflerView.activeButton = "run"
                            runLog = "Running File Shuffler...\n"
                            unifiedThoughts = false
                            ranShuffle = false
                            fileShufflerDialog.open()
                        }
                    }
                }
            }
        }

        // CONSOLE LOG TITLE
        Text {
            text: "Console Log"
            color: "white"
            font.bold: true
            font.pixelSize: 26
            horizontalAlignment: Text.AlignHCenter
            Layout.alignment: Qt.AlignHCenter
        }

        // WRAPPER THAT CONTROLS CONSOLE HEIGHT
        Rectangle {
            id: logsArea
            Layout.alignment: Qt.AlignHCenter
            Layout.preferredWidth: fileShufflerView.width * 0.7
            Layout.preferredHeight: fileShufflerView.height * 0.6
            radius: 4
            border.width: 1
            border.color: "#D0D0D0"
            color: "transparent"

            RowLayout {
                anchors.fill: parent
                anchors.margins: 20
                spacing: 10

                // Panel 1: Unify Thoughts
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    radius: 4
                    color: "white"

                    ScrollView {
                        anchors.fill: parent

                        TextArea {
                            text: fileShufflerView.unifyLog
                            readOnly: true
                            wrapMode: TextArea.Wrap
                            color: "black"
                            font.pixelSize: 13
                            background: null
                        }
                    }
                }

                // Panel 2: Remove 8 Channel
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    radius: 4
                    color: "white"

                    ScrollView {
                        anchors.fill: parent

                        TextArea {
                            text: fileShufflerView.removeLog
                            readOnly: true
                            wrapMode: TextArea.Wrap
                            color: "black"
                            font.pixelSize: 13
                            background: null
                        }
                    }
                }

                // Panel 3: Run Shuffler
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    radius: 4

                    color: "white"

                    ScrollView {
                        anchors.fill: parent

                        TextArea {
                            text: fileShufflerView.runLog
                            readOnly: true
                            wrapMode: TextArea.Wrap
                            color: "black"
                            font.pixelSize: 13
                            background: null
                        }
                    }
                }
            }
        }

        // STATUS TEXTS UNDER LOGS
        Text {
            id: ranText
            text: "Shuffle Complete!"
            color: "yellow"
            font.bold: true
            font.pixelSize: 16
            Layout.alignment: Qt.AlignHCenter
            visible: fileShufflerView.ranShuffle
        }

        Text {
            id: unifiedText
            text: "Thoughts Unified!"
            color: "lightgreen"
            font.bold: true
            font.pixelSize: 16
            Layout.alignment: Qt.AlignHCenter
            visible: fileShufflerView.unifiedThoughts
        }
    }

    //  FOLDER DIALOGS

    // Run File Shuffler - Renamed "Run File Shuffler" to "Anonymize"
    FolderDialog {
        id: fileShufflerDialog
        folder: "file:///"

        onAccepted: {
            console.log("Selected folder:", fileShufflerDialog.folder)
            var output = fileShufflerGui.run_file_shuffler_program(fileShufflerDialog.folder)
            fileShufflerView.runLog += output + "\n"
            fileShufflerView.ranShuffle = true
        }

        onRejected: {
            console.log("Folder dialog canceled")
        }
    }

    // Unify Thoughts
    FolderDialog {
        id: unifyThoughtsDialog
        folder: "file:///"

        onAccepted: {
            console.log("Selected folder:", unifyThoughtsDialog.folder)
            var output = fileShufflerGui.unify_thoughts(unifyThoughtsDialog.folder)
            fileShufflerView.unifyLog += output + "\nThoughts Unified!\n"
            fileShufflerView.unifiedThoughts = true
        }

        onRejected: {
            console.log("Folder dialog canceled")
        }
    }

    // Remove 8 Channel Data
    FolderDialog {
        id: remove8channelDialog
        folder: "file:///"

        onAccepted: {
            console.log("Selected folder:", remove8channelDialog.folder)
            var output = fileShufflerGui.remove_8_channel(remove8channelDialog.folder)
            fileShufflerView.removeLog += output + "\n8 Channel Data Files Removed!\n"
            fileShufflerView.unifiedThoughts = false
            fileShufflerView.ranShuffle = false
        }

        onRejected: {
            console.log("Folder dialog canceled")
        }
    }
}
