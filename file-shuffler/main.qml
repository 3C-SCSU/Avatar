import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs

ApplicationWindow {
    visible: true
    width: 640
    height: 480
    color: "#4a5b7b" // Set the background color
    title: "File Shuffler"

    property string outputBoxText: ""
    property bool ranShuffle: false 

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
                    text: outputBoxText
                    color: "black"
                    readOnly: true
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
                id: folderButton
                text: "Select your Directory"
                onClicked: myFolderDialog.open()
            }

            Button {
                id: runButton
                text: "Run File Shuffler"
                onClicked: {
                    ranShuffle = true; 
                    outputBoxText = "Running File Shuffler...\n";
                    var output = fileShufflerGui.run_file_shuffler_program();
                    outputBoxText += output;
                }
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
            visible: ranShuffle 
        }
    }

    FolderDialog {
        id: myFolderDialog
        title: "Select Your Directory"
        onAccepted:
        {
            console.log(myFolderDialog.selectedFolder)
        }
    }
}
