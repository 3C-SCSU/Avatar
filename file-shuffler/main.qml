import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs
import Qt.labs.platform

ApplicationWindow {
    visible: true
    width: 640
    height: 480
    color: "#4a5b7b"
    title: "File Shuffler"

    RowLayout {
        anchors.centerIn: parent
        spacing: 20

      Button {
            text: "Unify Thoughts"
            Layout.alignment: Qt.AlignLeft
            onClicked: unifyThoughts.open()
            }


          Button {
            text: "Run File Shuffler"
            Layout.alignment: Qt.AlignRight
            onClicked: fileDialog.open()
            }



        FileDialog {
            id: fileDialog
            title: "Choose a CSV file to shuffle"
            nameFilters: ["CSV files (*.csv)"]
            fileMode: FileDialog.OpenFile
            visible: false

            onAccepted: {
                if (fileDialog.selectedFile) {
                    console.log("File selected:", fileDialog.selectedFile)
                    fileShufflerGui.shuffle_csv_file(fileDialog.selectedFile)
                } else {
                    console.log("No file returned")
                }
            }

            onRejected: {
                console.log("File dialog canceled")
            }
        }

        FolderDialog {
            id: unifyThoughts
             folder: "file:///"  // Or "." for current working directory

            onAccepted: {
             console.log("Selected folder:", unifyThoughts.folder)
                fileShufflerGui.unify_thoughts(unifyThoughts.folder)
            }

             onRejected: {
                console.log("Folder dialog canceled")
            }

        }
    }
}



