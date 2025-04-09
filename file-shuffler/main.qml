import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs

ApplicationWindow {
    visible: true
    width: 640
    height: 480
    color: "#4a5b7b"
    title: "File Shuffler"

    ColumnLayout {
        anchors.centerIn: parent
        spacing: 20

        Button {
            text: "Select CSV File"
            Layout.alignment: Qt.AlignHCenter
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
    }
}

