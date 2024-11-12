import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

ApplicationWindow {
    visible: true
    width: 640
    height: 480
    color: "#4a5b7b" // Set the background color
    title: "File Shuffler"

    property string outputBoxText: ""

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

            TextArea {
                id: outputBox
                text: outputBoxText
                color: "black"
                anchors.fill: parent
                readOnly: true
            }
        }

        Button {
            text: "Run File Shuffler"
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.top: parent.verticalCenter
            anchors.topMargin: parent.height * 0.3 + 10
            onClicked: {
                runFileShuffler()
            }
        }
    }

    function runFileShuffler() {
        outputBoxText = "Running File Shuffler...\n"
        
        

        var process = Qt.createQmlObject('import QtQuick 2.0; QtObject { signal readyReadStandardOutput; signal readyReadStandardError; signal finished; }', parent);

        process.start("python3", ["../run_file_shuffler.py"]);

        process.readyReadStandardOutput.connect(function() {
            var output = process.readAllStandardOutput();
            outputBoxText += output;
        });

        process.readyReadStandardError.connect(function() {
            var error = process.readAllStandardError();
            outputBoxText += error;
        });

        process.finished.connect(function() {
            outputBoxText += "\nProcess finished";
        });
    }
}
