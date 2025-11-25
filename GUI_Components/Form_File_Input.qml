import QtQuick.Dialogs
import Qt.labs.platform
import QtQuick 6.5
import QtQuick.Controls 6.5
import QtQuick.Layouts 1.15
import "."

ColumnLayout {
    id: formFileInputContainer
    
    property string labelText: ""
    property string dialogTitle: "Select File"
    property bool selectDirectory: false
    property alias text: fileInput.text
    property alias placeholderText: fileInput.placeholderText
    property alias objectName: fileInput.objectName
    property alias input: fileInput
    property string buttonText: "Browse"
    
    signal fileSelected(string filePath)
    
    spacing: 3
    
    Label {
        text: formFileInputContainer.labelText
        color: "white"
        font.bold: true
        Layout.fillWidth: true
        visible: formFileInputContainer.labelText !== ""
    }
    
    RowLayout {
        Layout.fillWidth: true
        spacing: 10
        
        Form_Input {
            id: fileInput
            labelText: ""
            Layout.fillWidth: true
        }
        
        Button {
            id: browseButton
            text: formFileInputContainer.buttonText
            font.bold: true
            implicitWidth: 120
            implicitHeight: 40
            
            property bool isHovering: false
            
            HoverHandler {
                onHoveredChanged: parent.isHovering = hovered
            }
            
            background: Rectangle {
                color: browseButton.isHovering ? "#3e4e7a" : "#2e3a5c"
                radius: 4
                
                Behavior on color {
                    ColorAnimation {
                        duration: 150
                    }
                }
            }
            
            contentItem: Text {
                text: browseButton.text
                font.pixelSize: 14
                font.bold: true
                color: "white"
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }
            
            onClicked: {
                if (formFileInputContainer.selectDirectory) {
                    folderDialog.open()
                } else {
                    fileDialog.open()
                }
            }
        }
    }
    
    FileDialog {
        id: fileDialog
        title: formFileInputContainer.dialogTitle
        fileMode: FileDialog.OpenFile
        onAccepted: {
            if (fileDialog.file) {
                // Convert QUrl to local file path
                var fileUrl = fileDialog.file.toString()
                // Remove "file://" prefix if present
                var filePath = fileUrl.replace(/^(file:\/{2,3})/, "")
                // Decode URL encoding
                filePath = decodeURIComponent(filePath)
                fileInput.text = filePath
                formFileInputContainer.fileSelected(filePath)
            }
        }
    }
    
    FolderDialog {
        id: folderDialog
        title: formFileInputContainer.dialogTitle
        onAccepted: {
            if (folderDialog.folder) {
                // Convert QUrl to local file path
                var folderUrl = folderDialog.folder.toString()
                // Remove "file://" prefix if present
                var folderPath = folderUrl.replace(/^(file:\/{2,3})/, "")
                // Decode URL encoding
                folderPath = decodeURIComponent(folderPath)
                fileInput.text = folderPath
                formFileInputContainer.fileSelected(folderPath)
            }
        }
    }
}

