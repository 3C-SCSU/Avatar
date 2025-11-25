import QtQuick 6.5
import QtQuick.Controls 6.5
import QtQuick.Layouts 1.15

ColumnLayout {
    id: formInputContainer
    
    property string labelText: ""
    property alias text: formInput.text
    property alias placeholderText: formInput.placeholderText
    property alias echoMode: formInput.echoMode
    property alias objectName: formInput.objectName
    property alias input: formInput
    
    spacing: 3
    
    Label {
        text: formInputContainer.labelText
        color: "white"
        font.bold: true
        Layout.fillWidth: true
        visible: formInputContainer.labelText !== ""
    }
    
    TextField {
        id: formInput
        
        // Styling properties
        color: "white"
        font.pixelSize: 14
        padding: 10
        Layout.fillWidth: true
        
        // Background styling
        background: Rectangle {
            id: bgRect
            color: "#3a4a6a"
            border.color: formInput.activeFocus ? "#4e5e8a" : "#2e3a5c"
            border.width: 1
            radius: 4
            
            Behavior on border.color {
                ColorAnimation {
                    duration: 150
                }
            }
        }
    }
}

