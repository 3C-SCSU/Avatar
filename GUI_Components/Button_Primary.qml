import QtQuick 6.5
import QtQuick.Controls 6.5 as Controls

Controls.Button {
    id: button
    
    property bool isHovering: false
    
    implicitWidth: 120
    implicitHeight: 40
    font.bold: true
    
    HoverHandler {
        onHoveredChanged: button.isHovering = hovered
    }
    
    background: Rectangle {
        color: button.isHovering ? "#3e4e7a" : "#2e3a5c"
        radius: 4
        
        Behavior on color {
            ColorAnimation {
                duration: 150
            }
        }
    }
    
    contentItem: Text {
        text: button.text
        font.pixelSize: 14
        font.bold: true
        color: "white"
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
    }
}

