import QtQuick 6.5
import QtQuick.Controls 6.4

TabButton {
    id: root
    
    // Property to specify which stack index this button controls
    property int targetIndex: 0
    
    // Property to bind to the stackLayout's currentIndex
    property int currentIndex: 0
    
    // Signal emitted when button is clicked, parent should handle updating stackLayout
    signal tabClicked(int index)
    
    // Computed property to determine if this button is active
    readonly property bool isActive: currentIndex === targetIndex
    
    font.bold: true
    
    onClicked: {
        tabClicked(targetIndex)
    }
    
    background: Rectangle {
        anchors.fill: parent
        radius: 4
        // Active: dark green, Non-active: dark blue
        color: root.isActive ? "#2d7a4a" : "#1e3a5f"
        border.width: root.isActive ? 2 : 0
        border.color: root.isActive ? "yellow" : "transparent" // Yellow border when active
    }
    
    contentItem: Text {
        text: root.text
        anchors.centerIn: parent
        // Active: yellow text, Non-active: white text
        color: root.isActive ? "yellow" : "white"
        font.bold: true
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
}
