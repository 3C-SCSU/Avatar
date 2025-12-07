import QtQuick
import QtQuick.Controls
import QtQuick.Layouts 1.15

TabButton {
    property int targetIndex: 0
    required property StackLayout stackLayout
    
    font.bold: true
    onClicked: stackLayout.currentIndex = targetIndex

    background: Rectangle {
        anchors.fill: parent
        color: {
            if (stackLayout.currentIndex === parent.targetIndex) return "green"
            return parent.hovered ? "#1e5f3a" : "#2e3a5c"
        }
    }

    HoverHandler {
        id: hoverer
        acceptedDevices: PointerDevice.Mouse | PointerDevice.TouchPad
        cursorShape: hoverer.hovered ? Qt.PointingHandCursor : Qt.ArrowCursor
    }

    contentItem: Text {
        text: parent.text
        anchors.centerIn: parent
        color: stackLayout.currentIndex === parent.targetIndex ? "yellow" : "white"
        font.bold: true
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
}