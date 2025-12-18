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
        color: stackLayout.currentIndex === parent.targetIndex ? "green" : "#242c4d"
        border.color: stackLayout.currentIndex === parent.targetIndex ? "yellow" : "transparent"
        border.width: stackLayout.currentIndex === parent.targetIndex ? 3 : 1
    }

    MouseArea {
        id: hoverArea
        anchors.fill: parent
        hoverEnabled: true
        acceptedButtons: Qt.NoButton
        cursorShape: hoverArea.containsMouse ? Qt.PointingHandCursor : Qt.ArrowCursor
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
