import QtQuick
import QtQuick.Controls
import QtQuick.Layouts 1.15

TabButton {
    property int targetIndex: 0
    required property StackLayout stackLayout
    
    implicitHeight: 24
    font.bold: true
    onClicked: stackLayout.currentIndex = targetIndex

    background: Rectangle {
        anchors.fill: parent

        // active/inactive helpers
        property bool isActive: stackLayout.currentIndex === parent.targetIndex
        property color inactiveColor: parent.hovered ? "#1e5f3a" : "#242c4d"

        // predeclare gradient and toggle via binding
        Gradient {
            id: hoverGradient
            orientation: Gradient.Horizontal
            GradientStop { position: 0.0; color: '#005e00' }
            GradientStop { position: 1.0; color: '#0d7e0d' }
        }

        Gradient {
            id: inactiveGradient
            orientation: Gradient.Horizontal
            GradientStop { position: 0.0; color: '#242c4d' }
            GradientStop { position: 1.0; color: '#5f70b4' }//#3b4e97
        }

        // gradient has higher precedence than color, color is fallback
        color: isActive ? "green" : inactiveColor

        Gradient {
            id: activeGradient
            orientation: Gradient.Horizontal
            GradientStop { position: 0.0; color: "#007400" }
            GradientStop { position: 1.0; color: '#13c713' }//11b811
        }

        // apply gradient only on condition
        gradient: isActive ? activeGradient : (parent.hovered ? hoverGradient : inactiveGradient)
    }

    MouseArea {
        id: hoverArea
        anchors.fill: parent
        hoverEnabled: true
        acceptedButtons: Qt.NoButton
        cursorShape: stackLayout.currentIndex === parent.targetIndex ? Qt.ArrowCursor : (hoverArea.containsMouse ? Qt.PointingHandCursor : Qt.ArrowCursor)
    }

    contentItem: Text {
        text: parent.text
        anchors.centerIn: parent
        color: stackLayout.currentIndex === parent.targetIndex ? "yellow" : (parent.hovered ? '#fffb1c' : "white")
        font.bold: true
        font.pixelSize: stackLayout.currentIndex === parent.targetIndex ? 15 : 14
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
}
