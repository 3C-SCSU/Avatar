import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

ApplicationWindow {
    visible: true
    width: 1600
    height: 900
    title: "Drone Control"
    color: "#4a5b7b"

    ListModel {
        id: logModel
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 10

        // Top Row - Home, Up, Flight Log
        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 150 // Larger height for the buttons


            // Home Button
            Button {
                Layout.preferredWidth: 200 // Set larger width for Home button
                Layout.preferredHeight: 150 // Set height for Home button

                // Define the custom background
                background: Rectangle {
                    id: buttonBackground // Give an ID to reference
                    color: "#242c4d" // Initial background color
                    border.color: "black" // Border color
                }

                // Stack Image and Text on top of each other and center them
                contentItem: Item {
                    anchors.fill: parent

                    Image {
                        source: "GUI_Pics/home.png"
                        width: 150
                        height: 150
                        anchors.centerIn: parent
                    }

                    Text {
                        id: buttonText // Give an ID to reference
                        text: "Home"
                        color: "white" // Initial text color
                        font.pixelSize: 18
                        anchors.horizontalCenter: parent.horizontalCenter
                        anchors.verticalCenter: parent.verticalCenter
                    }

                    MouseArea {
                        anchors.fill: parent
                        onEntered: {
                            buttonBackground.color = "white"; // Change background to white on hover
                            buttonText.color = "black"; // Change text color to black on hover
                        }
                        onExited: {
                            buttonBackground.color = "#242c4d"; // Revert background color on exit
                            buttonText.color = "white"; // Revert text color to white on exit
                        }
                        onClicked: {
                            droneController.getDroneAction("home");
                        }
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                Layout.preferredHeight: 150

                // Up Button
                Button {
                    Layout.preferredWidth: 1000
                    Layout.preferredHeight: 150

                    // Define the custom background
                    background: Rectangle {
                        id: upButtonBackground // Unique ID for the Up button background
                        color: "#242c4d" // Initial background color
                        border.color: "black" // Border color
                    }

                    // Stack Image and Text on top of each other and center them
                    contentItem: Item {
                        anchors.fill: parent

                        Image {
                            source: "GUI_Pics/up.png"
                            width: 150
                            height: 150
                            anchors.centerIn: parent
                        }

                        Text {
                            id: upButtonText // Unique ID for the Up button text
                            text: "Up"
                            color: "white" // Initial text color
                            font.pixelSize: 18
                            anchors.horizontalCenter: parent.horizontalCenter
                            anchors.verticalCenter: parent.verticalCenter
                        }

                        MouseArea {
                            anchors.fill: parent
                            onEntered: {
                                upButtonBackground.color = "white"; // Change background to white on hover
                                upButtonText.color = "black"; // Change text color to black on hover
                            }
                            onExited: {
                                upButtonBackground.color = "#242c4d"; // Revert background color on exit
                                upButtonText.color = "white"; // Revert text color to white on exit
                            }
                            onClicked: {
                                droneController.getDroneAction("up");
                            }
                        }
                    }
                }
                // Flight Log Label and Space beside Up Button
                ColumnLayout {
                    spacing: 5

                    // Flight Log Label
                    Text {
                        id: flightlog
                        text: "Flight Log"
                        font.pixelSize: 20
                        color: "white"
                    }

                    // Flight Log TextArea (Box)
                    Rectangle {
                        width: 250
                        height: 100
                        color: "white"
                        border.color: "#2E4053"
                        anchors.leftMargin: 20

                        TextArea {
                            id: flightLogSpace
                            width: 400 // Adjust to account for scrollbar width
                            height: 100
                            // Ensure vertical scrollbar is always on

                            ScrollBar {
                                id: flightLogScrollBar
                                orientation: Qt.Vertical
                                anchors.right: parent.right
                                anchors.top: parent.top
                                anchors.bottom: parent.bottom
                                width: 20 // Set width for the scrollbar
                            }
                        }
                    }
                }
            }

            // Log ListView
            ListView {
                Layout.fillWidth: true
                Layout.preferredHeight: 150
                model: logModel
                delegate: Item {
                    Text {
                        text: modelData
                    }
                }
            }
        }

        // Forward Button
        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 150

            // Forward Button
            Button {
                Layout.preferredWidth: 1400
                Layout.preferredHeight: 150

                // Define the custom background
                background: Rectangle {
                    id: forwardBackground // Unique ID for the Forward button background
                    color: "#242c4d" // Initial background color
                    border.color: "black" // Border color
                }

                // Stack Image and Text on top of each other and center them
                contentItem: Item {
                    anchors.fill: parent

                    Image {
                        source: "images/Forward.png"
                        width: 150
                        height: 150
                        anchors.centerIn: parent
                    }

                    Text {
                        id: forwardText // Unique ID for the Forward button text
                        text: "Forward"
                        color: "white" // Initial text color
                        font.pixelSize: 18
                        anchors.horizontalCenter: parent.horizontalCenter
                        anchors.verticalCenter: parent.verticalCenter
                    }

                    MouseArea {
                        anchors.fill: parent
                        onEntered: {
                            forwardBackground.color = "white"; // Change background to white on hover
                            forwardText.color = "black"; // Change text color to black on hover
                        }
                        onExited: {
                            forwardBackground.color = "#242c4d"; // Revert background color on exit
                            forwardText.color = "white"; // Revert text color to white on exit
                        }
                        onClicked: droneController.getDroneAction("forward")
                    }
                }
            }
        }
        // Turn Left, Left, Stream, Right, Turn Right
        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 150

            Button {
                Layout.preferredWidth: 280
                Layout.preferredHeight: 150

                // Define the custom background
                background: Rectangle {
                    id: turnLeftBackground // Unique ID for the Turn Left button background
                    color: "#242c4d" // Initial background color
                    border.color: "black" // Border color
                }

                // Stack Image and Text on top of each other and center them
                contentItem: Item {
                    anchors.fill: parent

                    Image {
                        source: "GUI_Pics/turnLeft.png"
                        width: 150
                        height: 150
                        anchors.centerIn: parent
                    }

                    Text {
                        id: turnLeftText // Unique ID for the Turn Left button text
                        text: "Turn Left"
                        color: "white" // Initial text color
                        font.pixelSize: 18
                        anchors.horizontalCenter: parent.horizontalCenter
                        anchors.verticalCenter: parent.verticalCenter
                    }

                    MouseArea {
                        anchors.fill: parent
                        onEntered: {
                            turnLeftBackground.color = "white"; // Change background to white on hover
                            turnLeftText.color = "black"; // Change text color to black on hover
                        }
                        onExited: {
                            turnLeftBackground.color = "#242c4d"; // Revert background color on exit
                            turnLeftText.color = "white"; // Revert text color to white on exit
                        }
                        onClicked: {
                            droneController.getDroneAction("turn_left");
                        }
                    }
                }
            }

            Button {
                Layout.preferredWidth: 280
                Layout.preferredHeight: 150

                // Define the custom background
                background: Rectangle {
                    id: leftBackground // Unique ID for the Left button background
                    color: "#242c4d" // Initial background color
                    border.color: "black" // Border color
                }

                // Stack Image and Text on top of each other and center them
                contentItem: Item {
                    anchors.fill: parent

                    Image {
                        source: "GUI_Pics/left.png"
                        width: 150
                        height: 150
                        anchors.centerIn: parent
                    }

                    Text {
                        id: leftText // Unique ID for the Left button text
                        text: "Left"
                        color: "white" // Initial text color
                        font.pixelSize: 18
                        anchors.horizontalCenter: parent.horizontalCenter
                        anchors.verticalCenter: parent.verticalCenter
                    }

                    MouseArea {
                        anchors.fill: parent
                        onEntered: {
                            leftBackground.color = "white"; // Change background to white on hover
                            leftText.color = "black"; // Change text color to black on hover
                        }
                        onExited: {
                            leftBackground.color = "#242c4d"; // Revert background color on exit
                            leftText.color = "white"; // Revert text color to white on exit
                        }
                        onClicked: {
                            droneController.getDroneAction("left");
                        }
                    }
                }
            }

            Button {
                Layout.preferredWidth: 280
                Layout.preferredHeight: 150

                // Define the custom background
                background: Rectangle {
                    id: streamBackground // Unique ID for the Stream button background
                    color: "#242c4d" // Initial background color
                    border.color: "black" // Border color
                }

                // Stack Image and Text on top of each other and center them
                contentItem: Item {
                    anchors.fill: parent

                    Image {
                        source: "images/Stream.png"
                        width: 150
                        height: 150
                        anchors.centerIn: parent
                    }

                    Text {
                        id: streamText // Unique ID for the Stream button text
                        text: "Stream"
                        color: "white" // Initial text color
                        font.pixelSize: 18
                        anchors.horizontalCenter: parent.horizontalCenter
                        anchors.verticalCenter: parent.verticalCenter
                    }

                    MouseArea {
                        anchors.fill: parent
                        onEntered: {
                            streamBackground.color = "white"; // Change background to white on hover
                            streamText.color = "black"; // Change text color to black on hover
                        }
                        onExited: {
                            streamBackground.color = "#242c4d"; // Revert background color on exit
                            streamText.color = "white"; // Revert text color to white on exit
                        }
                        onClicked: {
                            droneController.getDroneAction("stream");
                        }
                    }
                }
            }

            Button {
                Layout.preferredWidth: 280
                Layout.preferredHeight: 150

                // Define the custom background
                background: Rectangle {
                    id: rightBackground // Unique ID for the Right button background
                    color: "#242c4d" // Initial background color
                    border.color: "black" // Border color
                }

                // Stack Image and Text on top of each other and center them
                contentItem: Item {
                    anchors.fill: parent

                    Image {
                        source: "GUI_Pics/right.png"
                        width: 150
                        height: 150
                        anchors.centerIn: parent
                    }

                    Text {
                        id: rightText // Unique ID for the Right button text
                        text: "Right"
                        color: "white" // Initial text color
                        font.pixelSize: 18
                        anchors.horizontalCenter: parent.horizontalCenter
                        anchors.verticalCenter: parent.verticalCenter
                    }

                    MouseArea {
                        anchors.fill: parent
                        onEntered: {
                            rightBackground.color = "white"; // Change background to white on hover
                            rightText.color = "black"; // Change text color to black on hover
                        }
                        onExited: {
                            rightBackground.color = "#242c4d"; // Revert background color on exit
                            rightText.color = "white"; // Revert text color to white on exit
                        }
                        onClicked: {
                            droneController.getDroneAction("right");
                        }
                    }
                }
            }


            Button {
                Layout.preferredWidth: 280
                Layout.preferredHeight: 150

                // Define the custom background
                background: Rectangle {
                    id: turnRightBackground // Unique ID for the Turn Right button background
                    color: "#242c4d" // Initial background color
                    border.color: "black" // Border color
                }

                // Stack Image and Text on top of each other and center them
                contentItem: Item {
                    anchors.fill: parent

                    Image {
                        source: "GUI_Pics/turnRight.png"
                        width: 150
                        height: 150
                        anchors.centerIn: parent
                    }

                    Text {
                        id: turnRightText // Unique ID for the Turn Right button text
                        text: "Turn Right"
                        color: "white" // Initial text color
                        font.pixelSize: 18
                        anchors.horizontalCenter: parent.horizontalCenter
                        anchors.verticalCenter: parent.verticalCenter
                    }

                    MouseArea {
                        anchors.fill: parent
                        onEntered: {
                            turnRightBackground.color = "white"; // Change background to white on hover
                            turnRightText.color = "black"; // Change text color to black on hover
                        }
                        onExited: {
                            turnRightBackground.color = "#242c4d"; // Revert background color on exit
                            turnRightText.color = "white"; // Revert text color to white on exit
                        }
                        onClicked: {
                            droneController.getDroneAction("turn_right");
                        }
                    }
                }
            }
        }

        // Back Button
        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 150

            Button {
                Layout.preferredWidth: 1400
                Layout.preferredHeight: 150

                // Define the custom background
                background: Rectangle {
                    id: backButtonBackground // Unique ID for the Back button background
                    color: "#242c4d"    // Initial background color
                    border.color: "black" // Border color
                }

                // Stack Image and Text on top of each other and center them
                contentItem: Item {
                    anchors.fill: parent

                    Image {
                        source: "GUI_Pics/back.png"
                        width: 150
                        height: 150
                        anchors.centerIn: parent
                    }

                    Text {
                        id: backButtonText // Unique ID for the Back button text
                        text: "Back"
                        color: "white" // Initial text color
                        font.pixelSize: 18
                        anchors.horizontalCenter: parent.horizontalCenter
                        anchors.verticalCenter: parent.verticalCenter
                    }

                    MouseArea {
                        anchors.fill: parent
                        onEntered: {
                            backButtonBackground.color = "white"; // Change background to white on hover
                            backButtonText.color = "black"; // Change text color to black on hover
                        }
                        onExited: {
                            backButtonBackground.color = "#242c4d"; // Revert background color on exit
                            backButtonText.color = "white"; // Revert text color to white on exit
                        }
                        onClicked: {
                            droneController.getDroneAction("backward"); // Action for the "Back" button
                        }
                    }
                }
            }
        }

        // Connect, Down, Takeoff, Land
        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 150
            //Connect
            Button {
                Layout.preferredWidth: 200
                Layout.preferredHeight: 150

                // Define the custom background
                background: Rectangle {
                    id: connectButtonBackground // Unique ID for the Connect button background
                    color: "#242c4d"    // Initial background color
                    border.color: "black" // Border color
                }

                // Stack Image and Text on top of each other and center them
                contentItem: Item {
                    anchors.fill: parent

                    Image {
                        source: "GUI_Pics/connect.png"
                        width: 150
                        height: 150
                        anchors.centerIn: parent
                    }

                    Text {
                        id: connectButtonText // Unique ID for the Connect button text
                        text: "Connect"
                        color: "white" // Initial text color
                        font.pixelSize: 18
                        anchors.horizontalCenter: parent.horizontalCenter
                        anchors.verticalCenter: parent.verticalCenter
                    }

                    MouseArea {
                        anchors.fill: parent
                        onEntered: {
                            connectButtonBackground.color = "white"; // Change background to white on hover
                            connectButtonText.color = "black"; // Change text color to black on hover
                        }
                        onExited: {
                            connectButtonBackground.color = "#242c4d"; // Revert background color on exit
                            connectButtonText.color = "white"; // Revert text color to white on exit
                        }
                        onClicked: {
                           droneController.getDroneAction("connect"); // Action for the "Connect" button
                        }
                    }
                }
            }

            Button {
                Layout.preferredWidth: 1000
                Layout.preferredHeight: 150

                // Define the custom background
                background: Rectangle {
                    id: downButtonBackground // Unique ID for the Down button background
                    color: "#242c4d"    // Initial background color
                    border.color: "black" // Border color
                }

                // Stack Image and Text on top of each other and center them
                contentItem: Item {
                    anchors.fill: parent

                    Image {
                        source: "GUI_Pics/down.png"
                        width: 150
                        height: 150
                        anchors.centerIn: parent
                    }

                    Text {
                        id: downButtonText // Unique ID for the Down button text
                        text: "Down"
                        color: "white" // Initial text color
                        font.pixelSize: 18
                        anchors.horizontalCenter: parent.horizontalCenter
                        anchors.verticalCenter: parent.verticalCenter
                    }

                    MouseArea {
                        anchors.fill: parent
                        onEntered: {
                            downButtonBackground.color = "white"; // Change background to white on hover
                            downButtonText.color = "black"; // Change text color to black on hover
                        }
                        onExited: {
                            downButtonBackground.color = "#242c4d"; // Revert background color on exit
                            downButtonText.color = "white"; // Revert text color to white on exit
                        }
                        onClicked: {
                            droneController.getDroneAction("down"); // Action for the "Down" button
                        }
                    }
                }
            }

            Button {
                Layout.preferredWidth: 200
                Layout.preferredHeight: 150

                // Define the custom background
                background: Rectangle {
                    id: takeoffButtonBackground // Unique ID for the Takeoff button background
                    color: "#242c4d"    // Initial background color
                    border.color: "black" // Border color
                }

                // Stack Image and Text on top of each other and center them
                contentItem: Item {
                    anchors.fill: parent

                    Image {
                        source: "GUI_Pics/takeoff.png"
                        width: 150
                        height: 150
                        anchors.centerIn: parent
                    }

                    Text {
                        id: takeoffButtonText // Unique ID for the Takeoff button text
                        text: "Takeoff"
                        color: "white" // Initial text color
                        font.pixelSize: 18
                        anchors.horizontalCenter: parent.horizontalCenter
                        anchors.verticalCenter: parent.verticalCenter
                    }

                    MouseArea {
                        anchors.fill: parent
                        onEntered: {
                            takeoffButtonBackground.color = "white"; // Change background to white on hover
                            takeoffButtonText.color = "black"; // Change text color to black on hover
                        }
                        onExited: {
                            takeoffButtonBackground.color = "#242c4d"; // Revert background color on exit
                            takeoffButtonText.color = "white"; // Revert text color to white on exit
                        }
                        onClicked: droneController.getDroneAction("takeoff")
                    }
                }
            }

            Button {
                Layout.preferredWidth: 200
                Layout.preferredHeight: 150

                // Define the custom background
                background: Rectangle {
                    id: landButtonBackground // Unique ID for the Land button background
                    color: "#242c4d"    // Initial background color
                    border.color: "black" // Border color
                }

                // Stack Image and Text on top of each other and center them
                contentItem: Item {
                    anchors.fill: parent

                    Image {
                        source: "GUI_Pics/land.png"
                        width: 150
                        height: 150
                        anchors.centerIn: parent
                    }

                    Text {
                        id: landButtonText // Unique ID for the Land button text
                        text: "Land"
                        color: "white" // Initial text color
                        font.pixelSize: 18
                        anchors.horizontalCenter: parent.horizontalCenter
                        anchors.verticalCenter: parent.verticalCenter
                    }

                    MouseArea {
                        anchors.fill: parent
                        onEntered: {
                            landButtonBackground.color = "white"; // Change background to white on hover
                            landButtonText.color = "black"; // Change text color to black on hover
                        }
                        onExited: {
                            landButtonBackground.color = "#242c4d"; // Revert background color on exit
                            landButtonText.color = "white"; // Revert text color to white on exit
                        }
                        onClicked: {
                            droneController.getDroneAction("land"); // Action for the "Land" button
                        }
                    }
                }
            }
        }
    }
    function getDroneAction(action) {
        //logModel.append({ action: action + " button pressed" })
        // Here you would implement the actual drone control logic
        console.log(action + " triggered.")
    }
}
