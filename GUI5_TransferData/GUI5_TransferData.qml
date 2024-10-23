import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

ApplicationWindow {
    visible: true
    width: 640
    height: 480
    title: "Avatar"

    ColumnLayout {
           anchors.fill: parent
           width: parent.width * 2/3 // Adjust width to 2/3 of the parent width

           // Tab bar
           TabBar {
               id: tabBar
               position: TabBar.Header
               width: parent.width
               height: 40 // Fixed height for the tab bar

               // Use Row instead of the default layout
               contentItem: Row {
                   spacing: 0 // No spacing between tabs

                   Repeater {
                       model: tabBar.contentModel
                   }
               }

               TabButton {
                   text: "Brainwave Reading"
                   width: contentItem.implicitWidth + 20 // Add some padding
               }
               TabButton {
                   text: "Transfer Data"
                   width: contentItem.implicitWidth + 20
               }
               TabButton {
                   text: "Manual Drone Control"
                   width: contentItem.implicitWidth + 20
               }
           }
        // Stack layout for different views
        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: tabBar.currentIndex

            // Brainwave Reading view
            Rectangle {
                color: "lightgrey"
                Text {
                    anchors.centerIn: parent
                    //text: "Brainwave Reading View"
                }
            }

            // Transfer Data view
            Rectangle {
                color: "#4a5b7b"
                ScrollView {
                    anchors.centerIn: parent
                    width: Math.min(parent.width * 0.9, 600) // Set a maximum width
                    height: Math.min(parent.height * 0.9, contentHeight)
                    clip: true

                    ColumnLayout {
                        id: contentLayout
                        width: parent.width
                        spacing: 10

                        Label { text: "Target IP"; color: "white" }
                        TextField { Layout.fillWidth: true }

                        Label { text: "Target Username"; color: "white" }
                        TextField { Layout.fillWidth: true }

                        Label { text: "Target Password"; color: "white" }
                        TextField {
                            Layout.fillWidth: true
                            echoMode: TextInput.Password
                        }

                        Label { text: "Private Key Directory:"; color: "white" }
                        RowLayout {
                            Layout.fillWidth: true
                            TextField {
                                id: privateKeyDirInput
                                Layout.fillWidth: true
                            }
                            Button {
                                text: "Browse"
                                onClicked: console.log("Browse for Private Key Directory")
                            }
                        }

                        CheckBox {
                            text: "Ignore Host Key"
                            checked: true
                            contentItem: Text {
                                text: parent.text
                                color: "white"
                                leftPadding: parent.indicator.width + parent.spacing
                            }
                        }

                        Label { text: "Source Directory:"; color: "white" }
                        RowLayout {
                            Layout.fillWidth: true
                            TextField {
                                id: sourceDirInput
                                Layout.fillWidth: true
                            }
                            Button {
                                text: "Browse"
                                onClicked: console.log("Browse for Source Directory")
                            }
                        }

                        Label { text: "Target Directory:"; color: "white" }
                        TextField {
                            Layout.fillWidth: true
                            text: "/home/"
                            placeholderText: "/home/"
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            Button {
                                text: "Save Config"
                                onClicked: console.log("Save Config clicked")
                            }
                            Button {
                                text: "Load Config"
                                onClicked: console.log("Load Config clicked")
                            }
                            Button {
                                text: "Clear Config"
                                onClicked: console.log("Clear Config clicked")
                            }
                            Button {
                                text: "Upload"
                                onClicked: console.log("Upload clicked")
                            }
                        }
                    }
                }
            }

            // Manual Drone Control view
            Rectangle {
                color: "lightgrey"
                Text {
                    anchors.centerIn: parent
                  //  text: "Manual Drone Control View"
                }
            }
        }
    }
}
