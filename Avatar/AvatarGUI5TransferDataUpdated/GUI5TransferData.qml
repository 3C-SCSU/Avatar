import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs 1.3

ApplicationWindow {
    visible: true
    width: 640
    height: 480
    title: "Avatar"

    signal saveConfig(string host, string username, string privateKeyDir, string targetDir, bool ignoreHostKey, string sourceDir, string configPath)
    signal loadConfig(string configPath)
    signal clearConfig()
    signal upload(string host, string username, string privateKeyDir, string password, bool ignoreHostKey, string sourceDir, string targetDir)

    ColumnLayout {
        anchors.fill: parent
        width: parent.width * 2/3 // Adjust width to 2/3 of the parent width

        TabBar {
            id: tabBar
            position: TabBar.Header
            width: parent.width
            height: 40 // Fixed height for the tab bar

            TabButton {
                text: "Brainwave Reading"
                width: contentItem.implicitWidth + 20 // Add some padding
            }
            TabButton {
                text: "Manual Drone Control"
                width: contentItem.implicitWidth + 20
            }
            TabButton {
                text: "Transfer Data"
                width: contentItem.implicitWidth + 20
            }
        }

        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: tabBar.currentIndex

            // Brainwave Reading view
            Rectangle {
                color: "lightgrey"
                Text {
                    anchors.centerIn: parent
                }
            }

            // Manual Drone Control view
            Rectangle {
                color: "lightgrey"
                Text {
                    anchors.centerIn: parent
                    text: "Manual Drone Control View"
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
                        spacing: 10
                        width: parent.width

                        Label { text: "Target IP"; color: "white" }
                        TextField {
                            id: hostInput
                            objectName: "hostInput"
                            Layout.fillWidth: true
                            text: ""
                        }

                        Label { text: "Target Username"; color: "white" }
                        TextField {
                            id: usernameInput
                            objectName: "usernameInput"
                            Layout.fillWidth: true
                            text: ""
                        }

                        Label { text: "Target Password"; color: "white" }
                        TextField {
                            id: passwordInput
                            objectName: "passwordInput"
                            Layout.fillWidth: true
                            echoMode: TextInput.Password
                            text: ""
                        }

                        Label { text: "Private Key Directory:"; color: "white" }
                        RowLayout {
                            Layout.fillWidth: true
                            TextField {
                                id: privateKeyDirInput
                                objectName: "privateKeyDirInput"
                                Layout.fillWidth: true
                                text: ""
                            }
                            Button {
                                id: privateKeyDirButton
                                objectName: "privateKeyDirButton"
                                text: "Browse"
                            }
                        }

                        FileDialog {
                            id: privateKeyFileDialog
                            title: "Select Private Key Directory"
                            onAccepted: {
                                privateKeyDirInput.text = fileUrl.toLocalFile();
                            }
                        }

                        CheckBox {
                            id: ignoreHostKeyCheckbox
                            objectName: "ignoreHostKeyCheckbox"
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
                                objectName: "sourceDirInput"
                                Layout.fillWidth: true
                                text: ""
                            }
                            Button {
                                id: sourceDirButton
                                objectName: "sourceDirButton"
                                text: "Browse"
                            }
                        }

                        FileDialog {
                            id: sourceDirFileDialog
                            title: "Select Source Directory"
                            onAccepted: {
                                sourceDirInput.text = fileUrl.toLocalFile();
                            }
                        }

                        Label { text: "Target Directory:"; color: "white" }
                        RowLayout {
                            Layout.fillWidth: true
                            TextField {
                                id: targetDirInput
                                objectName: "targetDirInput"
                                Layout.fillWidth: true
                                text: "/home/"
                                placeholderText: "/home/"
                            }
                            Button {
                                id: targetDirButton
                                objectName: "targetDirButton"
                                text: "Browse"
                            }
                        }

                        FileDialog {
                            id: targetDirFileDialog
                            title: "Select Target Directory"
                            onAccepted: {
                                targetDirInput.text = fileUrl.toLocalFile();
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            Button {
                                id: saveConfigButton
                                objectName: "saveConfigButton"
                                text: "Save Config"
                            }
                            Button {
                                id: loadConfigButton
                                objectName: "loadConfigButton"
                                text: "Load Config"
                            }
                            Button {
                                id: clearConfigButton
                                objectName: "clearConfigButton"
                                text: "Clear Config"
                            }
                            Button {
                                id: uploadButton
                                objectName: "uploadButton"
                                text: "Upload"
                            }
                        }

                        FileDialog {
                            id: configFileDialog
                            title: "Select Configuration File"
                            onAccepted: {
                                if (saveConfigButton.down) {
                                    saveConfig(
                                        hostInput.text,
                                        usernameInput.text,
                                        privateKeyDirInput.text,
                                        targetDirInput.text,
                                        ignoreHostKeyCheckbox.checked,
                                        sourceDirInput.text,
                                        fileUrl.toLocalFile()
                                    );
                                } else {
                                    loadConfig(fileUrl.toLocalFile());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
