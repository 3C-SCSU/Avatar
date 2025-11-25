import QtQuick.Dialogs
import Qt.labs.platform
import QtQuick 6.5
import QtQuick.Controls 6.5
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import "GUI_Components"

// Transfer Data is renamed to Cloud Computing
Rectangle {
    id: root
    color: "#718399"

    signal saveConfig(string host, string username, string privateKeyDir, string targetDir, bool ignoreHostKey, string sourceDir, string configPath)
    signal loadConfig(string configPath)
    signal clearConfig()
    signal upload(string host, string username, string privateKeyDir, string password, bool ignoreHostKey, string sourceDir, string targetDir)

    property bool isSavingConfig: false


    ScrollView {
        anchors.centerIn: parent
        width: Math.min(parent.width * 0.9, 600)
        height: Math.min(parent.height * 0.9, contentHeight)
        clip: true

        ColumnLayout {
            id: contentLayout
            width: parent.width
            spacing: 10

            Form_Input {
                id: hostInput
                labelText: "Target IP"
                objectName: "hostInput"
                Layout.fillWidth: true
                text: ""
                placeholderText: "192.168.1.100"
            }

            Form_Input {
                id: usernameInput
                labelText: "Target Username"
                objectName:"usernameInput"
                Layout.fillWidth: true
                text: ""
                placeholderText: "username"
            }

            Form_Input {
                id: passwordInput
                labelText: "Target Password"
                objectName:"passwordInput"
                Layout.fillWidth: true
                echoMode: TextInput.Password
                text: "password"
            }

            Form_File_Input {
                id: privateKeyDirInput
                labelText: "Private Key Directory:"
                dialogTitle: "Select Private Key Directory"
                selectDirectory: true
                objectName: "privateKeyDirInput"
                Layout.fillWidth: true
                text: ""
                placeholderText: "/home/{username}/.ssh"
            }

            CheckBox {
                id: ignoreHostKeyCheckbox
                objectName: "ignoreHostKeyCheckbox"
                text: "Ignore Host Key"
                font.bold: true
                checked: true
                contentItem: Text {
                    text: ignoreHostKeyCheckbox.text
                    font.bold: true
                    color: "white"
                    leftPadding: ignoreHostKeyCheckbox.indicator.width + ignoreHostKeyCheckbox.spacing
                }
            }

            Form_File_Input {
                id: sourceDirInput
                labelText: "Source Directory:"
                dialogTitle: "Select Source Directory"
                selectDirectory: true
                objectName: "sourceDirInput"
                Layout.fillWidth: true
                text: ""
                placeholderText: "/home/{username}/Documents/source"
            }

            Form_File_Input {
                id: targetDirInput
                labelText: "Target Directory:"
                dialogTitle: "Select Target Directory"
                selectDirectory: true
                objectName: "targetDirInput"
                Layout.fillWidth: true
                text: "/home/"
                placeholderText: "/home/{username}/Documents/target"
            }

            RowLayout {
                Layout.fillWidth: true

                Button {
                    id: saveConfigButton
                    objectName: "saveConfigButton"
                    text: "Save Config"
                    font.bold: true
                    implicitWidth: 120
                    implicitHeight: 40
                    
                    property bool isHovering: false
                    
                    HoverHandler {
                        onHoveredChanged: parent.isHovering = hovered
                    }
                    
                    background: Rectangle {
                        color: saveConfigButton.isHovering ? "#3e4e7a" : "#2e3a5c"
                        radius: 4
                        
                        Behavior on color {
                            ColorAnimation {
                                duration: 150
                            }
                        }
                    }
                    
                    contentItem: Text {
                        text: saveConfigButton.text
                        font.pixelSize: 14
                        font.bold: true
                        color: "white"
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    
                    onClicked: {
                        root.isSavingConfig = true
                        configFileDialog.open()
                    }
                }
    
                Button {
                    id: loadConfigButton
                    objectName: "loadConfigButton"
                    text: "Load Config"
                    font.bold: true
                    implicitWidth: 120
                    implicitHeight: 40
                    
                    property bool isHovering: false
                    
                    HoverHandler {
                        onHoveredChanged: parent.isHovering = hovered
                    }
                    
                    background: Rectangle {
                        color: loadConfigButton.isHovering ? "#3e4e7a" : "#2e3a5c"
                        radius: 4
                        
                        Behavior on color {
                            ColorAnimation {
                                duration: 150
                            }
                        }
                    }
                    
                    contentItem: Text {
                        text: loadConfigButton.text
                        font.pixelSize: 14
                        font.bold: true
                        color: "white"
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    
                    onClicked: {
                        root.isSavingConfig = false
                        configFileDialog.open()
                    }
                }

                Button {
                    id: clearConfigButton
                    objectName: "clearConfigButton"
                    text: "Clear Config"
                    font.bold: true
                    implicitWidth: 120
                    implicitHeight: 40
                    
                    property bool isHovering: false
                    
                    HoverHandler {
                        onHoveredChanged: parent.isHovering = hovered
                    }
                    
                    background: Rectangle {
                        color: clearConfigButton.isHovering ? "#3e4e7a" : "#2e3a5c"
                        radius: 4
                        
                        Behavior on color {
                            ColorAnimation {
                                duration: 150
                            }
                        }
                    }
                    
                    contentItem: Text {
                        text: clearConfigButton.text
                        font.pixelSize: 14
                        font.bold: true
                        color: "white"
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    
                    onClicked: console.log("Clear Config clicked")
                }

                Button {
                    id: uploadButton
                    objectName: "uploadButton"
                    text: "Upload"
                    font.bold: true
                    implicitWidth: 120
                    implicitHeight: 40
                    
                    property bool isHovering: false
                    
                    HoverHandler {
                        onHoveredChanged: parent.isHovering = hovered
                    }
                    
                    background: Rectangle {
                        color: uploadButton.isHovering ? "#3e4e7a" : "#2e3a5c"
                        radius: 4
                        
                        Behavior on color {
                            ColorAnimation {
                                duration: 150
                            }
                        }
                    }
                    
                    contentItem: Text {
                        text: uploadButton.text
                        font.pixelSize: 14
                        font.bold: true
                        color: "white"
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    
                    onClicked: console.log("Upload clicked")
                }
            }
            FileDialog {
                id: configFileDialog
                title: "Select Configuration File"
                onAccepted: {
                    if (root.isSavingConfig) {
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
