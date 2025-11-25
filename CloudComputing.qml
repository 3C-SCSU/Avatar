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
                text: ""
                placeholderText: "Enter Password"
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

                Button_Primary {
                    id: saveConfigButton
                    objectName: "saveConfigButton"
                    text: "Save Config"
                    
                    onClicked: {
                        root.isSavingConfig = true
                        configFileDialog.open()
                    }
                }
    
                Button_Primary {
                    id: loadConfigButton
                    objectName: "loadConfigButton"
                    text: "Load Config"
                    
                    onClicked: {
                        root.isSavingConfig = false
                        configFileDialog.open()
                    }
                }

                Button_Primary {
                    id: clearConfigButton
                    objectName: "clearConfigButton"
                    text: "Clear Config"
                    
                    onClicked: console.log("Clear Config clicked")
                }

                Button_Primary {
                    id: uploadButton
                    objectName: "uploadButton"
                    text: "Upload"
                    
                    onClicked: console.log("Upload clicked")
                }
            }
            FileDialog {
                id: configFileDialog
                title: "Select Configuration File"
                onAccepted: {
                    if (root.isSavingConfig) {
                        root.saveConfig(
                            hostInput.text,
                            usernameInput.text,
                            privateKeyDirInput.text,
                            targetDirInput.text,
                            ignoreHostKeyCheckbox.checked,
                            sourceDirInput.text,
                            fileUrl.toLocalFile()
                        );
                    } else {
                        root.loadConfig(fileUrl.toLocalFile());
                    }
                }
            }
        }
    }
}
