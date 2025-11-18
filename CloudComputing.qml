import QtQuick.Dialogs
import Qt.labs.platform
import QtQuick 6.5
import QtQuick.Controls 6.4
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtQuick3D 6.7

// Transfer Data is renamed to Cloud Computing
Rectangle {
    color: "#718399"

    signal saveConfig(string host, string username, string privateKeyDir, string targetDir, bool ignoreHostKey, string sourceDir, string configPath)
    signal loadConfig(string configPath)
    signal clearConfig()
    signal upload(string host, string username, string privateKeyDir, string password, bool ignoreHostKey, string sourceDir, string targetDir)


    ScrollView {
        anchors.centerIn: parent
        width: Math.min(parent.width * 0.9, 600)
        height: Math.min(parent.height * 0.9, contentHeight)
        clip: true

        ColumnLayout {
            id: contentLayout
            width: parent.width
            spacing: 10

            Label {
                text: "Target IP"
                color: "white"
                font.bold: true
            }

            TextField {
                id: hostInput
                objectName: "hostInput"
                Layout.fillWidth: true
                text: ""
            }

            Label {
                text: "Target Username"
                color: "white"
                font.bold: true
            }

            TextField {
                id: usernameInput
                objectName:"usernameInput"
                Layout.fillWidth: true
                text: ""
            }

            Label {
                text: "Target Password"
                color: "white"
                font.bold: true
            }

            TextField {
                id: passwordInput
                objectName:"passwordInput"
                Layout.fillWidth: true
                echoMode: TextInput.Password
                text: ""
            }

            Label {
                text: "Private Key Directory:"
                color: "white"
                font.bold: true
            }

            RowLayout {
                Layout.fillWidth: true

                TextField {
                    id: privateKeyDirInput
                    objectName: "privateKeyDirInput"
                    Layout.fillWidth: true
                    text: ""
                }

                Button {
                    id:privateKeyDirButton
                    objectName: "privateKeyDirButton"
                    text: "Browse"
                    font.bold: true
                    onClicked: console.log("Browse for Private Key Directory")
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
                font.bold: true
                checked: true
                contentItem: Text {
                    text: parent.text
                    font.bold: true
                    color: "white"
                    leftPadding: parent.indicator.width + parent.spacing
                }
            }

            Label {
                text: "Source Directory:"
                color: "white"
                font.bold: true
            }

            RowLayout {
                Layout.fillWidth: true

                TextField {
                    id: sourceDirInput
                    objectName: "sourceDirInput"
                    text: ""
                    Layout.fillWidth: true
                }

                Button {
                    id: sourceDirButton
                    objectName: "sourceDirButton"
                    text: "Browse"
                    font.bold: true
                    onClicked: console.log("Browse for Source Directory")
                }
            }

            FileDialog {
                id: sourceDirFileDialog
                title: "Select Source Directory"
                onAccepted: {
                    sourceDirInput.text = fileUrl.toLocalFile();
                }
            }

            Label {
                text: "Target Directory:"
                color: "white"
                font.bold: true
            }
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
                    font.bold: true

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
                    font.bold: true
                    onClicked: console.log("Save Config clicked")
                }
    
                Button {
                    id: loadConfigButton
                    objectName: "loadConfigButton"
                    text: "Load Config"
                    font.bold: true
                    onClicked: console.log("Load Config clicked")
                }

                Button {
                    id: clearConfigButton
                    objectName: "clearConfigButton"
                    text: "Clear Config"
                    font.bold: true
                    onClicked: console.log("Clear Config clicked")
                }

                Button {
                    id: uploadButton
                    objectName: "uploadButton"
                    text: "Upload"
                    font.bold: true
                    onClicked: console.log("Upload clicked")
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
