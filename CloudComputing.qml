import QtQuick.Dialogs
import Qt.labs.platform
import QtQuick 6.5
import QtQuick.Controls 6.5
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15

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
                placeholderText: "192.168.1.100"
                color: "white"
                font.pixelSize: 14
                padding: 10
                
                background: Rectangle {
                    color: "#3a4a6a"
                    border.color: hostInput.activeFocus ? "#4e5e8a" : "#2e3a5c"
                    border.width: 1
                    radius: 4
                    
                    Behavior on border.color {
                        ColorAnimation {
                            duration: 150
                        }
                    }
                }
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
                placeholderText: "username"
                color: "white"
                font.pixelSize: 14
                padding: 10
                
                background: Rectangle {
                    color: "#3a4a6a"
                    border.color: usernameInput.activeFocus ? "#4e5e8a" : "#2e3a5c"
                    border.width: 1
                    radius: 4
                    
                    Behavior on border.color {
                        ColorAnimation {
                            duration: 150
                        }
                    }
                }
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
                text: "password"
                color: "white"
                font.pixelSize: 14
                padding: 10
                
                background: Rectangle {
                    color: "#3a4a6a"
                    border.color: passwordInput.activeFocus ? "#4e5e8a" : "#2e3a5c"
                    border.width: 1
                    radius: 4
                    
                    Behavior on border.color {
                        ColorAnimation {
                            duration: 150
                        }
                    }
                }
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
                    placeholderText: "/home/{username}/.ssh"
                    color: "white"
                    font.pixelSize: 14
                    padding: 10
                    
                    background: Rectangle {
                        color: "#3a4a6a"
                        border.color: privateKeyDirInput.activeFocus ? "#4e5e8a" : "#2e3a5c"
                        border.width: 1
                        radius: 4
                        
                        Behavior on border.color {
                            ColorAnimation {
                                duration: 150
                            }
                        }
                    }
                }

                Button {
                    id: privateKeyDirButton
                    objectName: "privateKeyDirButton"
                    text: "Browse"
                    font.bold: true
                    implicitWidth: 120
                    implicitHeight: 40
                    
                    property bool isHovering: false
                    
                    HoverHandler {
                        onHoveredChanged: parent.isHovering = hovered
                    }
                    
                    background: Rectangle {
                        color: privateKeyDirButton.isHovering ? "#3e4e7a" : "#2e3a5c"
                        radius: 4
                        
                        Behavior on color {
                            ColorAnimation {
                                duration: 150
                            }
                        }
                    }
                    
                    contentItem: Text {
                        text: privateKeyDirButton.text
                        font.pixelSize: 14
                        font.bold: true
                        color: "white"
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    
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
                    text: ignoreHostKeyCheckbox.text
                    font.bold: true
                    color: "white"
                    leftPadding: ignoreHostKeyCheckbox.indicator.width + ignoreHostKeyCheckbox.spacing
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
                    placeholderText: "/home/{username}/Documents/source"
                    Layout.fillWidth: true
                    color: "white"
                    font.pixelSize: 14
                    padding: 10
                    
                    background: Rectangle {
                        color: "#3a4a6a"
                        border.color: sourceDirInput.activeFocus ? "#4e5e8a" : "#2e3a5c"
                        border.width: 1
                        radius: 4
                        
                        Behavior on border.color {
                            ColorAnimation {
                                duration: 150
                            }
                        }
                    }
                }

                Button {
                    id: sourceDirButton
                    objectName: "sourceDirButton"
                    text: "Browse"
                    font.bold: true
                    implicitWidth: 120
                    implicitHeight: 40
                    
                    property bool isHovering: false
                    
                    HoverHandler {
                        onHoveredChanged: parent.isHovering = hovered
                    }
                    
                    background: Rectangle {
                        color: sourceDirButton.isHovering ? "#3e4e7a" : "#2e3a5c"
                        radius: 4
                        
                        Behavior on color {
                            ColorAnimation {
                                duration: 150
                            }
                        }
                    }
                    
                    contentItem: Text {
                        text: sourceDirButton.text
                        font.pixelSize: 14
                        font.bold: true
                        color: "white"
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    
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
                    placeholderText: "/home/{username}/Documents/target"
                    color: "white"
                    font.pixelSize: 14
                    padding: 10
                    
                    background: Rectangle {
                        color: "#3a4a6a"
                        border.color: targetDirInput.activeFocus ? "#4e5e8a" : "#2e3a5c"
                        border.width: 1
                        radius: 4
                        
                        Behavior on border.color {
                            ColorAnimation {
                                duration: 150
                            }
                        }
                    }
                }
                Button {
                    id: targetDirButton
                    objectName: "targetDirButton"
                    text: "Browse"
                    font.bold: true
                    implicitWidth: 120
                    implicitHeight: 40
                    
                    property bool isHovering: false
                    
                    HoverHandler {
                        onHoveredChanged: parent.isHovering = hovered
                    }
                    
                    background: Rectangle {
                        color: targetDirButton.isHovering ? "#3e4e7a" : "#2e3a5c"
                        radius: 4
                        
                        Behavior on color {
                            ColorAnimation {
                                duration: 150
                            }
                        }
                    }
                    
                    contentItem: Text {
                        text: targetDirButton.text
                        font.pixelSize: 14
                        font.bold: true
                        color: "white"
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
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
