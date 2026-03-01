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
        height: Math.min(parent.height * 0.9, 500)
        clip: true

        Rectangle {
            width: parent.width
            implicitHeight: contentLayout.implicitHeight + 20
            color: "#5a6b7d"
            border.color: "#CCCCCC"
            border.width: 1
            radius: 4

            ColumnLayout {
                id: contentLayout
                anchors.fill: parent
                anchors.margins: 10
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

                Rectangle {
                    Layout.fillWidth: true
                    Layout.topMargin: -5
                    height: 40
                    color: "transparent"
                    border.color:"#CCCCCC"
                    border.width: 1
                    radius: 4

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 4
                        spacing: 8

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

                            contentItem: Text {
                                text: parent.text
                                color: "white"
                                font.bold: true
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                            }

                            background: Rectangle {
                                color: "#2C3E50"
                                radius: 4
                            }
                        }
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

                Rectangle {
                    Layout.fillWidth: true
                    Layout.topMargin: -5
                    height: 40
                    color: "transparent"
                    border.width: 1
                    border.color: "#CCCCCC"
                    radius: 4

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 4
                        spacing: 8

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

                            contentItem: Text {
                                text: parent.text
                                color: "white"
                                font.bold: true
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                            }

                            background: Rectangle {
                                color: "#2C3E50"
                                radius: 4
                            }
                        }
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

                Rectangle {
                    Layout.fillWidth: true
                    Layout.topMargin: -5
                    height: 40
                    color: "transparent"
                    border.width: 1
                    border.color: "#CCCCCC"
                    radius: 4

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 4
                        spacing: 8

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

                            contentItem: Text {
                                text: parent.text
                                color: "white"
                                font.bold: true
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                            }

                            background: Rectangle {
                                color: "#2C3E50"
                                radius: 4
                            }
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
		    Layout.alignment: Qt.AlignHCenter
                    spacing: 8

                    Button {
                        id: saveConfigButton
                        objectName: "saveConfigButton"
                        text: "Save Config"
                        font.bold: true
                        onClicked: console.log("Save Config clicked")

                        contentItem: Text {
                            text: parent.text
                            color: "white"
                            font.bold: true
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }

                        background: Rectangle {
                            color: "#2C3E50"
                            radius: 4
                        }
                    }

                    Button {
                        id: loadConfigButton
                        objectName: "loadConfigButton"
                        text: "Load Config"
                        font.bold: true
                        onClicked: console.log("Load Config clicked")

                        contentItem: Text {
                            text: parent.text
                            color: "white"
                            font.bold: true
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }

                        background: Rectangle {
                            color: "#2C3E50"
                            radius: 4
                        }
                    }

                    Button {
                        id: clearConfigButton
                        objectName: "clearConfigButton"
                        text: "Clear Config"
                        font.bold: true
                        onClicked: console.log("Clear Config clicked")

                        contentItem: Text {
                            text: parent.text
                            color: "white"
                            font.bold: true
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }

                        background: Rectangle {
                            color: "#2C3E50"
                            radius: 4
                        }
                    }

                    Button {
                        id: uploadButton
                        objectName: "uploadButton"
                        text: "Upload"
                        font.bold: true
                        onClicked: console.log("Upload clicked")

                        contentItem: Text {
                            text: parent.text
                            color: "white"
                            font.bold: true
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }

                        background: Rectangle {
                            color: "#2C3E50"
                            radius: 4
                        }
                    }
                }

                // ==================== SEPARATOR ====================
                Rectangle {
                    Layout.fillWidth: true
                    height: 2
                    color: "#CCCCCC"
                    Layout.topMargin: 20
                    Layout.bottomMargin: 20
                }

                // ==================== OPEN DATA SECTION ====================
                Label {
                    text: "Open Data Publishing"
                    color: "white"
                    font.bold: true
                    font.pixelSize: 18
                    Layout.topMargin: 10
                }

                Label {
                    text: "Augment your BCI dataset by 60% through data replication and shuffling"
                    color: "#E0E0E0"
                    font.pixelSize: 12
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                    Layout.bottomMargin: 5
                }

                // Directory Selection Row
                Label {
                    text: "Parent Directory (contains brainwaves/):"
                    color: "white"
                    font.bold: true
                }

                Rectangle {
                    Layout.fillWidth: true
                    Layout.topMargin: -5
                    height: 40
                    color: "transparent"
                    border.color: "#CCCCCC"
                    border.width: 1
                    radius: 4

                    RowLayout {
                        anchors.fill: parent
                        anchors.margins: 4
                        spacing: 8

                        TextField {
                            id: openDataDirInput
                            objectName: "openDataDirInput"
                            Layout.fillWidth: true
                            text: ""
                            placeholderText: "Select parent directory containing brainwaves/"
                        }

                        Button {
                            id: openDataBrowseButton
                            objectName: "openDataBrowseButton"
                            text: "Browse"
                            font.bold: true

                            contentItem: Text {
                                text: parent.text
                                color: "white"
                                font.bold: true
                                horizontalAlignment: Text.AlignHCenter
                                verticalAlignment: Text.AlignVCenter
                            }

                            background: Rectangle {
                                color: "#2C3E50"
                                radius: 4
                            }

                            onClicked: {
                                openDataFolderDialog.open()
                            }
                        }
                    }
                }

                // Validation Status Text
                Text {
                    id: openDataValidationText
                    text: ""
                    color: "yellow"
                    font.pixelSize: 11
                    visible: text !== ""
                    Layout.fillWidth: true
                    wrapMode: Text.WordWrap
                }

                // Open Data Button
                Button {
                    id: openDataButton
                    objectName: "openDataButton"
                    text: "Run Open Data Augmentation"
                    font.bold: true
                    Layout.alignment: Qt.AlignHCenter
                    Layout.topMargin: 10
                    enabled: openDataDirInput.text !== ""

                    property string buttonState: "normal"

                    contentItem: Text {
                        text: parent.text
                        color: {
                            if (openDataButton.buttonState === "running") return "black"
                            if (openDataButton.buttonState === "success") return "black"
                            return "white"
                        }
                        font.bold: true
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }

                    background: Rectangle {
                        color: {
                            if (openDataButton.buttonState === "running") return "#FFD700"
                            if (openDataButton.buttonState === "success") return "#00CC44"
                            if (openDataButton.buttonState === "error") return "#CC0000"
                            return "#2C3E50"
                        }
                        radius: 4
                        border.color: {
                            if (openDataButton.buttonState === "running") return "#FFA500"
                            if (openDataButton.buttonState === "success") return "#00AA33"
                            return "transparent"
                        }
                        border.width: openDataButton.buttonState !== "normal" ? 2 : 0
                    }

                    onClicked: {
                        var validationResult = openDataAPI.validate_directory(openDataDirInput.text)
                        var validation = JSON.parse(validationResult)

                        if (!validation.valid) {
                            openDataValidationText.text = "✗ " + validation.error
                            openDataValidationText.color = "#FF6666"
                            openDataButton.buttonState = "error"
                            return
                        }

                        openDataValidationText.text = "✓ Found " + validation.files + " files in " + validation.categories + " categories. Will add ~" + validation.estimated_new + " augmented files."
                        openDataValidationText.color = "#00FF00"

                        openDataConfirmDialog.filesCount = validation.files
                        openDataConfirmDialog.estimatedNew = validation.estimated_new
                        openDataConfirmDialog.open()
                    }
                }

                // Console Log Label
                Label {
                    text: "Console Log"
                    color: "white"
                    font.bold: true
                    font.pixelSize: 14
                    Layout.topMargin: 15
                }

                // Console Output Area
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 250
                    color: "white"
                    radius: 4
                    border.color: "#CCCCCC"
                    border.width: 1

                    ScrollView {
                        anchors.fill: parent
                        anchors.margins: 5
                        clip: true

                        TextArea {
                            id: openDataConsoleOutput
                            objectName: "openDataConsoleOutput"
                            text: "Ready to augment dataset...\n\nPlease select a parent directory that contains a 'brainwaves/' subdirectory."
                            readOnly: true
                            wrapMode: TextArea.Wrap
                            color: "black"
                            font.family: "Courier"
                            font.pixelSize: 11
                            background: null
                        }
                    }
                }

                // Status indicator after completion
                Text {
                    id: openDataStatusText
                    text: ""
                    color: "lightgreen"
                    font.bold: true
                    font.pixelSize: 14
                    Layout.alignment: Qt.AlignHCenter
                    visible: text !== ""
                }

                // FolderDialog for selecting directory
                FolderDialog {
                    id: openDataFolderDialog
                    title: "Select Parent Directory (containing brainwaves/)"
                    folder: "file:///"

                    onAccepted: {
                        openDataDirInput.text = openDataFolderDialog.folder
                        openDataValidationText.text = ""
                        openDataButton.buttonState = "normal"
                        openDataStatusText.text = ""
                    }

                    onRejected: {
                        console.log("Folder selection canceled")
                    }
                }

                // Confirmation Dialog
                Dialog {
                    id: openDataConfirmDialog
                    title: "Confirm Data Augmentation"
                    modal: true
                    standardButtons: Dialog.Ok | Dialog.Cancel

                    property int filesCount: 0
                    property int estimatedNew: 0

                    anchors.centerIn: parent
                    width: 450

                    contentItem: Rectangle {
                        color: "#5a6b7d"
                        implicitHeight: confirmContent.implicitHeight + 40

                        ColumnLayout {
                            id: confirmContent
                            anchors.fill: parent
                            anchors.margins: 20
                            spacing: 15

                            Text {
                                text: "⚠ Warning: This operation will modify files in place"
                                color: "#FFD700"
                                font.bold: true
                                font.pixelSize: 13
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                            }

                            Rectangle {
                                Layout.fillWidth: true
                                height: 1
                                color: "#CCCCCC"
                            }

                            Text {
                                text: "Directory: " + openDataDirInput.text.replace("file://", "")
                                color: "white"
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                                font.pixelSize: 11
                            }

                            Text {
                                text: "• Current files: " + openDataConfirmDialog.filesCount
                                color: "white"
                                font.pixelSize: 11
                            }

                            Text {
                                text: "• Estimated augmented files to add: ~" + openDataConfirmDialog.estimatedNew
                                color: "white"
                                font.pixelSize: 11
                            }

                            Text {
                                text: "• Total after augmentation: ~" + (openDataConfirmDialog.filesCount + openDataConfirmDialog.estimatedNew)
                                color: "white"
                                font.pixelSize: 11
                            }

                            Rectangle {
                                Layout.fillWidth: true
                                height: 1
                                color: "#CCCCCC"
                            }

                            Text {
                                text: "The script will:\n  1. Rename files to sequential numbers\n  2. Replicate files randomly (60% increase)\n  3. Shuffle and rename all files\n  4. Report final statistics"
                                color: "#E0E0E0"
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                                font.pixelSize: 10
                            }
                        }
                    }

                    onAccepted: {
                        console.log("User confirmed Open Data augmentation")

                        openDataButton.buttonState = "running"
                        openDataButton.text = "Processing..."
                        openDataConsoleOutput.text = "Starting data augmentation...\nThis may take several minutes for large datasets.\n\n"
                        openDataStatusText.text = ""

                        var output = openDataAPI.run_open_data(openDataDirInput.text)

                        openDataConsoleOutput.text += output

                        if (output.indexOf("Data Generation Complete") >= 0) {
                            openDataButton.buttonState = "success"
                            openDataButton.text = "✓ Augmentation Complete!"
                            openDataStatusText.text = "✓ Dataset Augmentation Successful!"
                            openDataStatusText.color = "lightgreen"
                        } else if (output.indexOf("Error") >= 0 || output.indexOf("✗") >= 0) {
                            openDataButton.buttonState = "error"
                            openDataButton.text = "✗ Error Occurred"
                            openDataStatusText.text = "✗ Augmentation Failed - Check Console Log"
                            openDataStatusText.color = "#FF6666"
                        } else {
                            openDataButton.buttonState = "normal"
                            openDataButton.text = "Run Open Data Augmentation"
                        }
                    }

                    onRejected: {
                        console.log("User canceled Open Data augmentation")
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
