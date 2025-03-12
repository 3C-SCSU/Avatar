// Group 8
//Cha Vue, Kevin Gutierrez, Sagar Neupane

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Dialogs

ApplicationWindow {
    visible: true
    width: 640
    height: 480
    color: "#4a5b7b" // Set the background color
    title: "File Shuffler"

    Loader {
        source: "./file-shuffler-component/file-shuffler-view.qml"
        onStatusChanged: {
            if (status === Loader.error) {
                console.error("Error loading file-shuffler-view.qml:", source, errorString())
            } else if (status === Loader.Ready) {
                console.log("Successfully loaded file-shuffler-view.qml:")
            }
        }
    }
}
