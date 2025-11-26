# GUI Components

This directory contains reusable QML components for building consistent user interfaces across the Avatar application.

## Components Overview

- **Button_Primary.qml** - A styled primary button with hover effects
- **Form_Input.qml** - A text input field with optional label
- **Form_File_Input.qml** - A file/folder input component with browse dialog

## Usage

### Importing Components

To use these components in your QML file, import the directory:

```qml
import "path/to/GUI_Components"
```

Or if the components are in a relative path:

```qml
import "../GUI_Components"
```

### Button_Primary

A styled button component with hover effects and consistent theming.

#### Properties

- `text` (string) - The button text
- `isHovering` (bool, read-only) - Whether the button is currently being hovered
- `implicitWidth` (int) - Default width (120px)
- `implicitHeight` (int) - Default height (40px)

#### Signals

- `clicked()` - Emitted when the button is clicked

#### Example

```qml
import QtQuick 6.5
import QtQuick.Layouts 1.15
import "../GUI_Components"

RowLayout {
    Button_Primary {
        id: saveButton
        text: "Save Config"
        objectName: "saveButton"
        
        onClicked: {
            console.log("Save button clicked")
            // Your save logic here
        }
    }
    
    Button_Primary {
        id: cancelButton
        text: "Cancel"
        
        onClicked: {
            console.log("Cancel button clicked")
        }
    }
}
```

### Form_Input

A text input field component with an optional label and consistent styling.

#### Properties

- `labelText` (string) - The label text displayed above the input (empty string hides label)
- `text` (string) - The input text value
- `placeholderText` (string) - Placeholder text shown when input is empty
- `echoMode` (enum) - Text echo mode (e.g., `TextInput.Password` for password fields)
- `objectName` (string) - Object name for testing/automation
- `input` (alias) - Direct access to the underlying TextField component

#### Example

```qml
import QtQuick 6.5
import QtQuick.Layouts 1.15
import "../GUI_Components"

ColumnLayout {
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
        labelText: "Username"
        objectName: "usernameInput"
        Layout.fillWidth: true
        text: ""
        placeholderText: "Enter username"
    }
    
    Form_Input {
        id: passwordInput
        labelText: "Password"
        objectName: "passwordInput"
        Layout.fillWidth: true
        echoMode: TextInput.Password
        text: ""
        placeholderText: "Enter password"
    }
}
```

#### Accessing Input Value

```qml
// Get the text value
var hostValue = hostInput.text

// Access the underlying TextField directly
hostInput.input.selectAll()
```

### Form_File_Input

A file or folder input component with a browse button and file dialog integration.

#### Properties

- `labelText` (string) - The label text displayed above the input
- `dialogTitle` (string) - Title for the file/folder dialog (default: "Select File")
- `selectDirectory` (bool) - If true, opens folder dialog; if false, opens file dialog (default: false)
- `text` (string) - The selected file/folder path
- `placeholderText` (string) - Placeholder text shown when no file is selected
- `objectName` (string) - Object name for testing/automation
- `buttonText` (string) - Text for the browse button (default: "Browse")
- `input` (alias) - Direct access to the underlying Form_Input component

#### Signals

- `fileSelected(string filePath)` - Emitted when a file or folder is selected

#### Example

```qml
import QtQuick 6.5
import QtQuick.Layouts 1.15
import "../GUI_Components"

ColumnLayout {
    spacing: 10
    
    // File selection
    Form_File_Input {
        id: fileInput
        labelText: "Select File:"
        dialogTitle: "Choose a file"
        selectDirectory: false
        objectName: "fileInput"
        Layout.fillWidth: true
        placeholderText: "No file selected"
        
        onFileSelected: function(filePath) {
            console.log("File selected:", filePath)
            // Process the selected file
        }
    }
    
    // Directory selection
    Form_File_Input {
        id: directoryInput
        labelText: "Select Directory:"
        dialogTitle: "Choose a directory"
        selectDirectory: true
        objectName: "directoryInput"
        Layout.fillWidth: true
        placeholderText: "/home/user/Documents"
        buttonText: "Browse Folder"
        
        onFileSelected: function(folderPath) {
            console.log("Directory selected:", folderPath)
            // Process the selected directory
        }
    }
}
```

#### Accessing Selected Path

```qml
// Get the selected file/folder path
var selectedPath = fileInput.text

// Access the underlying Form_Input
fileInput.input.text = "/default/path"
```

## Creating New Components

Follow these guidelines when creating new reusable components:

### 1. Component Structure

```qml
import QtQuick 6.5
import QtQuick.Controls 6.5

// Use Controls prefix for Qt Quick Controls
Item {
    id: componentRoot
    
    // Define custom properties
    property string customProperty: ""
    property alias exposedProperty: internalItem.property
    
    // Define signals
    // Signals are a communication mechanism in QML that allow components to notify
    // other parts of the application when something happens. When a signal is emitted,
    // any connected handlers (using the 'onSignalName' syntax) will be called.
    // Signals can carry parameters (like 'string value' here) to pass data along.
    // Example: onCustomSignal: function(value) { console.log("Received:", value) }
    signal customSignal(string value)
    
    // Component implementation
    // ...
}
```

### 2. Property Naming Conventions

- Use camelCase for property names
- Use descriptive names (e.g., `labelText` not `label`)
- Expose commonly needed properties via `alias`
- Document all public properties in comments

### 3. Styling Consistency

Follow the existing color scheme:

```qml
// Primary colors
color: "#2e3a5c"        // Default background
color: "#3e4e7a"        // Hover state
color: "#3a4a6a"        // Input background
color: "#4e5e8a"        // Focus border
color: "white"          // Text color

// Border and radius
border.color: "#2e3a5c"
border.width: 1
radius: 4
```

### 4. Animation Patterns

Use smooth transitions for interactive elements:

```qml
Behavior on color {
    ColorAnimation {
        duration: 150
    }
}
```

### 5. Hover Effects

Implement hover effects using HoverHandler:

```qml
property bool isHovering: false

HoverHandler {
    onHoveredChanged: componentRoot.isHovering = hovered
}

background: Rectangle {
    color: componentRoot.isHovering ? "#3e4e7a" : "#2e3a5c"
}
```

### 6. Component Template

Here's a template for creating a new component:

```qml
import QtQuick 6.5
import QtQuick.Controls 6.5 as Controls
import QtQuick.Layouts 1.15

// ComponentName.qml
// Description: Brief description of what this component does
//
// Properties:
//   - propertyName (type): Description
//
// Signals:
//   - signalName(type): Description
//
// Example:
//   ComponentName {
//       propertyName: "value"
//       onSignalName: function(param) { ... }
//   }

ColumnLayout {
    id: componentRoot
    
    // Public properties
    property string labelText: ""
    property alias value: internalComponent.value
    
    // Signals
    signal valueChanged(string newValue)
    
    // Internal implementation
    Label {
        text: componentRoot.labelText
        color: "white"
        font.bold: true
        Layout.fillWidth: true
        visible: componentRoot.labelText !== ""
    }
    
    // Your component implementation here
    Item {
        id: internalComponent
        property string value: ""
        Layout.fillWidth: true
        
        // Component logic
    }
}
```

### 7. Best Practices

- **Idempotency**: Components should work independently and not rely on external state
- **Reusability**: Make components flexible with properties rather than hardcoded values
- **Accessibility**: Include `objectName` properties for testing and automation
- **Documentation**: Add comments explaining complex logic
- **Performance**: Use `Layout.fillWidth` and `Layout.fillHeight` appropriately
- **Consistency**: Follow existing patterns for similar functionality

### 8. Testing Your Component

Test your component in isolation:

```qml
import QtQuick 6.5
import QtQuick.Window 6.5
import "../GUI_Components"

Window {
    width: 400
    height: 300
    visible: true
    title: "Component Test"
    
    // Test your component here
    YourNewComponent {
        anchors.centerIn: parent
        // Set test properties
    }
}
```

## Component Dependencies

- **QtQuick 6.5** - Core QML types
- **QtQuick.Controls 6.5** - UI controls
- **QtQuick.Layouts 1.15** - Layout management
- **QtQuick.Dialogs** - File dialogs (for Form_File_Input)
- **Qt.labs.platform** - Platform-specific dialogs

## Color Palette Reference

The components use a consistent dark theme:

| Color | Hex | Usage |
|-------|-----|-------|
| Default Background | `#2e3a5c` | Button default, input border |
| Hover Background | `#3e4e7a` | Button hover state |
| Input Background | `#3a4a6a` | Text field background |
| Focus Border | `#4e5e8a` | Input focus state |
| Text Color | `white` | All text elements |

## Examples in Codebase

See `CloudComputing.qml` for real-world usage examples of all three components.

## Contributing

When adding new components:

1. Follow the naming convention: `ComponentName.qml` (PascalCase)
2. Include comprehensive property documentation
3. Add usage examples in this README
4. Test the component in isolation
5. Ensure consistent styling with existing components
6. Update this README with component documentation

