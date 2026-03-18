import QtQuick
import QtQuick3D

Node {
    id: node

    // Resources
    Texture {
        id: material_0_png_texture
        objectName: "material_0.png"
        generateMipmaps: true
        mipFilter: Texture.Linear
        source: "maps/material_0.png"
    }
    PrincipledMaterial {
        id: material_0_material
        objectName: "material_0"
        baseColor: "#ff666666"
        baseColorMap: material_0_png_texture
        indexOfRefraction: 1
    }

    // Main NAO Body Node
    Node {
        id: nao_body_node
        objectName: "Nao6_Body"
        Model {
            id: nao_body_model
            objectName: "nao6_body_model"
            source: "meshes/nao6_body/Nao6_Body.obj"
            materials: [
                material_0_material
            ]
        }
    }

    // NAO Head Node
    Node {
        id: nao_head_node
        objectName: "Nao6_Head"
        Model {
            id: nao_head_model
            objectName: "nao6_head_model"
            source: "meshes/nao6_body/Nao6_Head.obj"
            materials: [
                material_0_material
            ]
        }
    }

    // NAO Neck Node
    Node {
        id: nao_neck_node
        objectName: "Nao6_Neck"
        Model {
            id: nao_neck_model
            objectName: "nao6_neck_model"
            source: "meshes/nao6_body/Nao6_Neck.obj"
            materials: [
                material_0_material
            ]
        }
    }

    // Left Arm Node
    Node {
        id: nao_left_arm_node
        objectName: "Nao6_Left_Arm"
        Model {
            id: nao_left_arm_model
            objectName: "nao6_left_arm_model"
            source: "meshes/nao6_body/Left_Arm/Left_Arm.obj"
            materials: [
                material_0_material
            ]
        }
    }

    // Right Arm Node
    Node {
        id: nao_right_arm_node
        objectName: "Nao6_Right_Arm"
        Model {
            id: nao_right_arm_model
            objectName: "nao6_right_arm_model"
            source: "meshes/nao6_body/Right_Arm/Right_Arm.obj"
            materials: [
                material_0_material
            ]
        }
    }

    // Left Leg Node
    Node {
        id: nao_left_leg_node
        objectName: "Nao6_Left_Leg"
        Model {
            id: nao_left_leg_model
            objectName: "nao6_left_leg_model"
            source: "meshes/nao6_body/Left_Leg/Left_Leg.obj"
            materials: [
                material_0_material
            ]
        }
    }

    // Right Leg Node
    Node {
        id: nao_right_leg_node
        objectName: "Nao6_Right_Leg"
        Model {
            id: nao_right_leg_model
            objectName: "nao6_right_leg_model"
            source: "meshes/nao6_body/Right_Leg/Right_Leg.obj"
            materials: [
                material_0_material
            ]
        }
    }

    // Animation properties
    property real walkCycle: 0
    property real armSwing: 0
    property real headTurn: 0
    property bool isWalking: false
    property bool isAnimating: false

    // Walking animation timer
    Timer {
        id: walkTimer
        interval: 50 // 20 FPS
        running: isWalking
        repeat: true
        onTriggered: {
            walkCycle += 0.1
            if (walkCycle >= 1.0) walkCycle = 0
            
            // Animate legs
            var leftLegAngle = Math.sin(walkCycle * Math.PI * 2) * 15
            var rightLegAngle = Math.sin((walkCycle + 0.5) * Math.PI * 2) * 15
            
            nao_left_leg_node.eulerRotation.x = leftLegAngle
            nao_right_leg_node.eulerRotation.x = -rightLegAngle
            
            // Animate arms (swinging motion)
            armSwing += 0.05
            var armSwingAngle = Math.sin(armSwing * Math.PI * 2) * 10
            nao_left_arm_node.eulerRotation.y = armSwingAngle
            nao_right_arm_node.eulerRotation.y = -armSwingAngle
            
            // Slight body bounce
            var bodyBounce = Math.abs(Math.sin(walkCycle * Math.PI * 2)) * 2
            nao_body_node.position.y = bodyBounce
        }
    }

    // Arm movement animation
    Timer {
        id: armTimer
        interval: 100
        running: isAnimating
        repeat: true
        onTriggered: {
            armSwing += 0.1
            if (armSwing >= 1.0) armSwing = 0
            
            var armAngle = Math.sin(armSwing * Math.PI * 2) * 20
            nao_left_arm_node.eulerRotation.z = armAngle
            nao_right_arm_node.eulerRotation.z = -armAngle
        }
    }

    // Head turning animation
    Timer {
        id: headTimer
        interval: 80
        running: isAnimating
        repeat: true
        onTriggered: {
            headTurn += 0.1
            if (headTurn >= 1.0) headTurn = 0
            
            var headAngle = Math.sin(headTurn * Math.PI * 2) * 15
            nao_head_node.eulerRotation.y = headAngle
            nao_neck_node.eulerRotation.y = headAngle * 0.5
        }
    }

    // Public functions for animation control
    function startWalking() {
        if (!isWalking) {
            isWalking = true
            isAnimating = true
            walkTimer.start()
            console.log("🚶 NAO started walking animation")
        }
    }

    function stopWalking() {
        if (isWalking) {
            isWalking = false
            walkTimer.stop()
            resetPositions()
            console.log("🛑 NAO stopped walking animation")
        }
    }

    function startArmMovement() {
        if (!isAnimating) {
            isAnimating = true
            armTimer.start()
            console.log("💪 NAO started arm movement animation")
        }
    }

    function stopArmMovement() {
        if (isAnimating) {
            armTimer.stop()
            resetPositions()
            console.log("🛑 NAO stopped arm movement animation")
        }
    }

    function startHeadTurning() {
        if (!isAnimating) {
            isAnimating = true
            headTimer.start()
            console.log("👀 NAO started head turning animation")
        }
    }

    function stopHeadTurning() {
        if (isAnimating) {
            headTimer.stop()
            resetPositions()
            console.log("🛑 NAO stopped head turning animation")
        }
    }

    function resetPositions() {
        // Reset all body parts to neutral positions
        nao_left_leg_node.eulerRotation = Qt.vector3d(0, 0, 0)
        nao_right_leg_node.eulerRotation = Qt.vector3d(0, 0, 0)
        nao_left_arm_node.eulerRotation = Qt.vector3d(0, 0, 0)
        nao_right_arm_node.eulerRotation = Qt.vector3d(0, 0, 0)
        nao_head_node.eulerRotation = Qt.vector3d(0, 0, 0)
        nao_neck_node.eulerRotation = Qt.vector3d(0, 0, 0)
        nao_body_node.position = Qt.vector3d(0, 0, 0)
        
        // Reset animation properties
        walkCycle = 0
        armSwing = 0
        headTurn = 0
        
        if (!isWalking && !isAnimating) {
            isAnimating = false
        }
    }

    function stopAllAnimations() {
        stopWalking()
        stopArmMovement()
        stopHeadTurning()
        console.log("🛑 All NAO animations stopped")
    }

    // Initialize
    Component.onCompleted: {
        resetPositions()
        console.log("🤖 Animated NAO model initialized with joint movements")
    }
}
