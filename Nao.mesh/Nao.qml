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

    // Nodes:
    Node {
        id: nao_obj
        objectName: "Nao.obj"
        Model {
            id: defaultobject
            objectName: "defaultobject"
            source: "meshes/defaultobject_mesh.mesh"
            materials: [
                material_0_material
            ]
        }
    }

    // Animations:
}
