import glob, os
import bpy
import shutil
from pathlib import Path, PurePath
import mathutils
from mathutils import Vector
import bmesh

MAX_DIMENSION = 12


def download_model(link):
    pass


def join_path_parts(path_parts):
    ans = ""
    for part in path_parts:
        ans = os.path.join(ans, part)

    return ans


def unzip_recursively(zip_path):  # Unzip main archive if one exists
    if zip_path.suffix == ".zip":
        split_path = PurePath(zip_path).parts
        extract_dir = os.path.join('./model/', join_path_parts(split_path[1:-1]))
        print(extract_dir)

        os.system(f"unzip {zip_path} -d {extract_dir}")  # Unzip archive
        os.remove(zip_path)

        for root, dirs, files in os.walk(join_path_parts(split_path[:-1])):
            for filename in files:
                unzip_recursively(Path(os.path.join(root, filename)))


def calc_center_point(system_objects):
    face_cnt, vert_cnt, edge_cnt = 0, 0, 0
    avg_face, avg_edge, avg_vert = Vector((0, 0, 0)), Vector((0, 0, 0)), Vector((0, 0, 0))
    for obj in bpy.data.objects:
        if obj.name in system_objects or obj.data is None: continue

        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)

        for vert in bm.verts:
            vert_cnt += 1
            avg_vert += vert.co + obj.matrix_world.translation

        for face in bm.faces:
            face_cnt += 1
            avg_face += (sum([vert.co + obj.matrix_world.translation for vert in face.verts], Vector()) / len(
                face.verts))

        for edge in bm.edges:
            edge_cnt += 1
            avg_edge += (sum([vert.co + obj.matrix_world.translation for vert in edge.verts], Vector()) / len(
                edge.verts))

        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.context.scene.cursor.location = ((avg_edge / edge_cnt) + (avg_vert / vert_cnt) + (avg_face / face_cnt)) / 3


def orbit_render(file_name):
    input_path = Path(os.path.join('./input', file_name))
    extract_path = Path(os.path.join('./model', file_name))

    # Clear working directory
    os.system(f"rm -rf ./model/*")
    os.system(f"cp -r {input_path} ./model")

    unzip_recursively(extract_path)

    bpy.ops.wm.open_mainfile(filepath="template.blend")  # Open template project (moving camera and lights)
    system_objects = []
    for name in bpy.context.scene.objects:  # Save all object names from template
        system_objects.append(name.name)

    for root, dirs, files in os.walk("./model"):
        for filename in files:
            if Path(filename).suffix == ".obj":
                bpy.ops.import_scene.obj(filepath=os.path.join(root, filename), filter_glob="*.obj;*.mtl",
                                         use_edges=True, use_smooth_groups=True, use_split_objects=True,
                                         use_split_groups=True,
                                         use_groups_as_vgroups=False, use_image_search=True, split_mode='ON',
                                         global_clamp_size=0, axis_forward='-Z', axis_up='Y')

            if Path(filename).suffix in [".glb", ".gltf"]:
                bpy.ops.import_scene.gltf(filepath=os.path.join(root, filename), filter_glob='*.glb;*.gltf',
                                          loglevel=0, import_pack_images=True, merge_vertices=False,
                                          import_shading='NORMALS', bone_heuristic='TEMPERANCE',
                                          guess_original_bind_pose=True)

            if Path(filename).suffix == ".fbx":
                bpy.ops.import_scene.fbx(filepath=os.path.join(root, filename), directory=root, filter_glob='*.fbx',
                                         ui_tab='MAIN', use_manual_orientation=False, global_scale=1.0,
                                         bake_space_transform=False, use_custom_normals=True, use_image_search=True,
                                         use_alpha_decals=False, decal_offset=0.0, use_anim=True, anim_offset=1.0,
                                         use_subsurf=False, use_custom_props=True, use_custom_props_enum_as_string=True,
                                         ignore_leaf_bones=False, force_connect_children=False,
                                         automatic_bone_orientation=False, primary_bone_axis='Y',
                                         secondary_bone_axis='X', use_prepost_rot=True, axis_forward='-Z', axis_up='Y')

    all_objects = bpy.context.scene.objects  # Get all objects
    new_objects = [all_objects[i] for i, elem in enumerate(all_objects) if elem.name not in system_objects]

    max_scale = -1000
    for obj in new_objects:  # Calculate max size of all objects
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        max_scale = max(max_scale, max(obj.dimensions))

    scale_factor = MAX_DIMENSION / max_scale
    print(f"Model max scale: {max_scale}, scaling to {scale_factor}x to normalise size")

    bpy.ops.transform.resize(value=(scale_factor, scale_factor, scale_factor))  # Resize
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)  # Apple transforms to all objects

    # Bounding box
    bound_box_max, bound_box_min = Vector((-1e7, -1e7, -1e7)), Vector((1e7, 1e7, 1e7))

    for obj in new_objects:
        for point in obj.bound_box:
            for i in range(3):
                bound_box_max[i] = max(bound_box_max[i], point[i] + obj.matrix_world.translation[i])

            for i in range(3):
                bound_box_min[i] = min(bound_box_min[i], point[i] + obj.matrix_world.translation[i])

    bpy.context.scene.cursor.location = ((bound_box_max[0] + bound_box_min[0]) / 2,
                                         (bound_box_max[1] + bound_box_min[1]) / 2,
                                         (bound_box_max[2] + bound_box_min[2]) / 2)

    for obj in new_objects:  # Center by cursor position
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
        obj.location = (0, 0, 0)
    bpy.context.scene.cursor.location = (0, 0, 0)

    # Save project
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath('project.blend'))


if __name__ == '__main__':
    orbit_render("tilbury_fort_trailer.zip")
