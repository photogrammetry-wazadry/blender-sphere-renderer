import glob, os
import bpy
import shutil
from pathlib import Path, PurePath


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


def orbit_render(file_name):
    input_path = Path(os.path.join('./input', file_name))
    extract_path = Path(os.path.join('./model', file_name))

    # Clear working directory
    os.system(f"rm -rf ./model/*")
    os.system(f"cp -r {input_path} ./model")

    unzip_recursively(extract_path)

    bpy.ops.wm.open_mainfile(filepath="template.blend")

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

    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath('project.blend'))


if __name__ == '__main__':
    orbit_render("2007_koenigsegg_ccx_no_roof.zip")
