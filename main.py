import os
from glob import glob
import bpy
import shutil
from pathlib import Path, PurePath
import bmesh
import subprocess
import argparse
import math
import random


MAX_DIMENSION = 12


def join_path_parts(path_parts):
    ans = ""
    for part in path_parts:
        ans = os.path.join(ans, part)

    return ans


def unzip_recursively(zip_path):  # Unzip main archive if one exists
    if zip_path.suffix == ".zip":
        split_path = PurePath(zip_path).parts
        extract_dir = os.path.join('./temp/', join_path_parts(split_path[1:-1]))

        os.system(f"unzip {zip_path} -d {extract_dir}")  # Unzip archive
        os.remove(zip_path)

        for root, dirs, files in os.walk(join_path_parts(split_path[:-1])):
            for filename in files:
                unzip_recursively(Path(os.path.join(root, filename)))


def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def orbit_render(file_name, output_file='project.blend'):
    print(file_name, output_file)
    input_path = Path(os.path.join('./input', file_name))
    extract_path = Path(os.path.join('./temp', file_name))

    # Clear working directory
    os.system("rm -rf ./temp/*")
    os.system(f"cp -r {input_path} ./temp")

    print("Starting unzip")
    unzip_recursively(extract_path)
    print("Unzip successful")

    bpy.ops.wm.open_mainfile(filepath="template.blend")  # Open template project (moving camera and lights)
    system_objects = []
    for name in bpy.context.scene.objects:  # Save all object names from template
        system_objects.append(name.name)

    for root, _, files in os.walk("./temp"):
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
    print(f"Model max scale: {max_scale}, scaling to {scale_factor}x to normalise size\n")

    bpy.ops.transform.resize(value=(scale_factor, scale_factor, scale_factor))  # Resize
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)  # Apple transforms to all objects

    # Bounding box
    bound_box_max, bound_box_min = [-1e7, -1e7, -1e7], [1e7, 1e7, 1e7]

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

    scene = bpy.data.scenes['Scene']
    frame_positions = []
    scene.frame_current = scene.frame_end
    bpy.ops.ptcache.bake_all(bake=False)

    camera = bpy.data.objects['Camera']

    # Iterate over all frames
    steps_on_each_axis = int(scene.frame_end ** 0.5)
    delta1, delta2 = 360 / steps_on_each_axis, 180 / steps_on_each_axis
    distance_from_center = 23
    random_factor = 0.05  # Fraction of `distance_from_center`

    ax1, ax2 = 0, -1 * steps_on_each_axis / 2

    for frame in range(scene.frame_end):
        scene.frame_current = frame

        a1 = delta1 * ax1
        a2 = delta2 * ax2

        distance2 = distance_from_center * math.cos(a2 * math.pi / 180)
        camera.location = (distance2 * math.sin(a1 * math.pi / 180),
                           distance2 * math.cos(a1 * math.pi / 180),
                           distance_from_center * math.sin(a2 * math.pi / 180))

        for i in range(3):
            camera.location[i] += random.uniform(-1, 1) * distance_from_center * random_factor

        frame_positions.append(camera.location)
        camera.keyframe_insert(data_path="location", frame=frame)  # Add keyframe

        # Update indexes
        ax1 += 1
        if ax1 >= steps_on_each_axis:
            ax1 = 0
            ax2 += 1


    # Save project
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(output_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--render", help="Render images for every project file", action='store_true')
    parser.add_argument("-c", "--cycles", help="Render project in CYCLES (EEVEE by default)", action='store_true')
    parser.add_argument("-b", "--blender", required=False, help="Blender call path/command", default="blender")
    opt = parser.parse_args()

    # Check needed directories
    if not os.path.exists("input/"):
        print("No input/ directory with 3D models found. Exiting")
        exit(0)

    needed_dirs = ["output/", "temp/"]
    for needed_dir in needed_dirs:
        if not os.path.exists(needed_dir):
            os.mkdir(needed_dir)

    for model_index, filename in enumerate(os.listdir("input/")):
        output_folder_name = f"{str(model_index).zfill(3)}_{os.path.splitext(filename)[0]}"
        output_dir = os.path.join("output/", output_folder_name)

        if not os.path.exists(output_dir):  # Create output_dir
            os.mkdir(output_dir)
        else:
            for delete_file_name in glob(os.path.join(output_dir, "*")):
                if os.path.isdir(delete_file_name):
                    shutil.rmtree(delete_file_name)  # Delete directory
                else:
                    os.remove(delete_file_name)  # Delete file

        orbit_render(filename)  # Import and normalise size of the model
        shutil.copy("project.blend", output_dir)
        print(f"Saved blender project file at {output_dir}")

        if opt.render:
            # Clear and create render directory (where files are stored at the end)

            for line in execute([str(opt.blender), "-b", "project.blend",
                                 "-E", ["BLENDER_EEVEE", "CYCLES"][opt.cycles],
                                 "--python", "use_gpu.py", "-o",
                                 f"{os.path.join(os.getcwd(), output_dir)}/###",
                                 "-s", "1", "-a"]):
                try:
                    print(line, end='')
                except subprocess.CalledProcessError as e:
                    print(e.output)

            print(' ---- ' * 10)
