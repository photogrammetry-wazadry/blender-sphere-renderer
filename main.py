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
import json


MAX_DIMENSION = 8


class Vector():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def len(self):
        return math.hypot(self.x, self.y, self.z)

    def norm(self):
        return Vector(self.x / self.len(), self.y / self.len(), self.z / self.len())


"""
class Vector(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __abs__(self):
        return math.hypot(self.x, self.y)

    def norm(self):
        return Vector(self.x / len(self), self.y / len(self))

    def dp(other):
        return x * other.x + y * other.y

    def cp(other):
        return x * other.y + y * other.x

    def angle(other):
        a = math.atan2(self.cp(other), self.dp(other))
        if a >= 0 return a
        return a + 2 * math.pi
"""


def join_path_parts(path_parts):
    ans = ""
    for part in path_parts:
        ans = os.path.join(ans, part)

    return ans


def unzip_recursively(zip_path, temp_path):  # Unzip main archive if one exists
    if zip_path.suffix == ".zip":
        split_path = PurePath(zip_path).parts
        temp_path_depth = len(PurePath(temp_path).parts)
        extract_dir = os.path.join(temp_path, join_path_parts(split_path[temp_path_depth:-1]))

        os.system(f"unzip {zip_path} -d {extract_dir}")  # Unzip archive
        os.remove(zip_path)

        for root, dirs, files in os.walk(join_path_parts(split_path[:-1])):
            for filename in files:
                unzip_recursively(Path(os.path.join(root, filename)), temp_path)


def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def orbit_render(file_name, prefix_path, output_file='project.blend'):
    prefix_path = os.path.abspath(prefix_path)
    temp_path = os.path.join(prefix_path, 'temp')

    input_path = Path(os.path.join(prefix_path, 'input', file_name))
    extract_path = Path(os.path.join(temp_path, file_name))

    # Clear working directory
    os.system(f"rm -rf {os.path.join(prefix_path, 'temp/*')}")
    os.system(f"cp -r {input_path} {os.path.join(prefix_path, 'temp')}")

    print("Starting unzip", flush=True)
    unzip_recursively(extract_path, temp_path)
    print("Unzip successful", flush=True)

    # Open template project. template.blend has to start with `prefix_path`, otherwise import fails
    bpy.ops.wm.open_mainfile(filepath=os.path.join(prefix_path, "template.blend"))
    system_objects = []
    for name in bpy.context.scene.objects:  # Save all object names from template
        system_objects.append(name.name)

    for root, _, files in os.walk(temp_path):
        for filename in files:
            if Path(filename).suffix == ".obj":
                bpy.ops.import_scene.obj(filepath=os.path.join(root, filename), filter_glob="*.obj;*.mtl",
                                         use_edges=True, use_smooth_groups=True, use_split_objects=True,
                                         use_split_groups=True,
                                         use_groups_as_vgroups=False, use_image_search=True, split_mode='ON',
                                         global_clamp_size=0, axis_forward='-Z', axis_up='Y')

            if Path(filename).suffix in [".glb", ".gltf"]:
                print(os.path.join(root, filename), flush=True)
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
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)  # Apply transforms to all objects

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

    total_frames = 500

    # Iterate over all frames
    steps_on_each_axis = int(total_frames ** 0.5)
    delta1, delta2 = 360 / steps_on_each_axis, 180 / steps_on_each_axis
    distance_from_center = 23
    random_factor = 0.05  # Fraction of `distance_from_center`

    ax1, ax2 = 0, -1 * steps_on_each_axis / 2

    for frame in range(total_frames):
        scene.frame_current = frame

        a1 = delta1 * ax1
        a2 = delta2 * ax2

        distance2 = distance_from_center * math.cos(a2 * math.pi / 180)
        camera.location = (distance2 * math.sin(a1 * math.pi / 180),
                           distance2 * math.cos(a1 * math.pi / 180),
                           distance_from_center * math.sin(a2 * math.pi / 180))

        # for i in range(3):
        #     camera.location[i] += random.uniform(-1, 1) * distance_from_center * random_factor

        V = Vector(-camera.location[0], -camera.location[1], -camera.location[2]).norm()

        # angle1, angle2, angle3 = math.atan2(V.x, V.z) * 180 / math.pi, math.asin(-V.y) * 180 / math.pi, 0
        angle1, angle2, angle3 = math.atan2(V.x, V.z) * 180 / math.pi,  math.atan2(V.x, V.y) * 180 / math.pi,  math.atan2(V.y, V.z) * 180 / math.pi
        # angle1 = math.atan2(camera.location[0], camera.location[1]) * 180 / math.pi
        # angle2 = math.atan2(camera.location[1], camera.location[2]) * 180 / math.pi
        # angle3 = math.atan2(camera.location[2], camera.location[0]) * 180 / math.pi

        # frame_positions.append([camera.location[0], camera.location[1], camera.location[2], angle1, angle2, angle3])  # All the given angles are in degrees
        frame_positions.append([camera.location[0], camera.location[1], camera.location[2], a1, a2, 0])  # All the given angles are in degrees
        camera.keyframe_insert(data_path="location", frame=frame)  # Add keyframe

        # Update indexes
        ax1 += 1
        if ax1 >= steps_on_each_axis:
            ax1 = 0
            ax2 += 1


    # Save project
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(prefix_path, output_file))

    # Save camera coordinates to file
    with open(os.path.join(prefix_path, "cameras.json"), "w") as f:
        f.write(json.dumps(frame_positions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--render", help="Render images for every project file", action='store_true')
    parser.add_argument("-c", "--cycles", help="Render project in CYCLES (EEVEE by default)", action='store_true')
    parser.add_argument("-b", "--blender", required=False, help="Blender call path/command", default="blender")
    parser.add_argument("-p", "--path", required=False, help="Path to output folder", default="./")
    opt = parser.parse_args()

    # Check needed directories
    if not os.path.exists(os.path.join(opt.path, "input/")):
        print("No input/ directory with 3D models found. Exiting")
        exit(0)

    needed_dirs = ["output/", "temp/"]
    for needed_dir in needed_dirs:
        needed_dir = os.path.join(opt.path, needed_dir)
        if not os.path.exists(needed_dir):
            os.mkdir(needed_dir)

    for model_index, filename in enumerate(os.listdir(os.path.join(opt.path, "input/"))):
        output_folder_name = f"{str(model_index).zfill(3)}_{os.path.splitext(filename)[0]}"
        output_dir = os.path.join(opt.path, "output/", output_folder_name)

        if not os.path.exists(output_dir):  # Create output_dir
            os.mkdir(output_dir)
        else:
            for delete_file_name in glob(os.path.join(output_dir, "*")):
                if os.path.isdir(delete_file_name):
                    shutil.rmtree(delete_file_name)  # Delete directory
                else:
                    os.remove(delete_file_name)  # Delete file

        orbit_render(filename, prefix_path=opt.path)  # Import and normalise size of the model
        shutil.copy(os.path.join(opt.path, "project.blend"), output_dir)
        print(f"Saved blender project file at {output_dir}")

        if opt.render:
            # Clear and create render directory (where files are stored at the end)

            for line in execute([str(opt.blender), "-b", os.path.join(opt.path, "project.blend"),
                                 "-E", ["BLENDER_EEVEE", "CYCLES"][opt.cycles],
                                 "--python", "use_gpu.py", "-o",
                                 f"{os.path.join(os.getcwd(), output_dir)}/###",
                                 "-s", "1", "-a"]):
                try:
                    print(line, end='')
                except subprocess.CalledProcessError as e:
                    print(e.output)

            print(' ---- ' * 10)
