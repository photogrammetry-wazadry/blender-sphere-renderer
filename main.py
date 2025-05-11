import os
from glob import glob
import bpy
import shutil
from zipfile import ZipFile
from pathlib import Path, PurePath
import bmesh
import subprocess
import argparse
import math
import random
import json


MAX_DIMENSION = 10


class Vector():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def len(self):
        return math.hypot(self.x, self.y, self.z)

    def norm(self):
        return Vector(self.x / self.len(), self.y / self.len(), self.z / self.len())



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

        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
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


def orbit_render(file_name, prefix_path, template_path, total_frames, total_random_frames=0, cycles_samples=128, render_resolution=3840, light_type="soft", hdri_path=None, camera_jitter=0, output_file='project.blend'):
    total_frames = int(total_frames)
    total_random_frames = int(total_random_frames)
    prefix_path = os.path.abspath(prefix_path)
    source_path = os.path.join(prefix_path, 'source')

    print("Starting unzip", flush=True)
    unzip_recursively(Path(os.path.join(source_path, file_name)), source_path)
    print("Unzip successful", flush=True)

    # Open template project. template.blend has to start with `prefix_path`, otherwise import fails
    bpy.ops.wm.open_mainfile(filepath=template_path)
    system_objects = []
    for name in bpy.context.scene.objects:  # Save all object names from template
        system_objects.append(name.name)

    for root, _, files in os.walk(source_path):
        for filename in files:
            if Path(filename).suffix == ".obj":
                bpy.ops.import_scene.obj(filepath=os.path.join(root, filename), filter_glob="*.obj;*.mtl",
                                         use_edges=True, use_smooth_groups=True, use_split_objects=True,
                                         use_split_groups=True,
                                         use_groups_as_vgroups=False, use_image_search=True, split_mode='ON',
                                         global_clamp_size=0, axis_forward='-Z', axis_up='Y')

            if Path(filename).suffix in [".glb", ".gltf"]:
                print("GLTF path", root, filename, flush=True)
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

    print("Model imported")
    project_file = os.path.join(prefix_path, output_file)
    bpy.ops.wm.save_as_mainfile(filepath=project_file)

    all_objects = bpy.context.scene.objects  # Get all objects
    new_objects = [all_objects[i] for i, elem in enumerate(all_objects) if elem.name not in system_objects]

    max_scale = -1000
    for obj in new_objects:  # Calculate max size of all objects
        obj.animation_data_clear()
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        max_scale = max(max_scale, max(obj.dimensions))

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

    bound_box_scale = max(bound_box_max[0] - bound_box_min[0], bound_box_max[1] - bound_box_min[1], bound_box_max[2] - bound_box_min[2])
    scale_factor = MAX_DIMENSION / bound_box_scale
    print(f"Model max scale: {bound_box_scale}, scaling by {scale_factor}x to normalise size\n")

    for obj in new_objects:  # Center by cursor position
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
        obj.location = (0, 0, 0)
    bpy.context.scene.cursor.location = (0, 0, 0)

    bpy.ops.object.select_all(action='DESELECT')
    for obj in new_objects:  # Select the whole model
        obj.select_set(True)

    bpy.ops.transform.resize(value=(scale_factor, scale_factor, scale_factor))  # Resize
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)  # Apply transforms to all objects

    scene = bpy.data.scenes['Scene']
    frame_positions = []
    scene.frame_current = scene.frame_end
    bpy.ops.ptcache.bake_all(bake=False)

    camera = bpy.data.objects['Camera']
    bpy.context.scene.render.resolution_x = render_resolution
    bpy.context.scene.render.resolution_y = render_resolution
    bpy.context.scene.cycles.samples = cycles_samples

    # Set HDRI if needed
    if hdri_path is not None and hdri_path != "None":
        # Enable "World" nodes if not already
        bpy.context.scene.world.use_nodes = True
        world_nodes = bpy.context.scene.world.node_tree.nodes
        world_links = bpy.context.scene.world.node_tree.links
        # Clear existing nodes
        world_nodes.clear()
        # Create Environment Texture node
        env_node = world_nodes.new(type='ShaderNodeTexEnvironment')

        shutil.copy(hdri_path, os.path.join(prefix_path, "hdri" + Path(hdri_path).suffix))
        env_node.image = bpy.data.images.load("//hdri" + Path(hdri_path).suffix)
        env_node.location = (-300, 0)
        # Create Background node
        bg_node = world_nodes.new(type='ShaderNodeBackground')
        bg_node.location = (0, 0)
        # Create World Output node
        out_node = world_nodes.new(type='ShaderNodeOutputWorld')
        out_node.location = (300, 0)
        # Link nodes
        world_links.new(env_node.outputs['Color'], bg_node.inputs['Color'])
        world_links.new(bg_node.outputs['Background'], out_node.inputs['Surface'])
        # IMPORTANT: Disable transparent film to see HDRI
        bpy.context.scene.render.film_transparent = False

    # Turn off metalic value for all materials (they interfere with photogrammetry)
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            mat.metallic = 1
            continue
        for n in mat.node_tree.nodes:
            if n.type == 'BSDF_PRINCIPLED':
                n.inputs["Metallic"].default_value = 0
                n.inputs["Roughness"].default_value = 1


    if total_frames > 0:
        # Iterate over all frames (first wide pass)
        steps_on_each_axis = int(total_frames ** 0.5)
        delta1, delta2 = 360 / steps_on_each_axis, 180 / steps_on_each_axis
        distance_from_center = 20

        ax1, ax2 = 0, -1 * steps_on_each_axis / 2

        # Set all main frames
        for frame in range(1, total_frames + 1):
            scene.frame_current = frame

            a1 = delta1 * ax1
            a2 = delta2 * ax2

            distance2 = distance_from_center * math.cos(a2 * math.pi / 180)
            camera.location = (distance2 * math.sin(a1 * math.pi / 180),
                               distance2 * math.cos(a1 * math.pi / 180),
                               distance_from_center * math.sin(a2 * math.pi / 180))

            # Add random jitter to camera positions
            # for i in range(3):
            #     camera.location[i] += random.uniform(-1, 1) * camera_jitter

            angle1 = math.atan2(camera.location[0], camera.location[1]) * 180 / math.pi

            hyp = math.sqrt(camera.location[0] ** 2 + camera.location[1] ** 2)
            angle2 = math.atan2(hyp, camera.location[2]) * 180 / math.pi + 90

            frame_positions.append([camera.location[0], camera.location[1], camera.location[2], a1, a2, 0])  # All the given angles are in degrees
            camera.rotation_mode = 'XYZ'  # or 'ZXY', etc. (default is 'XYZ')
            camera.rotation_euler = (
                math.radians(-a2),   # X (pitch)
                math.radians(0),                  # Y (roll — often 0)
                math.radians(180 - a1)    # Z (yaw)
            )

            camera.keyframe_insert(data_path="rotation_euler", frame=frame)
            camera.keyframe_insert(data_path="location", frame=frame)  # Add keyframe

            # Update indexes
            ax1 += 1
            if ax1 >= steps_on_each_axis:
                ax1 = 0
                ax2 += 1


    # Iterate over all frames (second righter pass)
    if total_random_frames > 0:
        steps_on_each_axis = int(total_random_frames ** 0.5)
        delta1, delta2 = 360 / steps_on_each_axis, 180 / steps_on_each_axis
        distance_from_center = 8

        ax1, ax2 = 0, -1 * steps_on_each_axis / 2

        # Set all main frames
        for frame in range(1, total_random_frames + 1):
            scene.frame_current = total_frames + frame

            a1 = delta1 * ax1
            a2 = delta2 * ax2

            distance2 = distance_from_center * math.cos(a2 * math.pi / 180)
            camera.location = (distance2 * math.sin(a1 * math.pi / 180),
                               distance2 * math.cos(a1 * math.pi / 180),
                               distance_from_center * math.sin(a2 * math.pi / 180))

            # Add random jitter to camera positions
            for i in range(3):
                camera.location[i] += random.uniform(-1, 1) * camera_jitter

            angle1 = math.atan2(camera.location[0], camera.location[1]) * 180 / math.pi

            hyp = math.sqrt(camera.location[0] ** 2 + camera.location[1] ** 2)
            angle2 = math.atan2(hyp, camera.location[2]) * 180 / math.pi + 90

            frame_positions.append([camera.location[0], camera.location[1], camera.location[2], a1, a2, 0])  # All the given angles are in degrees
            camera.rotation_mode = 'XYZ'  # or 'ZXY', etc. (default is 'XYZ')
            camera.rotation_euler = (
                math.radians(-a2),   # X (pitch)
                0,                  # Y (roll — often 0)
                math.radians(180 - a1)    # Z (yaw)
            )

            camera.keyframe_insert(data_path="rotation_euler", frame=total_frames + frame)
            camera.keyframe_insert(data_path="location", frame=total_frames + frame)  # Add keyframe

            # Update indexes
            ax1 += 1
            if ax1 >= steps_on_each_axis:
                ax1 = 0
                ax2 += 1


    # Set all random frames (DEPRECATED FOR NOW)
    # for frame in range(1, total_frames + 1):
    #     scene.frame_current = frame

    #     x, y, z = random.uniform(-1, 1) * MAX_DIMENSION

    #     a1 = delta1 * ax1
    #     a2 = delta2 * ax2

    #     distance2 = distance_from_center * math.cos(a2 * math.pi / 180)
    #     camera.location = (distance2 * math.sin(a1 * math.pi / 180),
    #                        distance2 * math.cos(a1 * math.pi / 180),
    #                        distance_from_center * math.sin(a2 * math.pi / 180))

    #     angle1 = math.atan2(camera.location[0], camera.location[1]) * 180 / math.pi

    #     hyp = math.sqrt(camera.location[0] ** 2 + camera.location[1] ** 2)
    #     angle2 = math.atan2(hyp, camera.location[2]) * 180 / math.pi + 90

    #     frame_positions.append([camera.location[0], camera.location[1], camera.location[2], a1, a2, 0])  # All the given angles are in degrees
    #     camera.keyframe_insert(data_path="location", frame=frame)  # Add keyframe


    bpy.ops.file.make_paths_relative()

    if Path(output_file).suffix == ".blend":  # CYCLES render mode
        # Save project
        project_file = os.path.join(prefix_path, output_file)
        bpy.ops.wm.save_as_mainfile(filepath=project_file)
        with ZipFile(os.path.join(prefix_path, "project.zip"), "w") as zip_file:
            zip_file.write(project_file, output_file)
            if hdri_path is not None and hdri_path != "None":
                zip_file.write(hdri_path, "hdri" + Path(hdri_path).suffix)


    elif Path(output_file).suffix == ".obj":  # PYTORCH3D render mode
        # Export 3D model
        bpy.ops.export_scene.obj(filepath=os.path.join(prefix_path, output_file),
                     check_existing=True, filter_glob='*.obj;*.mtl', use_selection=False,
                     use_animation=False, use_mesh_modifiers=True,
                     use_edges=True, use_smooth_groups=False, use_smooth_groups_bitflags=False,
                     use_normals=True, use_uvs=True,
                     use_materials=True, use_triangles=False, use_nurbs=False, use_vertex_groups=False,
                     use_blen_objects=True,
                     group_by_object=False, group_by_material=False, keep_vertex_order=False,
                     global_scale=1.0, path_mode='AUTO', axis_forward='-Z', axis_up='Y')

        with ZipFile(os.path.join(prefix_path, "project.zip"), "w") as zip_file:
            obj_path = os.path.join(prefix_path, output_file)
            zip_file.write(obj_path, output_file)  # project.obj
            zip_file.write(os.path.join(prefix_path, Path(output_file).stem + ".mtl"), Path(output_file).stem + ".mtl")  # project.mtl
            for directory, folders, files in os.walk(os.path.join(prefix_path, "source/textures")):
                for filename in files:
                    zip_file.write(os.path.join(directory, filename), os.path.join("source/textures", filename))


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

    template_path = os.path.join(opt.path, "template.blend")
    if not os.path.isfile(template_path):
        shutil.copy("template.blend", template_path)
        print("No blender template file found, copying")

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


        os.mkdir(os.path.join(output_dir, "source"))
        shutil.copy(os.path.join(opt.path, "input/", filename), os.path.join(output_dir, "source"))  # Copy model to output folder

        orbit_render(filename, prefix_path=os.path.join(opt.path, "output/", output_folder_name), template_path=os.path.join(opt.path, "template.blend"), total_frames=300, camera_jitter=1)  # Import and normalise size of the model
        print(f"Saved blender project file at {output_dir}")

        if opt.render:
            # Clear and create render directory (where files are stored at the end)

            for line in execute([str(opt.blender), "-b", os.path.join(opt.path, "output/", output_folder_name, "project.blend"),
                                 "-E", ["BLENDER_EEVEE", "CYCLES"][opt.cycles],
                                 "--python", "use_gpu.py", "-o",
                                 f"{os.path.join(os.getcwd(), output_dir)}/###",
                                 "-s", "1", "-a"]):
                try:
                    print(line, end='')
                except subprocess.CalledProcessError as e:
                    print(e.output)

            print(' ---- ' * 10)
