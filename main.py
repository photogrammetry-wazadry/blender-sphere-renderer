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


MAX_DIMENSION = 7


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


def bake_normals_in(file_path, prefix_path, template_path):
    file_path = Path(file_path)
    source_path = os.path.join(prefix_path, 'source')  # Temp path that zip will be extracted to and then loaded to bpy


    if file_path.suffix == ".zip":
        # Move from input to source dir to extract
        shutil.rmtree(source_path)  # Delete directory
        os.mkdir(source_path)
        shutil.copy(file_path, source_path)  # Copy model to output folder
        file_path = Path(os.path.join(source_path, file_path.parts[-1]))
        print(file_path)

        print("Starting unzip", flush=True)
        unzip_recursively(file_path, source_path)
        print("Unzip successful", flush=True)


    if file_path.suffix == ".blend":
        bpy.ops.wm.open_mainfile(filepath=file_path)
    

    if file_path.suffix in [".zip", ".gltf"]:
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

        all_objects = bpy.context.scene.objects  # Get all objects
        new_objects = [all_objects[i] for i, elem in enumerate(all_objects) if elem.name not in system_objects]

        bpy.ops.object.select_all(action='DESELECT')
        for obj in new_objects:  # Select mesh objects
            if obj.type == 'MESH':
                print(obj, obj.type)
                obj.animation_data_clear()
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            else:
                obj.select_set(False)

        # Ensure you are in Object Mode
        if bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # Get the active object
        obj = bpy.context.active_object
        print("ACTIVE: ", obj)
        # bpy.ops.wm.save_as_mainfile(filepath="/usr/local/workdir/output/test.blend")

        if not obj:
            print("No active object selected.")
            exit()

        if obj.type != 'MESH':
            print("Selected object is not a mesh.")
            exit()

        # Ensure the object has UV coordinates
        if not obj.data.uv_layers:
            print("The object does not have UV coordinates. Add a UV map first.")
            exit()

        # Create a temporary displacement material
        temp_material = bpy.data.materials.new(name="TempDisplacementMaterial")
        temp_material.use_nodes = True
        obj.data.materials.append(temp_material)

        # Get the node tree of the material
        node_tree = temp_material.node_tree
        nodes = node_tree.nodes
        links = node_tree.links

        # Clear existing nodes
        for node in nodes:
            nodes.remove(node)

        # Add nodes for displacement setup
        tex_node = nodes.new(type='ShaderNodeTexImage')
        tex_node.image = None  # User must load the normal map
        tex_node.interpolation = 'Closest'
        tex_node.extension = 'REPEAT'

        disp_node = nodes.new(type='ShaderNodeDisplacement')
        output_node = nodes.new(type='ShaderNodeOutputMaterial')

        # Connect nodes
        links.new(tex_node.outputs['Color'], disp_node.inputs['Height'])
        links.new(disp_node.outputs['Displacement'], output_node.inputs['Displacement'])

        # Assign a displacement map
        bpy.ops.image.open(filepath="./output/DisplacementMap.png")  # Change to your normal map path
        # bpy.ops.image.open(filepath="//normal_map.png")  # Change to your normal map path
        image = bpy.data.images['DisplacementMap.png']  # Ensure the name matches the file
        tex_node.image = image

        # Set render engine to Cycles and enable displacement
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.feature_set = 'EXPERIMENTAL'

        # Enable displacement on the material
        temp_material.cycles.displacement_method = 'BOTH'

        # Apply a Subdivision Surface Modifier for finer displacement
        subdiv_mod = obj.modifiers.new(name="Subdivision", type='SUBSURF')
        subdiv_mod.levels = 4  # Change the subdivision levels as needed
        subdiv_mod.render_levels = 4

        # Add a Displacement Modifier
        disp_mod = obj.modifiers.new(name="Displacement", type='DISPLACE')
        disp_mod.texture = bpy.data.textures.new(name="DisplacementTexture", type='IMAGE')
        disp_mod.texture.image = image
        disp_mod.texture_coords = 'UV'
        disp_mod.mid_level = 0.5
        disp_mod.strength = 0.3  # Adjust strength as needed

        # Apply the Displacement Modifier to update the geometry
        bpy.ops.object.modifier_apply(modifier=subdiv_mod.name)
        bpy.ops.object.modifier_apply(modifier=disp_mod.name)

        # Cleanup: Remove the temporary material
        # obj.data.materials.remove(temp_material)
        for i, material in enumerate(obj.data.materials):
            if material == temp_material:
                obj.data.materials.pop(index=i)
                break

        # Save the updated model to a file
        output_path = bpy.path.abspath("./output/updated_model.obj")  # Change to your desired output path
        bpy.ops.export_scene.obj(filepath=output_path, use_selection=True)
        print(f"Updated model saved to {output_path}")

        # bpy.ops.export_scene.obj(filepath=os.path.join(export_dir, dir_name + "_input.obj"),
        #                        check_existing=True, filter_glob='*.obj;*.mtl', use_selection=False,
        #                        use_animation=False, use_mesh_modifiers=True,
        #                        use_edges=True, use_smooth_groups=False, use_smooth_groups_bitflags=False,
        #                        use_normals=True, use_uvs=True,
        #                        use_materials=True, use_triangles=False, use_nurbs=False, use_vertex_groups=False,
        #                        use_blen_objects=True,
        #                        group_by_object=False, group_by_material=False, keep_vertex_order=False,
        #                        global_scale=1.0, path_mode='AUTO', axis_forward='-Z', axis_up='Y')


def orbit_render(file_name, prefix_path, template_path, total_frames, cycles_samples=128, render_resolution=3840, light_type="soft", output_file='project.blend'):
    total_frames = int(total_frames)
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

    all_objects = bpy.context.scene.objects  # Get all objects
    new_objects = [all_objects[i] for i, elem in enumerate(all_objects) if elem.name not in system_objects]

    max_scale = -1000
    for obj in new_objects:  # Calculate max size of all objects
        obj.animation_data_clear()
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
    bpy.context.scene.render.resolution_x = render_resolution
    bpy.context.scene.render.resolution_y = render_resolution
    bpy.context.scene.cycles.samples = cycles_samples

    # Turn off metalic value for all materials (they interfere with photogrammetry)
    for mat in bpy.data.materials:
        if not mat.use_nodes:
            mat.metallic = 1
            continue
        for n in mat.node_tree.nodes:
            if n.type == 'BSDF_PRINCIPLED':
                n.inputs["Metallic"].default_value = 0
                n.inputs["Roughness"].default_value = 1


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

        # Add random jitter to camera positions
        # for i in range(3):
        #     camera.location[i] += random.uniform(-1, 1) * distance_from_center * random_factor

        angle1 = math.atan2(camera.location[0], camera.location[1]) * 180 / math.pi

        hyp = math.sqrt(camera.location[0] ** 2 + camera.location[1] ** 2)
        angle2 = math.atan2(hyp, camera.location[2]) * 180 / math.pi + 90

        frame_positions.append([camera.location[0], camera.location[1], camera.location[2], a1, a2, 0])  # All the given angles are in degrees
        camera.keyframe_insert(data_path="location", frame=frame)  # Add keyframe

        # Update indexes
        ax1 += 1
        if ax1 >= steps_on_each_axis:
            ax1 = 0
            ax2 += 1

    if Path(output_file).suffix == ".blend":  # CYCLES render mode
        # Save project
        project_file = os.path.join(prefix_path, output_file)
        bpy.ops.wm.save_as_mainfile(filepath=project_file)
        with ZipFile(os.path.join(prefix_path, "project.zip"), "w") as file:
            file.write(project_file, output_file)

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
    bake_normals_in("input/antique_table.zip", "./", "./template.blend")
    exit(0)

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

        orbit_render(filename, prefix_path=os.path.join(opt.path, "output/", output_folder_name), template_path=os.path.join(opt.path, "template.blend"), total_frames=300)  # Import and normalise size of the model
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
