import os
import bpy


bpy.ops.import_scene.obj(filepath="./model/source/model/model.obj", filter_glob="*.obj;*.mtl",
use_edges=True, use_smooth_groups=True, use_split_objects=True, use_split_groups=True,
use_groups_as_vgroups=False, use_image_search=True, split_mode='ON',
global_clamp_size=0, axis_forward='-Z', axis_up='Y')

bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath('project.blend'))

