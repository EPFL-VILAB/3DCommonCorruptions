""" Based on
  Name: create_semantic_images.py
  Author: Sasha Sax, CVGL
  Desc: Creates semantically tagged versions standard RGB images by using the matterport models and 
    semantic labels.
    This reads in all the point#.json files and rendering the corresponding images with semantic labels. 

  Usage:
    blender -b -noaudio -enable-autoexec --python create_semantic_images.py --
"""

# Import these two first so that we can import other packages
from __future__ import division

import os
import sys
import pdb
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

#
import io_utils

# Import remaining packages
import pickle
import bpy
import bpy_extras.mesh_utils
import bmesh
from collections import defaultdict, Counter
import glob
import json
import math
from mathutils import Vector, Euler, Color
# from matplotlib import cm
import numpy as np
import random
import settings
import shutil  # Temporary dir
import time
import tempfile  # Temporary dir
import utils
import uuid as uu
from utils import Profiler
from plyfile import *
import numpy as np
from scipy.signal import find_peaks
from create_images_utils import *


SCRIPT_DIR_ABS_PATH = os.path.dirname(os.path.realpath(__file__))
TASK_NAME = 'semantic'

utils.set_random_seed()
basepath = settings.MODEL_PATH

things_classes = ['backpack', 'basket', 'bathtub', 'beam', 'beanbag', 'bed', 'bench', 'bike', 'bin',
                  'blinds', 'book', 'bottle', 'box', 'bowl', 'camera', 'candle', 'chair',
                  'chopping-board', 'clock', 'coaster', 'computer-keyboard', 'cup',
                  'cushion', 'curtain', 'cooktop', 'desk', 'desk-organizer', 'desktop-computer',
                  'door', 'exercise-ball', 'faucet', 'handbag', 'hair-dryer', 'indoor-plant',
                  'knife-block', 'kitchen-utensil', 'lamp', 'laptop', 'major-appliance', 'microwave', 'monitor',
                  'mouse', 'nightstand', 'pan', 'phone', 'picture', 'pipe', 'plant-stand', 'plate', 'pot', 'rack',
                  'refrigerator', 'remote-control', 'sculpture', 'shoe', 'shower-stall', 'sink', 'small-appliance',
                  'sofa', 'stool', 'switch', 'table', 'tablet', 'toilet', 'toothbrush', 'tv-screen', 'tv-stand',
                  'umbrella', 'utensil-holder', 'vase', 'vent', 'wardrobe', 'window', 'logo', 'bag']

stuff_classes = ['base-cabinet', 'wall', 'pillar', 'panel', 'wall-cabinet', 'wall-plug', 'floor', 'ceiling', 'stair',
                 'handrail', 'cabinet', 'countertop', 'shelf', 'set-of-clothing', 'paper-towel', 'tissue-paper',
                 'towel', 'cloth', 'clothing', 'scarf', 'rug', 'mat', 'table-runner', 'blanket', 'comforter', 'pillow']

classes = ['undefined', 'backpack', 'base-cabinet', 'basket', 'bathtub', 'beam', 'beanbag', 'bed', 'bench', 'bike',
           'bin', 'blanket', 'blinds', 'book', 'bottle', 'box', 'bowl', 'camera', 'cabinet', 'candle', 'chair',
           'chopping-board', 'clock', 'cloth', 'clothing', 'coaster', 'comforter', 'computer-keyboard', 'cup',
           'cushion', 'curtain', 'ceiling', 'cooktop', 'countertop', 'desk', 'desk-organizer', 'desktop-computer',
           'door', 'exercise-ball', 'faucet', 'floor', 'handbag', 'hair-dryer', 'handrail', 'indoor-plant',
           'knife-block', 'kitchen-utensil', 'lamp', 'laptop', 'major-appliance', 'mat', 'microwave', 'monitor',
           'mouse', 'nightstand', 'pan', 'panel', 'paper-towel', 'phone', 'picture', 'pillar', 'pillow', 'pipe',
           'plant-stand', 'plate', 'pot', 'rack', 'refrigerator', 'remote-control', 'scarf', 'sculpture', 'shelf',
           'shoe', 'shower-stall', 'sink', 'small-appliance', 'sofa', 'stair', 'stool', 'switch', 'table',
           'table-runner', 'tablet', 'tissue-paper', 'toilet', 'toothbrush', 'towel', 'tv-screen', 'tv-stand',
           'umbrella', 'utensil-holder', 'vase', 'vent', 'wall', 'wall-cabinet', 'wall-plug', 'wardrobe', 'window',
           'rug', 'logo', 'bag', 'set-of-clothing']


def get_face_semantics(class_id):
    """
      Find faces for each object.

      Returns:
        objects: A Dict of object id to faces.
    """
    # pdb.set_trace()
    path_in = os.path.join(basepath, settings.MODEL_FILE)

    print("Reading semantic mesh...")
    file_in = PlyData.read(path_in)
    # pdb.set_trace()
    vertices_in = file_in.elements[0]
    faces_in = file_in.elements[1]

    print("Finding object id for each face...")
    object_to_face = defaultdict(list)
    face_to_object = {}
    # pdb.set_trace()
    for f in faces_in:
        # if f[1] not in [12, 65, 68, 75, 115, 222]:   # if label is table
        if f[1] not in [class_id]:   # if label is fridge
        # if f[1] not in [185, 216]:   # if label is bike
            object_id = 0
        else:
            object_id = 1
        object_to_face[object_id].append(tuple(sorted(f[0].tolist())))
        face_to_object[tuple(sorted(f[0].tolist()))] = object_id

    # print("Number of faces with obj_id=0 and class_id=-2 : ", len(object_to_face[0]))
    return object_to_face, face_to_object


def main():
    global basepath
    global TASK_NAME
    utils.delete_all_objects_in_context()

    # cannot use parser, blender will read it as its args
    # parser = argparse.ArgumentParser(description='create 3dcc semantics data')
    # parser.add_argument('--object_class', dest='object_class', help="name of object")
    # parser.add_argument('--generate_occluded', dest='isoccluded', help="occluded or not")
    # args = parser.parse_args()
    
    # model = io_utils.import_mesh(os.path.join(basepath, 'habitat'))
    
    class2id = {'fridge':96,'refrigerator':96,'couch':186,'sofa':186,'table':75}
    assert sys.argv[-1] in class2id.keys()

    # pdb.set_trace()
    model = io_utils.import_mesh(basepath)
    object_name = sys.argv[-1]
    class_id = class2id[sys.argv[-1]]   
    isoccluded = int(sys.argv[-2])
    set_id = int(sys.argv[-3])
    engine = 'BI'
    
    face_to_object = semantically_annotate_mesh(engine, model, class_id, isoccluded)
    
    setpath = os.path.join(basepath,'set'+str(set_id))   # path to set info e.g. point infos and corresponding rgbs
    point_infos = io_utils.load_saved_points_of_interest(setpath)
    
    # create directories where outputs will be saved 
    if not os.path.exists(os.path.join(setpath,object_name)): 
        os.makedirs(os.path.join(setpath,object_name))
    if not os.path.exists(os.path.join(setpath,object_name,'occluded' if isoccluded else 'unoccluded')): 
        os.makedirs(os.path.join(setpath,object_name,'occluded' if isoccluded else 'unoccluded'))
    
    # render + save
    for point_info in point_infos:
        for view_number, view_dict in enumerate(point_info):
            view_id = view_dict['view_id']
            setup_and_render_image(TASK_NAME, object_name, setpath,
                                clean_up=True,
                                execute_render_fn=render_semantic_img_occ if isoccluded else render_semantic_img_unocc,
                                logger=None,
                                view_dict=view_dict,
                                view_number=view_id,
                                face_to_object=face_to_object)

'''
    SEMANTICS
'''


def add_materials_to_mesh(materials_dict, mesh):
    bpy.context.scene.objects.active = mesh
    materials_idxs = {}  # defaultdict( dict )
    for label, mat in materials_dict.items():
        bpy.ops.object.material_slot_add()
        mesh.material_slots[-1].material = mat
        materials_idxs[label] = len(mesh.material_slots) - 1
    return materials_dict, materials_idxs


def build_colormap(objects, semantic_info, isoccluded):

    if isoccluded:
        colors = [np.array([0.,0.,0.]),np.array([0., 0., 1.])]
    else:
        colors = [np.array([0.,0.,0.]),np.array([1., 0., 0.])]   ## for unoccluded
    # colors = [np.array([0., 0., 1.])]    ## for occluded - only visible part of obj
    # colors = [np.array([0.,0.,0.]),np.array([0., 0., 1.])]    ## for occluded - only visible part of obj
    def get_color(obj_id):
        return colors[obj_id]

    return get_color


def build_materials_dict(engine, objects, semantic_info, isoccluded):
    '''
    Args:
        colormap: A function that returns a color for a semantic label

    Returns:
        materials_dict: A dict: materials_dict[ class ][ instance_num ] -> material

    '''

    colormap = build_colormap(objects, semantic_info, isoccluded)
    materials_dict = {}
    for i, obj_id in enumerate(objects):
        materials_dict[str(obj_id)] = utils.create_material_with_color(colormap(obj_id), name=str(obj_id),
                                                                       engine=engine)
    return materials_dict


def semantically_annotate_mesh(engine, mesh, class_id, isoccluded):
    file_path = os.path.join(basepath, 'info_semantic.json')
    with open(file_path) as json_file:
        semantic_info = json.load(json_file)

    with Profiler("Read semantic annotations") as prf:
        object_to_face, face_to_object = get_face_semantics(class_id)
        print("Number of objects : {} - Number of labeled faces : {} ".format(len(object_to_face.keys()),
                                                                              len(face_to_object.keys())))
    #### save face2obj!
    print("len(id_to_label) : ", len(semantic_info['id_to_label']))

    ids = []
    for obj in semantic_info['objects']:
        ids.append(obj['id'])
    print("obj ids : ", sorted(ids))

    materials_dict = build_materials_dict(engine, object_to_face.keys(), semantic_info, isoccluded)

    # create materials
    with Profiler('Create materials on mesh'):
        _, materials_idxs = add_materials_to_mesh(materials_dict, mesh)
    bpy.context.scene.objects.active = mesh
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh.data)
    bm.select_mode = {'FACE'}  # Go to face selection mode

    # Deselect all faces
    for face in bm.faces:
        face.select_set(False)

    mesh.data.update()
    bm.faces.ensure_lookup_table()

    replica_face2obj=np.zeros(len(bm.faces))
    with Profiler("Applying materials") as prf:
        # Count the votes and apply materials
        for i, face in enumerate(bm.faces):  # Iterate over all of the object's faces
            face_vertices = tuple(sorted([v.index for v in face.verts]))
            if face_vertices not in face_to_object.keys():
                continue
            label = face_to_object[face_vertices]
            replica_face2obj[i] = label
            face.material_index = materials_idxs[str(label)]  # Assing random material to face
                        
        mesh.data.update()
        bpy.ops.object.mode_set(mode='OBJECT')

    return face_to_object


'''
    RENDER
'''


def render_semantic_img_occ(scene, save_path, face_to_object):
    """
      Renders an image from the POV of the camera and save it out
      Args:
        camera: A Blender camera already pointed in the right direction
        camera_data:
        scene: A Blender scene that the camera will render
        save_path: Where to save the image
        model: The model in context after loading the .ply
        view_dict: The loaded view dict from point_uuid.json
    """
    save_path_dir, img_filename = os.path.split(save_path)
    with Profiler("Render") as prf:

        utils.set_preset_render_settings(scene, presets=['BASE', 'NON-COLOR'])
        render_save_path = setup_scene_for_semantic_render(scene, save_path_dir)
        prf.step("Setup")

        bpy.ops.render.render()
        prf.step("Render")

    with Profiler("Saving") as prf:
        new_save_path, img_filename = os.path.split(save_path)
        new_save_path += '/occluded/' + img_filename
        shutil.move(render_save_path, new_save_path)


def render_semantic_img_unocc(scene, save_path, face_to_object):
    """
      Renders an image from the POV of the camera and save it out

      Args:
        camera: A Blender camera already pointed in the right direction
        camera_data:
        scene: A Blender scene that the camera will render
        save_path: Where to save the image
        model: The model in context after loading the .ply
        view_dict: The loaded view dict from point_uuid.json
    """
    # pdb.set_trace()
    mesh_curr = bpy.context.object
    prevMode = mesh_curr.mode
    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
    # Create a bmesh access
    bm = bmesh.new()
    bm.from_mesh( mesh_curr.data )
    
    if len(bm.faces)>100000: ## if original mesh
        camera_curr = bpy.context.scene.camera
        bm.faces.ensure_lookup_table()

        faces_delete = []
        for ind,face in enumerate(bm.faces):
            face_vertices_curr_tuple = tuple(sorted([v.index for v in face.verts]))
            label = face_to_object[face_vertices_curr_tuple]
            
            if label not in [1]:   
                faces_delete.append(face)

        bmesh.ops.delete( bm, geom = faces_delete, context = 5 )
        # Push the geometry back to the mesh
        bm.to_mesh( mesh_curr.data )
        # Back to the initial mode
        bpy.ops.object.mode_set(mode=prevMode, toggle=False)

    
    save_path_dir, img_filename = os.path.split(save_path)
    with Profiler("Render") as prf:

        utils.set_preset_render_settings(scene, presets=['BASE', 'NON-COLOR'])
        render_save_path = setup_scene_for_semantic_render(scene, save_path_dir)
        prf.step("Setup")

        bpy.ops.render.render()
        prf.step("Render")

    with Profiler("Saving") as prf:
        new_save_path, img_filename = os.path.split(save_path)
        new_save_path += '/unoccluded/' + img_filename
        shutil.move(render_save_path, new_save_path)



def setup_scene_for_semantic_render(scene, outdir):
    """
      Creates the scene so that a depth image will be saved.

      Args:
        scene: The scene that will be rendered
        camera: The main camera that will take the view
        model: The main model
        outdir: The directory to save raw renders to

      Returns:
        save_path: The path to which the image will be saved
    """
    # Use node rendering for python control
    scene.use_nodes = True
    tree = scene.node_tree
    links = tree.links

    # Make sure there are no existing nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    #  Set up a renderlayer and plug it into our remapping layer
    inp = tree.nodes.new('CompositorNodeRLayers')

    if (bpy.app.version[1] >= 70):  # Don't apply color transformation -- changed in Blender 2.70
        scene.view_settings.view_transform = 'Raw'
        scene.sequencer_colorspace_settings.name = 'Non-Color'

    # Save it out
    if outdir:
        out = tree.nodes.new('CompositorNodeOutputFile')
        ident = str(uu.uuid4())
        out.file_slots[0].path = ident
        out.base_path = outdir
        # out.format.color_mode = 'BW'
        # out.format.color_depth = settings.DEPTH_BITS_PER_CHANNEL
        out.format.color_mode = 'RGB'
        out.format.color_depth = settings.COLOR_BITS_PER_CHANNEL
        out.format.file_format = settings.PREFERRED_IMG_EXT.upper()
        links.new(inp.outputs[0], out.inputs[0])
        ext = utils.img_format_to_ext[settings.PREFERRED_IMG_EXT.lower()]
        temp_filename = "{0}0001.{1}".format(ident, ext)
        return os.path.join(outdir, temp_filename)
    else:
        out = tree.nodes.new('CompositorNodeComposite')
        links.new(inp.outputs[0], out.inputs[0])
        return None


if __name__ == '__main__':
    with Profiler("create_semantic_images.py"):
        main()
