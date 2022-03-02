"""
  Name: create_images_utils.py
  Desc: Contains utilities which can be used to run 
  
"""

import logging
import os
import sys
from load_settings import settings
import pdb
from PIL import Image
try:
    import bpy
    import numpy as np
    from mathutils import Vector, Matrix, Quaternion, Euler
    import io_utils
    from io_utils import get_number_imgs
    import utils
    from utils import Profiler

except:
    print("Can't import Blender-dependent libraries in io_utils.py. Proceeding, and assuming this is kosher...")

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

trans = [
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0,0,0,1]],
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0,0,0,1]],
    [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0,0,0,1]],
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0,0,0,1]],
    [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0,0,0,1]],
    [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0,0,0,1]],
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0,0,0,1]],
    [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0,0,0,1]],

    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0,0,0,1]],
    [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0,0,0,1]],
    [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0,0,0,1]],
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0,0,0,1]],
    [[-1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0,0,0,1]],
    [[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0,0,0,1]],
    [[1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0,0,0,1]],
    [[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0,0,0,1]],

    [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0,0,0,1]],
    [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0,0,0,1]],
    [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0,0,0,1]],
    [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0,0,0,1]],
    [[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0,0,0,1]],
    [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0,0,0,1]],
    [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, -1, 0], [0,0,0,1]],
    [[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, -1, 0], [0,0,0,1]],

    [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0,0,0,1]],
    [[0, -1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0,0,0,1]],
    [[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0,0,0,1]],
    [[0, 1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0,0,0,1]],
    [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0,0,0,1]],
    [[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0,0,0,1]],
    [[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0,0,0,1]],
    [[0, -1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0,0,0,1]],

    [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0,0,0,1]],
    [[0, 0, -1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0,0,0,1]],
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0,0,0,1]],
    [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0,0,0,1]],
    [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0,0,0,1]],
    [[0, 0, -1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0,0,0,1]],
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0,0,0,1]],
    [[0, 0, -1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0,0,0,1]],

    [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0,0,0,1]],    
    [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0,0,0,1]],    
    [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0,0,0,1]],    
    [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0,0,0,1]],    
    [[0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0,0,0,1]],    
    [[0, 0, -1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0,0,0,1]],    
    [[0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0], [0,0,0,1]],    
    [[0, 0, -1, 0], [0, -1, 0, 0], [-1, 0, 0, 0], [0,0,0,1]],    

    
]


def start_logging():
    ''' '''
    #   global logger
    logger = io_utils.create_logger(__name__)
    utils.set_random_seed()
    basepath = os.getcwd()
    return logger, basepath


def setup_rendering(setup_scene_fn, setup_nodetree_fn, logger, save_dir, apply_texture=None):
    ''' Sets up everything required to render a scene 
    Args:
    Returns:
        render_save_path: A path where rendered images will be saved (single file)
    '''
    scene = bpy.context.scene
    if apply_texture:
        apply_texture(scene=bpy.context.scene)
    setup_scene_fn(scene)
    render_save_path = setup_nodetree_fn(scene, save_dir)
    return render_save_path


def KRT_from_P(P):
    N = 3
    H = P[:,0:N]  # if not numpy,  H = P.to_3x3()

    [K,R] = rf_rq(H)
    K /= K[-1,-1]

    # from http://ksimek.github.io/2012/08/14/decompose/
    # make the diagonal of K positive
    sg = np.diag(np.sign(np.diag(K)))

    K = K * sg
    R = sg * R
    # det(R) negative, just invert; the proj equation remains same:
    if (np.linalg.det(R) < 0):
       R = -R
    # C = -H\P[:,-1]
    C = np.linalg.lstsq(-H, P[:,-1])[0]
    T = -R*C
    return K, R, T

# RQ decomposition of a numpy matrix, using only libs that already come with
# blender by default
#
# Author: Ricardo Fabbri
# Reference implementations: 
#   Oxford's visual geometry group matlab toolbox 
#   Scilab Image Processing toolbox
#
# Input: 3x4 numpy matrix P
# Returns: numpy matrices r,q
def rf_rq(P):
    P = P.T
    # numpy only provides qr. Scipy has rq but doesn't ship with blender
    q, r = np.linalg.qr(P[ ::-1, ::-1], 'complete')
    q = q.T
    q = q[ ::-1, ::-1]
    r = r.T
    r = r[ ::-1, ::-1]

    if (np.linalg.det(q) < 0):
        r[:,0] *= -1
        q[0,:] *= -1
    return r, q



def setup_and_render_image(task_name, basepath, view_number, view_dict, shot_number, light_loc, execute_render_fn, logger=None,
                           clean_up=True):
    ''' Mutates the given camera and uses it to render the image called for in 
        'view_dict'
    Args:
        task_name: task name + subdirectory to save images
        basepath: model directory
        view_number: The index of the current view
        view_dict: A 'view_dict' for a point/view
        execute_render_fn: A function which renders the desired image
        logger: A logger to write information out to
        clean_up: Whether to delete cameras after use
    Returns:
        None (Renders image)
    '''
    scene = bpy.context.scene
    camera_uuid = view_dict["camera_uuid"]
    point_uuid = view_dict["point_uuid"]
    if "camera_rotation_original" not in view_dict:
        view_dict["camera_rotation_original"] = view_dict["camera_original_rotation"]

    camera, camera_data, scene = utils.get_or_create_camera(
        location=view_dict['camera_location'],
        rotation=view_dict["camera_rotation_original"],
        field_of_view=view_dict["field_of_view_rads"],
        scene=scene,
        camera_name='RENDER_CAMERA')

    scene_and_light_loc = {}
    scene_and_light_loc['scene'] = scene
    scene_and_light_loc['light_loc'] = light_loc

    if settings.CREATE_PANOS:
        utils.make_camera_data_pano(camera_data)
        save_path = io_utils.get_file_name_for(
            dir=get_save_dir(basepath, task_name),
            point_uuid=camera_uuid,
            view_number=settings.PANO_VIEW_NAME,
            camera_uuid=camera_uuid,
            task=task_name,
            ext=io_utils.img_format_to_ext[settings.PREFERRED_IMG_EXT.lower()])
        camera.rotation_euler = Euler(view_dict["camera_rotation_original"],
                                      settings.EULER_ROTATION_ORDER)
        execute_render_fn(scene, save_path)

    elif settings.CREATE_FIXATED:
        # if settings.HYPERSIM: bpy.context.scene.camera.data.clip_end = 10000

        save_path = io_utils.get_file_name_for(
            dir=get_save_dir(basepath, task_name),
            point_uuid=point_uuid,
            view_number=view_number,
            camera_uuid=camera_uuid,
            task=task_name,
            shot_number=shot_number,
            ext=io_utils.img_format_to_ext[settings.PREFERRED_IMG_EXT.lower()])
        # Aim camera at target by rotating a known amount
        camera.rotation_euler = Euler(view_dict["camera_rotation_original"])
        camera.rotation_euler.rotate(
            Euler(view_dict["camera_rotation_from_original_to_final"]))


        ######################################################
        if settings.BLENDED_MVS:
            #print("asdasda")
            perm = 10
            m_t = np.array(trans[perm])[:3,:3]
            P = np.array(view_dict['extrinsics'])[:3,:]

            scale = 1
            K, R_world2cv, T_world2cv = KRT_from_P(np.matrix(P))

            sensor_width_in_mm = K[1,1]*K[0,2] / (K[0,0]*K[1,2])
            resolution_x_in_px = K[0,2]*2  # principal point assumed at the center
            resolution_y_in_px = K[1,2]*2  # principal point assumed at the center

            s_u = resolution_x_in_px / sensor_width_in_mm
            f_in_mm = K[0,0] / s_u

            # scene.render.resolution_x = int(resolution_x_in_px / scale)
            # scene.render.resolution_y = int(resolution_y_in_px / scale)
            # scene.render.resolution_percentage = scale * 100

            arr = np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]])
            # R_bcam2cv = Matrix(m_t)
            # R_bcam2cv = Matrix((m_t).dot(arr))
            R_bcam2cv = Matrix(
                ((1, 0, 0),
                (0, -1, 0),
                (0, 0, -1)))

            R_cv2world = R_world2cv.T
            rotation =  Matrix(R_cv2world.tolist()) * R_bcam2cv
            location = -R_cv2world * T_world2cv

            location = m_t.dot(location)
            rotation = Matrix(m_t.dot(rotation))

            camera.location = location
            # camera.rotation_euler = rotation.to_euler()

            focal_len_x = view_dict['intrinsics'][0][0]
            camera.data.lens = (focal_len_x / settings.RESOLUTION_X) * settings.SENSOR_WIDTH 
            camera.data.lens_unit = 'MILLIMETERS'
            # camera.data.sensor_width  = sensor_width_in_mm
            camera.matrix_world = Matrix.Translation(location)*rotation.to_4x4()

            # print(camera.data.sensor_width, scene.render.resolution_x, scene.render.resolution_y, camera.data.lens)
            # print(camera.location, camera.rotation_euler)
            camera.data.clip_end = 1e7
            bpy.context.scene.update()

        #####################################################3

        if settings.CLEVR:
            camera.matrix_world = Matrix(view_dict['matrix_world'])

        #pdb.set_trace()
        #execute_render_fn(scene, save_path)
        execute_render_fn(scene_and_light_loc, save_path)

        return save_path


    else:
        raise ('Neither settings.CREATE_PANOS nor settings.CREATE_FIXATED is specified')

    if clean_up:
        utils.delete_objects_starting_with("RENDER_CAMERA")  # Clean up

    
def get_save_dir(basepath, task_name):
    if settings.CREATE_PANOS:
        return os.path.join(basepath, 'pano', task_name)
    else:
        return os.path.join(basepath, task_name)


def run(setup_scene_fn, setup_nodetree_fn, model_dir, task_name, apply_texture_fn=None):
    ''' Runs image generation given some render helper functions 
    Args:
        stop_at: A 2-Tuple of (pt_idx, view_idx). If specified, running will cease (not cleaned up) at the given point/view'''
    
    if settings.CLEVR:
        run_clevr(
            setup_scene_fn,
            setup_nodetree_fn,
            model_dir=model_dir,
            task_name=task_name,
            apply_texture_fn=apply_texture_fn)
        return

    utils.set_random_seed()
    logger = io_utils.create_logger(__name__)

    with Profiler("Setup", logger) as prf:
        model_info = io_utils.load_model_and_points(model_dir)
        if apply_texture_fn:
            apply_texture_fn(scene=bpy.context.scene)
        if settings.SHADE_SMOOTH:
            current_mode = bpy.context.object.mode
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.mark_sharp(clear=True)
            bpy.ops.mesh.mark_sharp(clear=True)
            bpy.ops.mesh.mark_sharp(clear=True, use_verts=True)
            bpy.ops.mesh.faces_shade_smooth()
            bpy.ops.object.mode_set(mode=current_mode)
            bpy.ops.object.shade_smooth()
        execute_render = utils.make_render_fn(setup_scene_fn, setup_nodetree_fn,
                                              logger=logger)  # takes (scene, save_dir)
        n_imgs = get_number_imgs(model_info['point_infos'])

    
    #get vertices in global coords
    mesh_curr = model_info['model']
    vertices_curr = mesh_curr.data.vertices
    verts_curr = [mesh_curr.matrix_world * vert.co for vert in vertices_curr] 
    verts_curr = np.array(verts_curr)
    normals_curr = [mesh_curr.matrix_world * vert.normal for vert in vertices_curr] 
    normals_curr = np.array(normals_curr)
    #select where the shot starts and ends
    SHOT_START = 1
    SHOT_END = 100
    with Profiler('Render', logger) as pflr:
        img_number = 0
        shot_number_total = 0
        for point_number, point_info in enumerate(model_info['point_infos']):
            for view_number, view_dict in enumerate(point_info):
                img_number += 1
                view_id = view_number if settings.CREATE_PANOS else view_dict['view_id']
                for shot_number in range(10):
                    shot_number_total += 1
                    print(shot_number_total)
                    if shot_number_total < SHOT_START: continue                    
                    #search for light source loc with rejection sampling
                    min_dist, vec_angle_mindistvert, is_dark = 0, -1., True
                    while min_dist < 0.30  or vec_angle_mindistvert < 0 or is_dark:
                        random_coeff = (np.random.rand(1) ) * 0.5 - 0.5 
                        cam_loc, point_loc = view_dict['camera_location'], view_dict['point_location'] 
                        light_loc = (1-random_coeff) * cam_loc + random_coeff * point_loc 
                       
                        randomization_xy_plane = 1.5 * np.random.randn(3) 
                        randomization_xy_plane[1] = 0
                        light_loc = light_loc + randomization_xy_plane
                  
                        dists = np.sqrt(np.sum( (verts_curr-light_loc)**2,1))
                        min_dist = np.min(dists) 
                        min_dist_idx = np.argmin(dists)
                        min_dist_vert, min_dist_normal = verts_curr[min_dist_idx], normals_curr[min_dist_idx]
                        vec_mindistvert_normal = min_dist_normal - min_dist_vert
                        vec_mindistvert_lightloc = light_loc - min_dist_vert
                        vec_angle_mindistvert = np.sum(vec_mindistvert_normal*vec_mindistvert_lightloc)

                        if min_dist>0.30 and vec_angle_mindistvert>0: 
                            saved_img = setup_and_render_image(task_name, model_dir,
                                                clean_up=True,
                                                execute_render_fn=execute_render,
                                                logger=logger,
                                                view_dict=view_dict,
                                                view_number=view_id,
                                                shot_number=shot_number,
                                                light_loc = light_loc) #view_number

                            img_curr = np.array(Image.open(saved_img))
                            img_curr_mean = np.mean(img_curr)
                            if img_curr_mean > 10: 
                                is_dark = False
                                print("Point number: " + str(point_number) + " View number: " + str(view_number) + " Shot number: " + str(shot_number) + " Light loc: " + str(light_loc) + " Random coeff: " + str(random_coeff) + " Mindist to mesh: " + str(min_dist) + " Vec product with closest vertice: " + str(vec_angle_mindistvert) + " Img mean: " + str(img_curr_mean))

                    pflr.step('finished img {}/{}'.format(img_number, n_imgs))
                    if shot_number_total == SHOT_END: return #break
                if settings.CREATE_PANOS:
                    break  # we only want to create 1 pano per camera
                
    return


############# Clevr functions

def run_clevr(setup_scene_fn, setup_nodetree_fn, model_dir, task_name, apply_texture_fn=None):
    ''' Runs image generation given some render helper functions 
    Args:
        stop_at: A 2-Tuple of (pt_idx, view_idx). If specified, running will cease (not cleaned up) at the given point/view'''
    utils.set_random_seed()
    logger = io_utils.create_logger(__name__)

    point_infos = io_utils.load_saved_points_of_interest_clevr(model_dir)
    n_imgs = get_number_imgs(point_infos)

    img_number = 0

    point_info = point_infos[0] # only setting.POINT will be loaded

    for view_number, view_dict in enumerate(point_info):
        img_number += 1

        settings.MODEL_FILE = 'point_{}_view_{}_domain_obj.obj'.format(settings.POINT, view_number)

        utils.delete_all_objects_in_context()
        model = io_utils.import_mesh(os.path.join(model_dir, 'obj'))

        selection = bpy.context.selected_objects
        for o in selection:
            bpy.context.scene.objects.active = o
            bpy.ops.mesh.customdata_custom_splitnormals_clear()

        with Profiler("Setup", logger) as prf:
            if apply_texture_fn:
                apply_texture_fn(scene=bpy.context.scene)
            if settings.SHADE_SMOOTH:
                current_mode = bpy.context.object.mode
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.mark_sharp(clear=True)
                bpy.ops.mesh.mark_sharp(clear=True)
                bpy.ops.mesh.mark_sharp(clear=True, use_verts=True)
                bpy.ops.mesh.faces_shade_smooth()
                bpy.ops.object.mode_set(mode=current_mode)
                bpy.ops.object.shade_smooth()
            execute_render = utils.make_render_fn(setup_scene_fn, setup_nodetree_fn,
                                                logger=logger)  # takes (scene, save_dir)

        with Profiler('Render', logger) as pflr:
            setup_and_render_image(task_name, 
                                    model_dir,
                                    view_number,
                                    view_dict,
                                    execute_render,
                                    clean_up=True,
                                    logger=logger)

            pflr.step('finished img {}/{}'.format(img_number, n_imgs))
            
    return

