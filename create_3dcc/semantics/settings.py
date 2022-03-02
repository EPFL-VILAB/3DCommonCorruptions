"""
  Name: settings.py
  Desc: Contains all the settings that our scripts will use.

  Usage: for import only
"""
from math import pi
import math
from multiprocessing import cpu_count
from platform import platform
import sys

# CLEVR = False
# CARLA = False
# HYPERSIM = False
# TANKS_AND_TEMPLES = False
# BLENDED_MVS = False

POINT=10
OBJ_DENSITY=15
SPLIT='NONE'

# Images
CREATE_FIXATED = True
CREATE_NONFIXATED = False   # delete ?
CREATE_PANOS = False
POINT_TYPE = 'CORRESPONDENCES'       # 'CORRESPONDENCES'  or 'SWEEP' # The basis for how points are generated
OVERRIDE_MATTERPORT_MODEL_ROTATION = False  # Whether to try to find model rotation with BBoxes

# File paths
MESHLAB_SERVER_PATH = 'meshlabserver'
CAMERA_POSE_FILE = "camera_poses.json"
LOG_FILE = None  # sys.stderr  # Use sys.stderr to avoid Blender garbage

SEMANTIC_MODEL_FILE = "semantic_lookup.obj"       # delete ?
SEMANTIC_PRETTY_MODEL_FILE = "segmented.obj"      # delete ?

MODEL_NAME = 'Onaga'
MODEL_PATH = ''
MODEL_FILE = "mesh.ply"  
PANO_VIEW_NAME = 'equirectangular'
PREFERRED_IMG_EXT = 'PNG'  
OBJ_AXIS_FORWARD = '-Z'  
OBJ_AXIS_UP = 'Y' 

# Render settings and performance
RESOLUTION = 512
RESOLUTION_X = 512                      #1024:hypersim, 768:BlendedMVs
RESOLUTION_Y = 512                      #768:hypersim, 576:BlendedMVs
SENSOR_HEIGHT = 20                    #18:clevr , 20:all
SENSOR_WIDTH = 20                     #32:clevr , 20:all
STOP_VIEW_NUMBER = -1  # 2 # Generate up to (and including) this many views. -1 to disable.
TILE_SIZE = 128
PANO_RESOLUTION = (2048, 1024)
MAX_CONCURRENT_PROCESSES = cpu_count()
SHADE_SMOOTH = True

# Color and depth
BLENDER_VERTEX_COLOR_BIT_DEPTH = 8
COLOR_BITS_PER_CHANNEL = '8'  # bits per channel. PNG allows 8, 16.
DEPTH_BITS_PER_CHANNEL = '16'  # bits per channel. PNG allows 8, 16.

# With 128m and 16-bit channel, has sensitivity 1/512m (128 / 2^16)
DEPTH_ZBUFFER_MAX_DISTANCE_METERS = 128.    #64:clevr , 128:all  
# With 128m and 16-bit channel, has sensitivity 1/512m (128 / 2^16)
DEPTH_EUCLIDEAN_MAX_DISTANCE_METERS = 128.  #64:clevr , 128:all  

# Task settings

# -----Cameras------
GENERATE_CAMERAS = True
SCENE=True
BUILDING=True
MAX_CAMERA_ROLL = 10        # in degrees
MIN_CAMERA_DISTANCE = 1.5 #0.8   # in meters
POINTS_PER_CAMERA = 2 #5
MIN_CAMERA_HEIGHT = 0.5       # in meters
MAX_CAMERA_HEIGHT = 2       # in meters
MIN_CAMERA_DISTANCE_TO_MESH = 0.3  # in meters
MAX_CAMERA_ROLL = 10        # in degrees

# Setting for finding building floors
FLOOR_THICKNESS = 0.25      # in meters
FLOOR_HEIGHT = 2            # in meters

# -----Points------
NUM_POINTS = None
MIN_VIEWS_PER_POINT = 4 #3
MAX_VIEWS_PER_POINT = -1

# -----Curvature------
MIN_CURVATURE_RADIUS = 0.03           #0.0001:clevr , 0.03:all  # in meters
CURVATURE_OUTPUT_MODE = "PRINCIPAL_CURVATURES" #"MEAN_CURVATURE" #PRINCIPAL_CURVATURES, GAUSSIAN_DISPLAY
K1_MESHLAB_SCRIPT = "meshlab_principal_curvatures_k1.mlx"  # Settings can be edited directly in this XML file
K2_MESHLAB_SCRIPT = "meshlab_principal_curvatures_k2.mlx"  # Settings can be edited directly in this XML file
K1_PYMESHLAB_SCRIPT = "pymeshlab_principal_curvatures_k1.mlx"
K2_PYMESHLAB_SCRIPT = "pymeshlab_principal_curvatures_k2.mlx"
MEAN_MESHLAB_SCRIPT_NAME = "meshlab_mean_curvature.mlx"

# -----Edge------
EDGE_3D_THRESH = None  # 0.01

CANNY_RGB_BLUR_SIGMA = 3.0              # 1.0:clevr , 3.0:all
CANNY_RGB_MIN_THRESH = None  # 0.1
CANNY_RGB_MAX_THRESH = None  # 0.8
CANNY_RGB_USE_QUANTILES = True

# -----Keypoint------
# How many meters to use for the diameter of the search radius
# The author suggests 0.3 for indoor spaces:
#   http://www.pcl-users.org/NARF-Keypoint-extraction-parameters-td2874685.html
KEYPOINT_SUPPORT_SIZE = 0.3

# Applies a blur after keypoint detection, radius
KEYPOINT_BLUR_RADIUS = 5

# ----Reshading-----
LAMP_ENERGY = 2.5                 #2.5:all , 10:clevr
LAMP_HALF_LIFE_DISTANCE = 8.0
LAMP_FALLOFF = "INVERSE_SQUARE"

# ----Segmentation----
SEGMENTATION_2D_BLUR = 3.0
SEGMENTATION_2D_SCALE = 200
SEGMENTATION_2D_CUT_THRESH = 0.005
SEGMENTATION_2D_SELF_EDGE_WEIGHT = 2.0

# SEGMENTATION_25D_BLUR = 3.0
SEGMENTATION_25D_SCALE = 200
SEGMENTATION_25D_DEPTH_WEIGHT = 2.
SEGMENTATION_25D_NORMAL_WEIGHT = 1.
SEGMENTATION_25D_EDGE_WEIGHT = 10.
SEGMENTATION_25D_CUT_THRESH = 1.0
SEGMENTATION_25D_SELF_EDGE_WEIGHT = 1.0

# Field of view 
FIELD_OF_VIEW_RADS = math.radians(100)
FIELD_OF_VIEW_MIN_RADS = math.radians(90) #math.radians(35)
FIELD_OF_VIEW_MAX_RADS = math.radians(90) #math.radians(125)
FIELD_OF_VIEW_MATTERPORT_RADS = math.radians(90)
LINE_OF_SITE_HIT_TOLERANCE = 0.001  # Matterport has 1 unit = 1 meter, so 0.001 is 1mm

# Debugging
VERBOSITY_LEVELS = {'ERROR': 0,  # Everything >= VERBOSITY will be printed
                    'WARNING': 20,
                    'STANDARD': 50,
                    'INFO': 90,
                    'DEBUG': 100}
VERBOSTITY_LEVEL_TO_NAME = {v: k for k, v in VERBOSITY_LEVELS.items()}
VERBOSITY = VERBOSITY_LEVELS['INFO']
RANDOM_SEED = 42  # None to disable


# DO NOT CHANGE -- effectively hardcoded
CYCLES_DEVICE = 'GPU'  # Not yet implemented!
EULER_ROTATION_ORDER = 'XYZ'  # Not yet implemented!


DEPTH_ZBUFFER_SENSITIVITY = float(DEPTH_ZBUFFER_MAX_DISTANCE_METERS) / float(2 ** int(DEPTH_BITS_PER_CHANNEL))
DEPTH_EUCLIDEAN_SENSITIVITY = float(DEPTH_EUCLIDEAN_MAX_DISTANCE_METERS) / float(2 ** int(DEPTH_BITS_PER_CHANNEL))

