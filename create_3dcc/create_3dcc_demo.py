from create_dof import create_dof_data
from create_fog import create_fog_data
from create_non3d import create_non3d_data
from create_flash import create_flash_data
from create_shadow import create_shadow_data
from create_multi_illumination import create_multi_illumination_data
from motion_video.create_motion import create_motion_data
from motion_video.create_video import create_video_data
import argparse
import pdb

parser = argparse.ArgumentParser(description='create 3dcc data')

parser.add_argument('--path_source', dest='path_source', help="path to source data (rgb, depth, reshade, relighting)")
parser.set_defaults(path_source='NONE')


parser.add_argument('--path_target', dest='path_target', help="path to store 3dcc generated images")
parser.set_defaults(path_target='NONE')

parser.add_argument('--batch_size', dest='batch_size', help="batch Size for processing the data", type=int)
parser.set_defaults(batch_size=1)

args = parser.parse_args()
path_source = args.path_source
path_rgb = path_source + '/rgb'
path_depth = path_source + '/depth'
path_reshade = path_source + '/reshade'
path_relighting = path_source + '/relighting'
path_target = args.path_target
batch_size = args.batch_size


## Create Depth of Field Corruption
print("Near Focus, Far Focus")
create_dof_data(BASE_PATH_RGB=path_rgb, BASE_PATH_DEPTH=path_depth, BASE_TARGET_PATH=path_target, BATCH_SIZE=batch_size)

## Create Camera Motion Corruption 
#pip install cupy==7.7.0 -vvvv
#pip install gevent moviepy
print("XY-Motion Blur, Z-Motion Blur")
create_motion_data(BASE_PATH_RGB=path_rgb, BASE_PATH_DEPTH=path_depth, BASE_TARGET_PATH=path_target, BATCH_SIZE=batch_size)

## Create Fog Corruption 
print("Fog 3D")
create_fog_data(BASE_PATH_RGB=path_rgb, BASE_PATH_DEPTH=path_depth, BASE_TARGET_PATH=path_target, BATCH_SIZE=batch_size)

## Create Non-3D Corruptions 
print("Low-light Noise, Color Quantization, ISO Noise")
create_non3d_data(BASE_PATH_RGB=path_rgb, BASE_TARGET_PATH=path_target, BATCH_SIZE=batch_size)

## Create Lighting Corruptions 
print("Flash")
create_flash_data(BASE_PATH_RGB=path_rgb, BASE_PATH_RESHADE=path_reshade, BASE_TARGET_PATH=path_target, BATCH_SIZE=batch_size)
print("Shadow")
create_shadow_data(BASE_PATH_RGB=path_rgb, BASE_PATH_RELIGHTING=path_relighting, BASE_TARGET_PATH=path_target)
print("Multi-Illumination")
create_multi_illumination_data(BASE_PATH_RGB=path_rgb, BASE_PATH_RELIGHTING=path_relighting, BASE_TARGET_PATH=path_target)

## Create Video Corruption 
#apt install ffmpeg
print("CRF Compression, ABR Compression, Bit Error")
create_video_data(BASE_PATH_RGB=path_rgb, BASE_PATH_DEPTH=path_depth, BASE_TARGET_PATH=path_target, BATCH_SIZE=batch_size)


print("Done! Everything saved at: " + path_target)