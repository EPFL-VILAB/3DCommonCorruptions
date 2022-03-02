import glob
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

import argparse
import imageio

parser = argparse.ArgumentParser(description='overlay masks on rgb and save')
parser.add_argument('--scene', dest='scene', type=str, default='frl_apartment_4', help="name of object")
parser.add_argument('--object_class', dest='object', type=str, help="name of object")
parser.add_argument('--set_id', dest='set_id', type=int, help="which set to use")
args = parser.parse_args()

scene = args.scene
set_id = args.set_id
object = args.object
basedir = f'../../sample_data/semantics/set{set_id}'
rgb_dir = f'{basedir}/rgb/'
semantics_dir=f'{basedir}/{object}/'    

# create directories
if not os.path.exists(os.path.join(semantics_dir,'amodal_mask')):
    os.makedirs(os.path.join(semantics_dir,'amodal_mask'))
if not os.path.exists(os.path.join(semantics_dir,'amodal_mask_rgb')):
    os.makedirs(os.path.join(semantics_dir,'amodal_mask_rgb'))

print("saving to", os.path.join(semantics_dir,'amodal_mask'), os.path.join(semantics_dir,'amodal_mask_rgb'))

# for each point, overlay occluded, unoccluded masks on rgb
for f in tqdm(glob.glob(rgb_dir+'*')):
    _,file_name = os.path.split(f)

    file_name = file_name.replace('rgb','semantic')
    occ_file = os.path.join(semantics_dir,'occluded',file_name)
    occ = Image.open(occ_file)
    occ = np.array(occ)/255.

    unocc_file = os.path.join(semantics_dir,'unoccluded',file_name)
    unocc = Image.open(unocc_file)
    unocc = np.array(unocc)/255.

    rgb = Image.open(f)
    rgb = np.array(rgb)/255.

    mask = (~(occ.mean(-1)>np.zeros_like(occ.mean(-1)))).astype(float)

    overlay = unocc*mask[:,:,None] + occ
    mask_rgb = (~(unocc.mean(-1)>np.zeros_like(unocc.mean(-1)))).astype(float)
    overlay_rgb = rgb*mask_rgb[:,:,None] + overlay

    ## save outputs
    save_file_name = file_name.replace('semantic','amodal_mask')
    amodal_mask = Image.fromarray(np.uint8((overlay.clip(0.,1.0))*255))
    amodal_mask.save(semantics_dir+'amodal_mask/'+save_file_name)
    save_file_name = file_name.replace('semantic','amodal_mask_rgb')
    amodal_mask_rgb = Image.fromarray(np.uint8((overlay_rgb.clip(0.,1.0))*255))
    amodal_mask_rgb.save(semantics_dir+'amodal_mask_rgb/'+save_file_name)

## save output frames as gif   
images = []
for filename in sorted(glob.glob(semantics_dir+'amodal_mask_rgb/*.png')):
    images.append(imageio.imread(filename))
imageio.mimsave(f'{semantics_dir}amodal_mask_rgb/gif.gif', images)
    