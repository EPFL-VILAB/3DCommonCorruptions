import torch
from torchvision import transforms

from models.unet import UNet
from models import DPTRegressionModel
from models.models import TrainableModel, WrapperModel, DataParallelModel
from models.utils import *


import PIL
from PIL import Image

import argparse
import os.path
from pathlib import Path
import glob
import sys
import pdb
import tqdm


parser = argparse.ArgumentParser(description='Visualize output for a single Task')

parser.add_argument('--task', dest='task', help="normal, depth or reshading")
parser.set_defaults(task='NONE')

parser.add_argument('--img_path', dest='img_path', help="path to rgb image")
parser.set_defaults(im_name='NONE')

parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored")
parser.set_defaults(store_name='NONE')

args = parser.parse_args()

root_dir = './models/pretrained_models_3dcc/'
trans_totensor = transforms.Compose([transforms.Resize(384, interpolation=PIL.Image.BILINEAR),
                                    transforms.CenterCrop(384),
                                    transforms.ToTensor()])
trans_topil = transforms.ToPILImage()

os.system(f"mkdir -p {args.output_path}")

# get target task and model
target_tasks = ['normal']
try:
    task_index = target_tasks.index(args.task)
except:
    print("task should be one of the following: normal, depth, reshading")
    sys.exit()

# Tasko UNet, Tasko DPT, Omni DPT, Omni DPT + 2D3D Augs
models = [UNet(out_channels=3), DPTRegressionModel(num_channels = 6, backbone = 'vitb_rn50_384', non_negative=False) , DPTRegressionModel(num_channels = 3, backbone = 'vitb_rn50_384', non_negative=False), DPTRegressionModel(num_channels = 3, backbone = 'vitb_rn50_384', non_negative=False) ]
#model = models[task_index]

map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')

def save_outputs(img_path, output_file_name):

    img = Image.open(img_path)
    img_tensor = trans_totensor(img)[:3].unsqueeze(0)

    # compute outputs
    for ii, modelname in enumerate(['tasko_unet','tasko_dpt', 'omni_dpt', 'omni_dpt_2d3daug']):
        model = WrapperModel(DataParallelModel(models[ii].to(DEVICE)))
        path = root_dir + 'rgb2normal_'+modelname+'.pth'
        model_state_dict = torch.load(path, map_location=map_location)
        model.load_state_dict(model_state_dict["('rgb', '"+'normal'+"')"])
        if ii > 1 : 
            img_tensor = (img_tensor-0.5)/0.5 #normalize input for omnidata models
        baseline_output = model(img_tensor).clamp(min=0, max=1)
        if ii < 2: baseline_output = baseline_output[:,:3,:,:] #first 3 channels for tasko models
        trans_topil(baseline_output[0]).save(args.output_path+'/'+output_file_name+'_'+args.task+'_'+modelname+'.png')


img_path = Path(args.img_path)
print("Getting Predictions!")
if img_path.is_file():
    save_outputs(args.img_path, os.path.splitext(os.path.basename(args.img_path))[0])
elif img_path.is_dir():
    for f in tqdm.tqdm(glob.glob(args.img_path+'/*')):
        save_outputs(f, os.path.splitext(os.path.basename(f))[0])
else:
    print("invalid file path!")
    sys.exit()

print("Done!")

