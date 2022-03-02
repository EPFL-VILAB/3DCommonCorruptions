import sys
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils import data
from collections import defaultdict
from torch.nn.parallel import parallel_apply

import PIL
import random
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import matplotlib.pyplot as plt
import inspect
import natsort
import json
import pdb
import glob
import numpy as np
from fire import Fire
import tqdm

from dataset import RGBAndDepthDataset



def fog_3d(rgb, depth, severity=1):
    
    #transmission: exponential
    alpha = [4,8,12,16,20][severity - 1] 
    #t = torch.exp(-alpha*depth)
    t = torch.exp(-alpha*depth).unsqueeze(1)   
    #transmission: linear approx
#     t2 = 0.03/depth        
    t_choose = t
    I_s = rgb.mean() #1 #rgb.mean() #atmosphere color
    fog_img = t_choose * rgb + I_s * (1-t_choose)
    
    return fog_img


SEVERITIES = [1, 2, 3, 4, 5]
def save_batch(output, paths, severity, save_path = None, CLASS = None):
    
    corruption_curr = 'fog_3d'
    count, ig_count = 0, 0
    
    for filename, images in zip(paths, output):
        # make save directory if needed
        dir_path = os.path.join(save_path, corruption_curr, str(severity), CLASS)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # check if file already exists
        im_path = os.path.join(dir_path, filename)
        #if os.path.exists(im_path):
            #ig_count += 1
            #continue
        # save file if it doesn't already exist
        im = transforms.ToPILImage()(images)
        im.save(im_path)
        count += 1
            
    #print(f'\t\tsaved {count:3d} images (skipped {ig_count:3d})')

def save_corrupted_batches(loader, save_path= None, CLASS=None):
    
    for idx, group in enumerate(tqdm.tqdm(loader)):
                
        #print(f'PROCESSING BATCH {idx + 1:3d} of {len(loader)}')
        data, paths = group
        rgb_batch, depth_batch = data[:,:3,:,:].cuda(), data[:,3,:,:].cuda()
        paths = list(paths)
        #print(f'\tloaded all data types')
        
        for s in SEVERITIES:
            
            output = fog_3d(rgb_batch, depth_batch, severity = s)
        
            #print(f'\tapplied corruptions (severity = {s})')
        
            # save batch
            save_batch(output, paths, s, save_path=save_path, CLASS=CLASS)
            
    return 

def create_fog_data(BASE_PATH_RGB=None, BASE_PATH_DEPTH=None, BASE_TARGET_PATH=None, BATCH_SIZE=1):

    CLASS = ''# or e.g. an imagenet class 'n01440764'

    RGB_PATH = os.path.join(BASE_PATH_RGB, CLASS)
    DEPTH_PATH = os.path.join(BASE_PATH_DEPTH, CLASS)

    ## Set corruptions to generate and save images
    corruptions_to_generate = ['fog_3d']

    ## Create folders
    #print("creating folders:")
    for corruption in corruptions_to_generate:
        for severity in range(1,6):
            TARGET_PATH = os.path.join(BASE_TARGET_PATH, corruption, str(severity), CLASS)
            if not (os.path.isdir(TARGET_PATH)): os.makedirs(TARGET_PATH)

    rgb_and_depth_dataset = RGBAndDepthDataset(RGB_PATH, DEPTH_PATH)
    rgb_and_depth_loader = data.DataLoader(rgb_and_depth_dataset, batch_size = BATCH_SIZE, 
                            shuffle = False, num_workers = 0, drop_last = False)
       
    save_corrupted_batches(rgb_and_depth_loader, save_path = BASE_TARGET_PATH, CLASS = CLASS)
