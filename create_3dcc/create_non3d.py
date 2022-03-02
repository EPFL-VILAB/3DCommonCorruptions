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

from dataset import RGBDataset


def iso_noise(x, severity=1):
    c_poisson = 25 #ISO don't affect photon noise # [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.
    x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255

    c_gauss = 0.7 * [.08, .12, 0.18, 0.26, 0.38][severity - 1] #ISO increases electronic noise
    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale= c_gauss), 0, 1) * 255
    
    return Image.fromarray(np.uint8(x))

def poisson_gaussian_noise(x, severity=1):
    c_poisson = 10 * [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.
    x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255

    c_gauss = 0.1 * [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale= c_gauss), 0, 1) * 255

    return Image.fromarray(np.uint8(x))

def imadjust(x,a,b,c,d,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y

def low_light(x, severity=1):
    c = [.60, .50, 0.40, 0.30, 0.20][severity - 1]
    x = np.array(x)/255.
    x_scaled = imadjust(x, x.min(), x.max(), 0, c, gamma=2. )*255
    x_scaled = poisson_gaussian_noise(x_scaled, severity=3)
    return x_scaled

def color_quant(x, severity=1):
    bits = 6 - severity
    x = PIL.ImageOps.posterize(x, bits)
    return x


SEVERITIES = [1, 2, 3, 4, 5]
def save_corrupted_batches(loader,save_path=None, CLASS=None):
    
    for idx, group in enumerate(tqdm.tqdm(loader)):
                
        #print(f'PROCESSING BATCH {idx + 1:3d} of {len(loader)}')
        data, paths = group
        rgb_batch = data[:,:3,:,:]
        paths = list(paths)
        #print(f'\tloaded all data types')
        
        for s in SEVERITIES:
                        
            for ii, img in enumerate(rgb_batch):
                img = T.ToPILImage()(img)
                out1, out2, out3 = color_quant(img, severity=s), iso_noise(img, severity=s), low_light(img, severity=s)
                dir_path1 = os.path.join(save_path, 'color_quant', str(s), CLASS)
                dir_path2 = os.path.join(save_path, 'iso_noise', str(s), CLASS)
                dir_path3 = os.path.join(save_path, 'low_light', str(s), CLASS)
                filename = paths[ii]
                
                im_path1 = os.path.join(dir_path1, filename)
                im_path2 = os.path.join(dir_path2, filename)
                im_path3 = os.path.join(dir_path3, filename)
                
                out1.save(im_path1)
                out2.save(im_path2)
                out3.save(im_path3)
            

def create_non3d_data(BASE_PATH_RGB=None, BASE_TARGET_PATH=None, BATCH_SIZE=1):
    CLASS = ''#'n01440764'

    RGB_PATH = os.path.join(BASE_PATH_RGB, CLASS)

    ## Set corruptions to generate and save images
    corruptions_to_generate = ['color_quant','iso_noise','low_light']

    ## Create folders
    #print("creating folders:")
    for corruption in corruptions_to_generate:
        for severity in range(1,6):
            TARGET_PATH = os.path.join(BASE_TARGET_PATH, corruption, str(severity), CLASS)
            if not (os.path.isdir(TARGET_PATH)): os.makedirs(TARGET_PATH)


    rgb_dataset = RGBDataset(RGB_PATH)
    rgb_loader = data.DataLoader(rgb_dataset, batch_size = BATCH_SIZE, 
                         shuffle = False, num_workers = 0, drop_last = False)

       
    save_corrupted_batches(rgb_loader, save_path = BASE_TARGET_PATH, CLASS = CLASS)


