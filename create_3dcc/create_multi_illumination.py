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

def create_multi_illumination_data(BASE_PATH_RGB=None, BASE_PATH_RELIGHTING=None, BASE_TARGET_PATH=None):

    CLASS = ''#'n01440764'

    ## Set corruptions to generate and save images
    corruptions_to_generate = ['multi_illumination']

    ## Create folders
    #print("creating folders:")
    for corruption in corruptions_to_generate:
        for severity in range(1,6):
            TARGET_PATH = os.path.join(BASE_TARGET_PATH, corruption, str(severity), CLASS)
            # print(TARGET_PATH)
            if not (os.path.isdir(TARGET_PATH)): os.makedirs(TARGET_PATH)
            # print(os.path.isdir(TARGET_PATH))


    all_files = sorted(glob.glob(f'{BASE_PATH_RGB}/*'))
    lighting_files = sorted(glob.glob(f'{BASE_PATH_RELIGHTING}/*'))


    for i, f in enumerate(tqdm.tqdm(all_files)):
        im = Image.open(f)
        lighting_curr = lighting_files[i*10:(i+1)*10]
        im = np.array(im)/255.
        for d in corruptions_to_generate:
            for s in [1,2,3,4,5]:
                if s == 5:
                    random_pick = np.random.randint(0,10,1).item()
                    random_coeff = (np.random.rand(1)*0.1+1.0).item()
                    random_coeff_orig = (np.random.rand(1)*0.0+0.0).item()
                    lighting_img = np.array(Image.open(lighting_curr[random_pick]))/255.
                    im_d_s = im * random_coeff_orig + im * lighting_img * random_coeff
                    im_d_s = np.clip(im_d_s, a_min=0., a_max=1.)

                if s == 4:
                    random_pick = np.random.randint(0,10,2)
                    random_coeff = (np.random.rand(2)*0.1+0.5)
                    random_coeff_orig = (np.random.rand(1)*0.0+0.0).item()
                    lighting_img_1, lighting_img_2 = np.array(Image.open(lighting_curr[random_pick[0]]))/255., np.array(Image.open(lighting_curr[random_pick[1]]))/255.        
                    im_d_s = im * random_coeff_orig + im * lighting_img_1 * random_coeff[0] + im * lighting_img_2 * random_coeff[1] 
                    im_d_s = np.clip(im_d_s, a_min=0., a_max=1.)
                    
                if s == 3:
                    random_pick = np.random.randint(0,10,3)
                    random_coeff = (np.random.rand(3)*0.1+0.35)
                    random_coeff_orig = (np.random.rand(1)*0.0+0.0).item()
                    lighting_img_1, lighting_img_2, lighting_img_3 = np.array(Image.open(lighting_curr[random_pick[0]]))/255., np.array(Image.open(lighting_curr[random_pick[1]]))/255., np.array(Image.open(lighting_curr[random_pick[2]]))/255.       
                    im_d_s = im * random_coeff_orig + im * lighting_img_1 * random_coeff[0] + im * lighting_img_2 * random_coeff[1] + im * lighting_img_3 * random_coeff[2] 
                    im_d_s = np.clip(im_d_s, a_min=0., a_max=1.)
                    
                if s == 2:
                    random_pick = np.random.randint(0,10,4)
                    random_coeff = (np.random.rand(4)*0.1+0.25)
                    random_coeff_orig = (np.random.rand(1)*0.0+0.0).item()
                    lighting_img_1, lighting_img_2, lighting_img_3, lighting_img_4 = np.array(Image.open(lighting_curr[random_pick[0]]))/255., np.array(Image.open(lighting_curr[random_pick[1]]))/255., np.array(Image.open(lighting_curr[random_pick[2]]))/255., np.array(Image.open(lighting_curr[random_pick[3]]))/255.       
                    im_d_s = im * random_coeff_orig + im * lighting_img_1 * random_coeff[0] + im * lighting_img_2 * random_coeff[1] + im * lighting_img_3 * random_coeff[2] + im * lighting_img_4 * random_coeff[3] 
                    im_d_s = np.clip(im_d_s, a_min=0., a_max=1.)
                    
                if s == 1:
                    random_pick = np.random.randint(0,10,5)
                    random_coeff = (np.random.rand(5)*0.1+0.2)
                    random_coeff_orig = (np.random.rand(1)*0.0+0.0).item()
                    lighting_img_1, lighting_img_2, lighting_img_3, lighting_img_4, lighting_img_5 = np.array(Image.open(lighting_curr[random_pick[0]]))/255., np.array(Image.open(lighting_curr[random_pick[1]]))/255., np.array(Image.open(lighting_curr[random_pick[2]]))/255., np.array(Image.open(lighting_curr[random_pick[3]]))/255. , np.array(Image.open(lighting_curr[random_pick[4]]))/255.       
                    im_d_s = im * random_coeff_orig + im * lighting_img_1 * random_coeff[0] + im * lighting_img_2 * random_coeff[1] + im * lighting_img_3 * random_coeff[2] + im * lighting_img_4 * random_coeff[3] + im * lighting_img_5 * random_coeff[4] 
                    im_d_s = np.clip(im_d_s, a_min=0., a_max=1.)
                    
                im_d_s = Image.fromarray(np.uint8(im_d_s*255))
                f_name = f.split('/')[-1]
                im_d_s.save(f'{BASE_TARGET_PATH}/{d}/{s}/{f_name}')
