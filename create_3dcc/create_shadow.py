import sys
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision import transforms
#from skimage.measure import compare_ssim as SSIM
#from skimage.measure import compare_psnr as PSNR
from torch.utils.data import Dataset
from torch.utils import data
from collections import defaultdict
from torch.nn.parallel import parallel_apply
#from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

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



import pdb
def create_shadow_data(BASE_PATH_RGB=None, BASE_PATH_RELIGHTING=None, BASE_TARGET_PATH=None):

    CLASS = ''#'n01440764'

    ## Set corruptions to generate and save images
    corruptions_to_generate = ['shadow']

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
            random_pick = np.random.choice(10,5, replace=False)
            lighting_img_1, lighting_img_2, lighting_img_3, lighting_img_4, lighting_img_5 = np.array(Image.open(lighting_curr[random_pick[0]]))/255., np.array(Image.open(lighting_curr[random_pick[1]]))/255., np.array(Image.open(lighting_curr[random_pick[2]]))/255., np.array(Image.open(lighting_curr[random_pick[3]]))/255. , np.array(Image.open(lighting_curr[random_pick[4]]))/255.       
            lighting_1_region, lighting_2_region, lighting_3_region, lighting_4_region, lighting_5_region = np.mean(lighting_img_1), np.mean(lighting_img_2), np.mean(lighting_img_3), np.mean(lighting_img_4), np.mean(lighting_img_5)
            lightings_region_list = np.array([lighting_1_region, lighting_2_region, lighting_3_region, lighting_4_region, lighting_5_region])
            lightings_region_list_sortind = np.argsort(np.argsort(lightings_region_list))

            lightings_list = [lighting_img_1, lighting_img_2, lighting_img_3, lighting_img_4, lighting_img_5]
            lighting_img_1 = lightings_list[np.where(lightings_region_list_sortind==4)[0].item()]
            lighting_img_2 = lightings_list[np.where(lightings_region_list_sortind==3)[0].item()]
            lighting_img_3 = lightings_list[np.where(lightings_region_list_sortind==2)[0].item()]
            lighting_img_4 = lightings_list[np.where(lightings_region_list_sortind==1)[0].item()]
            lighting_img_5 = lightings_list[np.where(lightings_region_list_sortind==0)[0].item()]
            for s in [1,2,3,4,5]:                
                random_coeff = (np.random.rand(1)*0.1+1.0).item()
                random_coeff_orig = (np.random.rand(1)*0.0+0.0).item()
                if s == 1:
                    im_d_s = im * random_coeff_orig + im * lighting_img_1 * random_coeff 
                    im_d_s = np.clip(im_d_s, a_min=0., a_max=1.)
                if s == 2:
                    im_d_s = im * random_coeff_orig + im * lighting_img_2 * random_coeff
                    im_d_s = np.clip(im_d_s, a_min=0., a_max=1.)
                if s == 3:
                    im_d_s = im * random_coeff_orig + im * lighting_img_3 * random_coeff 
                    im_d_s = np.clip(im_d_s, a_min=0., a_max=1.)
                if s == 4:
                    im_d_s = im * random_coeff_orig + im * lighting_img_4 * random_coeff
                    im_d_s = np.clip(im_d_s, a_min=0., a_max=1.)
                if s == 5:
                    im_d_s = im * random_coeff_orig + im * lighting_img_5 * random_coeff
                    im_d_s = np.clip(im_d_s, a_min=0., a_max=1.)

                    
                im_d_s = Image.fromarray(np.uint8(im_d_s*255))
                f_name = f.split('/')[-1]
                im_d_s.save(f'{BASE_TARGET_PATH}/{d}/{s}/{f_name}')

