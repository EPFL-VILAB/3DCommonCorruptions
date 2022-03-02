#!/usr/bin/env python
import os

import torch
import torchvision

import base64
import cupy
import cv2
import flask
import getopt
import gevent
import gevent.pywsgi
import glob
import h5py
import io
import math
import moviepy
import moviepy.editor
import numpy
#import os
import random
import re
import scipy
import scipy.io
import shutil
import sys
import tempfile
import time
import urllib
import zipfile
import tqdm
import parse

import pdb
from PIL import Image
import numpy as np

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

from dataset import RGBAndDepthDataset

### based on https://github.com/sniklaus/3d-ken-burns

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 12) # requires at least pytorch version 1.2.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

objCommon = {}

exec(open('./motion_video/common.py', 'r').read())

exec(open('./motion_video/models/disparity-estimation.py', 'r').read())
exec(open('./motion_video/models/disparity-adjustment.py', 'r').read())
exec(open('./motion_video/models/disparity-refinement.py', 'r').read())
exec(open('./motion_video/models/pointcloud-inpainting.py', 'r').read())

##########################################################
def create_motion_data(BASE_PATH_RGB=None, BASE_PATH_DEPTH=None, BASE_TARGET_PATH=None, BATCH_SIZE=1):

    CLASS = '' #'n01440764'

    RGB_PATH = os.path.join(BASE_PATH_RGB, CLASS)
    DEPTH_PATH = os.path.join(BASE_PATH_DEPTH, CLASS)

    ## Set corruptions to generate and save images
    corruptions_to_generate = ['xy_motion_blur', 'z_motion_blur']

    ## Create folders
    for corruption in corruptions_to_generate:
        for severity in range(1,6):
            TARGET_PATH = os.path.join(BASE_TARGET_PATH, corruption, str(severity), CLASS)
            if not (os.path.isdir(TARGET_PATH)): os.makedirs(TARGET_PATH)
            
    ## Create vid folders
    vids_to_generate = ['motion_blur_vid', 'zoom_blur_vid']
    for vids in vids_to_generate:
        TARGET_PATH = os.path.join(BASE_TARGET_PATH, vids, CLASS)
        if not (os.path.isdir(TARGET_PATH)): os.makedirs(TARGET_PATH)

    #######

    rgb_and_depth_dataset = RGBAndDepthDataset(RGB_PATH, DEPTH_PATH)
    rgb_and_depth_loader = data.DataLoader(rgb_and_depth_dataset, batch_size = BATCH_SIZE, 
                            shuffle = False, num_workers = 0, drop_last = False)

    for idx, group in enumerate(rgb_and_depth_loader):
                
        #print(f'PROCESSING BATCH {idx + 1:3d} of {len(rgb_and_depth_loader)}')
        data_curr, paths = group
        rgb_batch, depth_batch = data_curr[:,:3,:,:], data_curr[:,3,:,:]
        paths = list(paths)
        break

    base_savedir = BASE_TARGET_PATH

    ##########################################################

    #ALL_CLASSES = glob.glob("/datasets/imagenet/val/*/", recursive = False) #e.g. for imagenet
    #ALL_CLASSES = [os.path.basename(os.path.normpath(path)) for path in ALL_CLASSES]
    ALL_CLASSES = [CLASS]

    # Save corruptions for all classes

    for i, CLASS in enumerate(ALL_CLASSES):
        RGB_PATH = os.path.join(BASE_PATH_RGB, CLASS)
        DEPTH_PATH = os.path.join(BASE_PATH_DEPTH, CLASS)
        rgb_and_depth_dataset = RGBAndDepthDataset(RGB_PATH, DEPTH_PATH)
        rgb_and_depth_loader = data.DataLoader(rgb_and_depth_dataset, batch_size = BATCH_SIZE, 
                            shuffle = False, num_workers = 0, drop_last = False)
        
        for idx, group in enumerate(rgb_and_depth_loader):
            #print(f'PROCESSING BATCH {idx + 1:3d} of {len(rgb_and_depth_loader)}')
            data_curr, paths = group
            rgb_batch, depth_batch = data_curr[:,:3,:,:], data_curr[:,3,:,:]
            paths = list(paths)

            all_rgb_files, all_depth_files = rgb_batch, depth_batch

            all_idx = paths

            for iii in tqdm.tqdm(range(len(all_rgb_files))):

                npyImage = all_rgb_files[iii].permute(1,2,0)*255
                npyImage = npyImage.numpy()
                npyImage= np.uint8(npyImage)
                depth_loaded = all_depth_files[iii].squeeze()#*65535

                idx_curr = paths[iii]

                motion_blur_vid_path = f"{base_savedir}/motion_blur_vid/{CLASS}/{idx_curr[:-5]}.mp4"
                zoom_blur_vid_path = f"{base_savedir}/zoom_blur_vid/{CLASS}/{idx_curr[:-5]}.mp4"

                motion_blur_sev1_path = f"{base_savedir}/xy_motion_blur/1/{CLASS}/{idx_curr}"
                motion_blur_sev2_path = f"{base_savedir}/xy_motion_blur/2/{CLASS}/{idx_curr}"
                motion_blur_sev3_path = f"{base_savedir}/xy_motion_blur/3/{CLASS}/{idx_curr}"
                motion_blur_sev4_path = f"{base_savedir}/xy_motion_blur/4/{CLASS}/{idx_curr}"
                motion_blur_sev5_path = f"{base_savedir}/xy_motion_blur/5/{CLASS}/{idx_curr}"

                zoom_blur_sev1_path = f"{base_savedir}/z_motion_blur/1/{CLASS}/{idx_curr}"
                zoom_blur_sev2_path = f"{base_savedir}/z_motion_blur/2/{CLASS}/{idx_curr}"
                zoom_blur_sev3_path = f"{base_savedir}/z_motion_blur/3/{CLASS}/{idx_curr}"
                zoom_blur_sev4_path = f"{base_savedir}/z_motion_blur/4/{CLASS}/{idx_curr}"
                zoom_blur_sev5_path = f"{base_savedir}/z_motion_blur/5/{CLASS}/{idx_curr}"


                intWidth = npyImage.shape[1]
                intHeight = npyImage.shape[0]

                fltRatio = float(intWidth) / float(intHeight)

                intWidth = min(int(1024 * fltRatio), 512) #1024
                intHeight = min(int(1024 / fltRatio), 512) #1024

                process_load(npyImage, {}, depth_loaded=depth_loaded)

                objFrom = {
                    'fltCenterU': intWidth / 2.0,
                    'fltCenterV': intHeight / 2.0,
                    'intCropWidth': int(math.floor(1.00 * intWidth)),
                    'intCropHeight': int(math.floor(1.00 * intHeight))
                }

                objTo = process_autozoom({
                    'fltShift': 100.0,
                    'fltZoom': 1.5,
                    'objFrom': objFrom
                })
                ############################################################
                #motion blur
                if bool(random.getrandbits(1)):
                    random_u = np.random.randint(-75, high=-50) # imagenet one: np.random.randint(-75, high=-50) # more severe one: np.random.randint(-125, high=-100)
                else: 
                    random_u = np.random.randint(50, high=75) # imagenet one: np.random.randint(50, high=75) # more severe one: np.random.randint(100, high=125)
                
                random_v = np.random.randint(-75, high=75) # imagenet one: np.random.randint(-75, high=75) # more severe one: np.random.randint(-125, high=125)

                #print("Motion blur-- u: " + str(random_u) + " v: " + str(random_v) )

                objTo['fltCenterU'] = objFrom['fltCenterU'] + random_u #- 75 #80  # 30 #50
                objTo['fltCenterV'] = objFrom['fltCenterV'] + random_v #+ 75 #+ 150
                objTo['intCropWidth'] = objFrom['intCropWidth'] #* 0.60 #0.80
                objTo['intCropHeight'] = objFrom['intCropHeight'] #* 0.60 #0.80

                npyResult = process_kenburns({
                    'fltSteps': numpy.linspace(0.0, 1.0, 60).tolist(), #75
                    'objFrom': objFrom,
                    'objTo': objTo,
                    'boolInpaint': True #True #False
                })

                moviepy.editor.ImageSequenceClip(sequence=[ npyFrame[:, :, :] for npyFrame in npyResult + list(reversed(npyResult))[1:] ], fps=25).write_videofile(motion_blur_vid_path, logger=None)

                all_frames = np.array(npyResult)
                blurry_frame_1 = np.mean(all_frames[:20,:],0)
                blurry_frame_PIL = Image.fromarray(np.uint8(blurry_frame_1[:,:,:])).convert('RGB')
                #blurry_frame_PIL = blurry_frame_PIL.resize((224,224))
                blurry_frame_PIL.save(motion_blur_sev1_path)

                blurry_frame_2 = np.mean(all_frames[:30,:],0)
                blurry_frame_PIL = Image.fromarray(np.uint8(blurry_frame_2[:,:,:])).convert('RGB')
                #blurry_frame_PIL = blurry_frame_PIL.resize((224,224))
                blurry_frame_PIL.save(motion_blur_sev2_path)

                blurry_frame_3 = np.mean(all_frames[:40,:],0)
                blurry_frame_PIL = Image.fromarray(np.uint8(blurry_frame_3[:,:,:])).convert('RGB')
                #blurry_frame_PIL = blurry_frame_PIL.resize((224,224))
                blurry_frame_PIL.save(motion_blur_sev3_path)

                blurry_frame_4 = np.mean(all_frames[:50,:],0)
                blurry_frame_PIL = Image.fromarray(np.uint8(blurry_frame_4[:,:,:])).convert('RGB')
                #blurry_frame_PIL = blurry_frame_PIL.resize((224,224))
                blurry_frame_PIL.save(motion_blur_sev4_path)

                blurry_frame_5 = np.mean(all_frames[:60,:],0)
                blurry_frame_PIL = Image.fromarray(np.uint8(blurry_frame_5[:,:,:])).convert('RGB')
                #blurry_frame_PIL = blurry_frame_PIL.resize((224,224))
                blurry_frame_PIL.save(motion_blur_sev5_path)


                ############################################################
                #zoom blur
                random_u = np.random.randint(-30, high=30)
                random_v = np.random.randint(-30, high=30)
                random_scale_u = np.random.rand(1)*0.1 + 0.5
                random_scale_v = np.random.rand(1)*0.1 + 0.5

                #print("Zoom blur-- u: " + str(random_u) + " v: " + str(random_v) + " scale_u: " + str(random_scale_u) + " scale_v: " + str(random_scale_v) )

                objTo['fltCenterU'] = objFrom['fltCenterU'] + random_u #+ 75 #80  # 30 #50
                objTo['fltCenterV'] = objFrom['fltCenterV'] + random_v #+ 75 #+ 150
                objTo['intCropWidth'] = objFrom['intCropWidth'] * random_scale_u  #0.60 #0.80
                objTo['intCropHeight'] = objFrom['intCropHeight'] * random_scale_v #0.60 #0.80

                npyResult = process_kenburns({
                    'fltSteps': numpy.linspace(0.0, 1.0, 60).tolist(), #75
                    'objFrom': objFrom,
                    'objTo': objTo,
                    'boolInpaint': True #True #False
                })

                moviepy.editor.ImageSequenceClip(sequence=[ npyFrame[:, :, :] for npyFrame in npyResult + list(reversed(npyResult))[1:] ], fps=25).write_videofile(zoom_blur_vid_path, logger=None)

                all_frames = np.array(npyResult)
                blurry_frame_1 = np.mean(all_frames[:20,:],0)
                blurry_frame_PIL = Image.fromarray(np.uint8(blurry_frame_1[:,:,:])).convert('RGB')
                #blurry_frame_PIL = blurry_frame_PIL.resize((224,224))
                blurry_frame_PIL.save(zoom_blur_sev1_path)

                blurry_frame_2 = np.mean(all_frames[:25,:],0) # more severe one: np.mean(all_frames[:30,:],0)
                blurry_frame_PIL = Image.fromarray(np.uint8(blurry_frame_2[:,:,:])).convert('RGB')
                #blurry_frame_PIL = blurry_frame_PIL.resize((224,224))
                blurry_frame_PIL.save(zoom_blur_sev2_path)

                blurry_frame_3 = np.mean(all_frames[:30,:],0) # more severe one np.mean(all_frames[:40,:],0)
                blurry_frame_PIL = Image.fromarray(np.uint8(blurry_frame_3[:,:,:])).convert('RGB')
                #blurry_frame_PIL = blurry_frame_PIL.resize((224,224))
                blurry_frame_PIL.save(zoom_blur_sev3_path)

                blurry_frame_4 = np.mean(all_frames[:35,:],0) # more severe one: np.mean(all_frames[:50,:],0)
                blurry_frame_PIL = Image.fromarray(np.uint8(blurry_frame_4[:,:,:])).convert('RGB')
                #blurry_frame_PIL = blurry_frame_PIL.resize((224,224))
                blurry_frame_PIL.save(zoom_blur_sev4_path)

                blurry_frame_5 = np.mean(all_frames[:38,:],0) # more severe one: np.mean(all_frames[:60,:],0)
                blurry_frame_PIL = Image.fromarray(np.uint8(blurry_frame_5[:,:,:])).convert('RGB')
                #blurry_frame_PIL = blurry_frame_PIL.resize((224,224))
                blurry_frame_PIL.save(zoom_blur_sev5_path)


               