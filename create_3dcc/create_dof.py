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


## Generate near focus and far focus

def gaussian(M, std, sym=True, device=None):
    
    if M < 1:
        return torch.tensor([])
    if M == 1:
        return torch.ones((1,))
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = torch.arange(0, M, device=device) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    if not sym and not odd:
        w = w[:-1]
        
    return w

def separable_gaussian(img, r=3.5, cutoff=None, device=None):
    
    if device is None:
        device = img.device
    if r < 1e-1:
        return img
    if cutoff is None:
        cutoff = int(r * 5)
        if (cutoff % 2) == 0: cutoff += 1

    assert (cutoff % 2) == 1
    img = img.to(device)
    _, n_channels, w, h = img.shape
    std = r
    fil = gaussian(cutoff, std, device=device).to(device)
    filsum = fil.sum() #/ n_channels
    fil = torch.stack([fil] * n_channels, dim=0)
    r_pad = int(cutoff) 
    r_pad_half = r_pad // 2
    #print(r_pad_half)
    img = F.pad(img, (r_pad_half, r_pad_half, r_pad_half, r_pad_half), "replicate", 0)  # effectively zero padding
    filtered = F.conv2d(img, fil.unsqueeze(1).unsqueeze(-2), bias=None, stride=1, padding=0, dilation=1, groups=n_channels)
    filtered /= filsum
    filtered = F.conv2d(filtered, fil.unsqueeze(1).unsqueeze(-1), bias=None, stride=1, padding=0, dilation=1, groups=n_channels)
    filtered /= filsum

    return filtered


def compute_circle_of_confusion(depths, aperture_size, focal_length, focus_distance):
    
    assert focus_distance > focal_length
    c = aperture_size * torch.abs(depths - focus_distance) / depths * (focal_length / (focus_distance - focal_length))
    
    return c

def compute_circle_of_confusion_no_magnification(depths, aperture_size, focus_distance):
        
    c = aperture_size * torch.abs(depths - focus_distance) / depths 
    
    return c

def compute_quantiles(depth, quantiles, eps=0.0001):
    
    depth_flat = depth.reshape(depth.shape[0], -1)
    quantile_vals = torch.quantile(depth_flat, quantiles, dim=1)
    quantile_vals[0] -= eps
    quantile_vals[-1] += eps
    
    return quantiles, quantile_vals

def compute_quantile_membership(depth, quantile_vals):
    
    quantile_dists = quantile_vals[1:]  - quantile_vals[:-1]
    depth_flat = depth.reshape(depth.shape[0], -1)
    calculated_quantiles = torch.searchsorted(quantile_vals, depth_flat)
    calculated_quantiles_left = calculated_quantiles - 1
    quantile_vals_unsqueezed = quantile_vals #.unsqueeze(-1).unsqueeze(-1)
    quantile_right = torch.gather(quantile_vals_unsqueezed, 1, calculated_quantiles).reshape(depth.shape)
    quantile_left = torch.gather(quantile_vals_unsqueezed, 1, calculated_quantiles_left).reshape(depth.shape)
    quantile_dists = quantile_right - quantile_left
    dist_right = ((quantile_right - depth) / quantile_dists) #/ quantile_dists[calculated_quantiles_left]
    dist_left = ((depth - quantile_left) / quantile_dists)  #/ quantile_dists[calculated_quantiles_left]
    
    return dist_left, dist_right, calculated_quantiles_left.reshape(depth.shape), calculated_quantiles.reshape(depth.shape)

def get_blur_stack_single_image(rgb, blur_radii, cutoff_multiplier):
    
    args = []
    
    for r in blur_radii:
        cutoff = None if cutoff_multiplier is None else int(r * cutoff_multiplier)
        if cutoff is not None and (cutoff % 2) == 0:
            cutoff += 1
        args.append((rgb, r, cutoff))
    
    blurred_ims = []
    
    blurred_ims = parallel_apply([separable_gaussian]*len(args), args)
    blurred_ims = torch.stack(blurred_ims, dim=1)
    
    return blurred_ims

def get_blur_stack(rgb, blur_radii, cutoff_multiplier=None):
    
    args = [(image.unsqueeze(0), radii, cutoff_multiplier) for image, radii in zip(rgb, blur_radii)]
    modules = [get_blur_stack_single_image for _ in args]
    outputs = []
    

    outputs = parallel_apply(modules, args)
    
    return torch.cat(outputs, dim=0)


def composite_blur_stack(blur_stack, dist_left, dist_right, values_left, values_right):
    
    shape = list(blur_stack.shape)
    shape[2] = 1
    composite_vals = torch.zeros(shape, dtype=torch.float32, device=blur_stack.device)
    sim_left = (1 - dist_left**2)
    sim_right = (1 - dist_right**2)
    
    _ = composite_vals.scatter_(1, index=values_left.unsqueeze(1).unsqueeze(2), src=sim_left.unsqueeze(1).unsqueeze(2))
    _ = composite_vals.scatter_(1, index=values_right.unsqueeze(1).unsqueeze(2), src=sim_right.unsqueeze(1).unsqueeze(2))
    
    composite_vals /= composite_vals.sum(dim=1, keepdims=True)
    composited = composite_vals * blur_stack
    composited = composited.sum(dim=1)
    
    return composited


def refocus_image(rgb, depth, focus_distance, aperture_size, quantile_vals, return_segments=False):
    quantile_vals_squeezed = quantile_vals.squeeze()
    dist_left, dist_right, calculated_quantiles_left, calculated_quantiles = compute_quantile_membership(depth, quantile_vals)
    blur_radii = compute_circle_of_confusion_no_magnification(quantile_vals, aperture_size, focus_distance)  
    #print(blur_radii)
    blur_stack = get_blur_stack(rgb, blur_radii, cutoff_multiplier=3)
    composited = composite_blur_stack(blur_stack, dist_left, dist_right, calculated_quantiles_left, calculated_quantiles)

    if return_segments:
        return composited, calculated_quantiles_left
    else:
        return composited
    
def replicate_batch(batch):
        
    batch = batch.unsqueeze(1)

    batch = torch.cat((batch, batch), dim = 1)
    
    return batch

    
def sample_idxs_2(quantiles):
    near = torch.arange(len(quantiles))[:1] #HARDCODED
    near = near[torch.randperm(len(near))[0]]
    
    far = torch.arange(len(quantiles))[7:8] #HARDCODED
    far = far[torch.randperm(len(far))[0]]
    
    return torch.stack((near, far))


def defocus_blur_3D(rgb_batch, 
                    depth_batch, 
                    n_quantiles = 8, 
                    severity = 1, 
                    target_depth = None, 
                    far_focus_penalty = 0.80):
    
    def refocus_image_(rgb, depth, focus_idxs, i, return_segments = False):
        
        with torch.no_grad():
                        
            device = depth.device
            
            quantiles = torch.arange(0, n_quantiles + 1, device = device) / n_quantiles
            depth_normalized = depth            
            
            quantiles, quantile_vals = compute_quantiles(depth_normalized, quantiles, eps = 0.0001)
            quantile_vals = quantile_vals.permute(1, 0)
            
            aperture =  severity + 2. #hard-coded!!
            
            focus_dist_idxs = sample_idxs_2(quantiles).cuda()
            focus_idxs[i] = focus_dist_idxs
            focus_dists = torch.gather(quantile_vals, 1, focus_dist_idxs.unsqueeze(0)).permute(1,0) 
            
            copies_to_return = 2
            apertures = torch.tensor([[aperture]] * copies_to_return, dtype = torch.float32, device = device)
            #apertures[1] = (apertures[1] - 2.5) / 2. #reduce aperture for far focus
            apertures[1] = (apertures[1] - 1.5) / 1. #reduce aperture for far focus
            
            return refocus_image(rgb, depth_normalized, focus_dists, apertures, quantile_vals, return_segments)
    
    batch_size = len(rgb_batch)
    
    rgb_batch, depth_batch = replicate_batch(rgb_batch), replicate_batch(depth_batch)
    
    focus_idxs = [None] * len(rgb_batch)
    
    # apply defocusing to all copies of images in batch
    modules = [refocus_image_ for _ in range(batch_size)]
    args = [(rgb_batch[i], depth_batch[i], focus_idxs, i) for i in range(batch_size)]
    composites = parallel_apply(modules, args)    
    
    # return all focus variants
    outputs = None
    outputs = composites[0].unsqueeze_(0)
    for composite in composites[1:]:
        outputs = torch.cat((outputs, composite.unsqueeze_(0)), dim = 0)
        
    return outputs, focus_idxs

SEVERITIES = [1, 2, 3, 4, 5]
N_QUANTILES = 8 # number of quantiles to partition depth map into        
def save_batch(output, paths, severity, save_path = None, CLASS = None):
    
    types = ['near_focus', 'far_focus']
    count, ig_count = 0, 0
    
    for filename, images in zip(paths, output):
        for type, image in zip(types, images):
            if type not in types: continue
            # make save directory if needed
            dir_path = os.path.join(save_path, type, str(severity), CLASS)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            # check if file already exists
            im_path = os.path.join(dir_path, filename)
            #if os.path.exists(im_path):
                #ig_count += 1
                #continue
            # save file if it doesn't already exist
            im = transforms.ToPILImage()(image)
            im.save(im_path)
            count += 1
            
    #print(f'\t\tsaved {count:3d} images (skipped {ig_count:3d})')

def save_corrupted_batches(loader, save_path= None, CLASS=None):
    
    for idx, group in enumerate(tqdm.tqdm(loader)):
        
        # TODO: check if files are already saved and skip them accordingly
        
        #print(f'PROCESSING BATCH {idx + 1:3d} of {len(loader)}')
        data, paths = group
        rgb_batch, depth_batch = data[:,:3,:,:].cuda(), data[:,3,:,:].cuda()
        paths = list(paths)
        #print(f'\tloaded all data types')
        
        for s in SEVERITIES:
            
            output, f = defocus_blur_3D(rgb_batch, 
                                     depth_batch, 
                                     n_quantiles = N_QUANTILES,
                                     severity = s)
            
            #print(f'\tapplied corruptions (severity = {s})')
            
        
            # save batch
            save_batch(output, paths, s, save_path=save_path, CLASS=CLASS)
            
    return 

def create_dof_data(BASE_PATH_RGB=None, BASE_PATH_DEPTH=None, BASE_TARGET_PATH=None, BATCH_SIZE=1):
    CLASS = ''#'n01440764'

    RGB_PATH = os.path.join(BASE_PATH_RGB, CLASS)
    DEPTH_PATH = os.path.join(BASE_PATH_DEPTH, CLASS)

    ## Set corruptions to generate and save images
    corruptions_to_generate = ['far_focus', 'near_focus']

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
