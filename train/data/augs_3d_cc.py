import random
import numpy as np
import torch
import pdb


import torchvision.transforms as T
from PIL import Image

##########################################################

def fog_3d(rgb, depth):
    
    alpha = (torch.rand(1)*12 + 3).cuda() 
    t = torch.exp(-alpha*depth)   
    #transmission: linear approx
#     t2 = 0.03/depth    
    t_choose = t
    I_s = rgb.mean() #1 #rgb.mean() #atmosphere color
    fog_img = t_choose * rgb + I_s * (1-t_choose)

    return fog_img
    
    
def flash_3d(rgb, reshading):    
    corr_scaler = (torch.rand(1)*0.8 + 0.1).cuda()
    flash_img = rgb + rgb * reshading/corr_scaler
    flash_img = torch.clamp(flash_img, min=0., max=1.)
    
    return flash_img


#####refocus#####
import torch.nn.functional as F
import torch
from torch.nn.parallel import parallel_apply

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

    
    img = F.pad(img, (r_pad_half, r_pad_half, r_pad_half, r_pad_half), "replicate", 0)  # effectively zero padding
    filtered = F.conv2d(img, fil.unsqueeze(1).unsqueeze(-2), bias=None, stride=1, padding=0, dilation=1, groups=n_channels)
    filtered /= filsum
    filtered = F.conv2d(filtered, fil.unsqueeze(1).unsqueeze(-1), bias=None, stride=1, padding=0, dilation=1, groups=n_channels)
    filtered /= filsum

    return filtered

def boxblur(img, r):
    _, _, w, h = img.shape
    f = BoxFilter(r)
    r_pad = int(r * 2)
    r_pad_half = r_pad // 2
    im = F.pad(img, (r_pad_half, r_pad_half, r_pad_half, r_pad_half), "replicate", 0)  # effectively zero padding
    filtered = f(im)
    filtered = filtered[..., r_pad_half:w+r_pad_half, r_pad_half:h+r_pad_half]
    filtered /= ((2 * r + 1) ** 2)
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
    
    blurred_ims = parallel_apply([separable_gaussian]*len(args), args)
    blurred_ims = torch.stack(blurred_ims, dim=1)
    return blurred_ims

def get_blur_stack(rgb, blur_radii, cutoff_multiplier=None):
    args = [(image.unsqueeze(0), radii, cutoff_multiplier) for image, radii in zip(rgb, blur_radii)]
    modules = [get_blur_stack_single_image for _ in args]
    outputs = parallel_apply(modules, args)
    return torch.cat(outputs, dim=0)


def composite_blur_stack(blur_stack, dist_left, dist_right, values_left, values_right):
    shape = list(blur_stack.shape)
    shape[2] = 1
    composite_vals = torch.zeros(shape, dtype=torch.float32, device=blur_stack.device)
    sim_left = (1 - dist_left**2)
    sim_right = (1 - dist_right**2)

    if len(composite_vals.shape) == len(values_left.unsqueeze(1).shape):
        _ = composite_vals.scatter_(1, index=values_left.unsqueeze(1), src=sim_left.unsqueeze(1))
        _ = composite_vals.scatter_(1, index=values_right.unsqueeze(1), src=sim_right.unsqueeze(1))
    else:
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

    # print(f'Quantiles: {quantile_vals}')
    # print(f'Blurs: {blur_radii.shape}')

    blur_stack = get_blur_stack(rgb, blur_radii, cutoff_multiplier=3)
    composited = composite_blur_stack(blur_stack, dist_left, dist_right, calculated_quantiles_left, calculated_quantiles)

    if return_segments:
        return composited, calculated_quantiles_left
    else:
        return composited

# ////////////////////// 3D DEFOCUS V2 MAIN FUNCTIONS /////////////////////

def defocus_blur_3d_random(rgb_batch, depth_batch):
    
    def defocus_blur_3D_image_(rgb, depth):
    
        #severity = np.random.uniform(1., 5.)
        n_quantiles = 8
        return_segments = False

        with torch.no_grad():

            # convert to tensors
            rgb = rgb.unsqueeze(0)
            device = depth.device

            quantiles = torch.arange(0, n_quantiles + 1, device = device) / n_quantiles
            quantiles, quantile_vals = compute_quantiles(depth, quantiles, eps = 0.0001)
            quantile_vals = quantile_vals.permute(1, 0)

            aperture_min = 4. 
            aperture_max = 16.

            log_min = torch.log(torch.tensor(aperture_min, device=device))
            log_max = torch.log(torch.tensor(aperture_max, device=device))
            aperture = torch.exp(torch.rand(size=(rgb.shape[0],1), device=device) * (log_max - log_min) + log_min + log_min )    
            print(aperture)

            # randomly select focal plane index
            focus_dist_idxs = torch.tensor(np.random.randint(low = 0, high = n_quantiles + 1, size = (1,))).cuda()
            focus_dists = torch.gather(quantile_vals, 1, focus_dist_idxs.unsqueeze(0)).permute(1,0) 
            apertures = torch.tensor([aperture], dtype = torch.float32, device = device).unsqueeze(1)

            return refocus_image(rgb, depth, focus_dists, apertures, quantile_vals, return_segments)[0]
    modules = [defocus_blur_3D_image_ for _ in range(len(rgb_batch))]
    args = [(rgb_batch[i], depth_batch[i]) for i in range(len(rgb_batch))]
    outputs = parallel_apply(modules, args)
    
    return torch.stack(outputs, dim = 0)
        


###########################

### 3D motion helpers ###
def compute_quantile_membership_motion(depth, quantile_vals):
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

def composite_blur_stack_motion(blur_stack, dist_left, dist_right, values_left, values_right):
    shape = list(blur_stack.shape)
    shape[2] = 1
    composite_vals = torch.zeros(shape, dtype=torch.float32, device=blur_stack.device)
    sim_left = (1 - dist_left**2)
    sim_right = (1 - dist_right**2)

    if len(composite_vals.shape) == len(values_left.unsqueeze(1).shape):
        _ = composite_vals.scatter_(1, index=values_left.unsqueeze(1), src=sim_left.unsqueeze(1))
        _ = composite_vals.scatter_(1, index=values_right.unsqueeze(1), src=sim_right.unsqueeze(1))
    else:
        _ = composite_vals.scatter_(1, index=values_left.unsqueeze(1).unsqueeze(2), src=sim_left.unsqueeze(1).unsqueeze(2))
        _ = composite_vals.scatter_(1, index=values_right.unsqueeze(1).unsqueeze(2), src=sim_right.unsqueeze(1).unsqueeze(2))

    composite_vals /= composite_vals.sum(dim=1, keepdims=True)
    composited = composite_vals * blur_stack
    composited = composited.sum(dim=1)
    return composited

ident = torch.nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
def get_motion_stack_single_image(rgb, m_blur_radii):
    args = []
    for r in m_blur_radii:
        args.append((rgb))
    m_blur_radii = m_blur_radii.cpu().int()
    #print(m_blur_radii)
    modules = []
    angle_rand = random.uniform(-180., 180.)
    for ii in range(len(args)):
        module_curr = RandomMotionBlur(m_blur_radii[ii].item(), [angle_rand-0.001,angle_rand+0.001], 0.0, p=1.) # for kornia motion aug
        if m_blur_radii[ii]==1: module_curr = ident
        modules.append(module_curr)
    
    blurred_ims = parallel_apply(modules, args)
    blurred_ims = torch.stack(blurred_ims, dim=1)
    return blurred_ims

def get_motion_stack(rgb, m_blur_radii):
    args = [(rgb[i].unsqueeze(0), m_blur_radii[i]) for i in range(len(rgb))]
    
    modules = [get_motion_stack_single_image for _ in args]
    outputs = parallel_apply(modules, args)
    return torch.cat(outputs, dim=0)

def motionblur_image(rgb, depth, n_quantiles, quantile_vals, return_segments=False):
    quantile_vals_squeezed = quantile_vals.squeeze()
    dist_left, dist_right, calculated_quantiles_left, calculated_quantiles = compute_quantile_membership_motion(depth, quantile_vals)
    m_blur_radii = 1. / quantile_vals  #compute_circle_of_confusion_no_magnification(quantile_vals, aperture_size, focus_distance)
    m_blur_radii = m_blur_radii / m_blur_radii.min()
    random_scalar = random.uniform(1.4,2.4) #easier: random.uniform(1,1.8)
    m_blur_radii = random_scalar* m_blur_radii
    m_blur_radii = 2 * (m_blur_radii/2).floor() + 1 #to ensure oddness
    pr = random.random()
    if pr > 0.25: #easier: pr>0.35: 
        m_blur_radii = m_blur_radii + 2
    elif pr > 0.65: #asier: pr > 0.8:
        m_blur_radii = m_blur_radii + 4    

    # print(f'Quantiles: {quantile_vals}')
    # print(f'Blurs: {m_blur_radii.shape}')

    blur_stack = get_motion_stack(rgb, m_blur_radii)
    composited = composite_blur_stack_motion(blur_stack, dist_left, dist_right, calculated_quantiles_left, calculated_quantiles)

    if return_segments:
        return composited, calculated_quantiles_left
    else:
        return composited

from kornia.augmentation import *

def motionkornia_blur_3d(rgb_batch, depth_batch):
    
    def motionkornia_blur_3D_image_(rgb, depth):
    
        n_quantiles = np.random.randint(6., 8.) #8 #RANDOMIZE
        return_segments = False

        with torch.no_grad():

            # convert to tensors
            rgb = rgb.unsqueeze(0)#.cuda()
            device = depth.device

            quantiles = torch.arange(0, n_quantiles + 1, device = device) / n_quantiles
            quantiles, quantile_vals = compute_quantiles(depth, quantiles, eps = 0.0001)
            quantile_vals = quantile_vals.permute(1, 0)

            return motionblur_image(rgb, depth, n_quantiles, quantile_vals, return_segments)[0]
    
    modules = [motionkornia_blur_3D_image_ for _ in range(len(rgb_batch))]
    args = [(rgb_batch[i], depth_batch[i]) for i in range(len(rgb_batch))]
    outputs = parallel_apply(modules, args)
    
    
    return torch.stack(outputs, dim = 0)

######Zoom 3D kornia
from scipy.ndimage import zoom as scizoom
def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]

def zoom_blur_2d(x, blur=1.):
    blur = blur.detach().cpu().numpy().item()

    if blur - 1 < 0.1:
        c = np.arange(1, blur, 0.01)
    elif blur - 1 < 0.2:
        c = np.arange(1, blur, 0.02)
    elif blur - 1 < 0.3:
        c = np.arange(1, blur, 0.03)
    else:
        c = np.arange(1, blur, 0.04)


    x = x.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    x = np.clip(x, 0, 1) #* 255
    
    x = torch.from_numpy(x).cuda().permute(2,0,1).unsqueeze(0)
    
    return x #Image.fromarray(np.uint8(x))

def get_zoom_stack_single_image(rgb, m_blur_radii):
    args = []
    for r in m_blur_radii:
        args.append((rgb, r))
    modules = []
    for ii in range(len(args)):
        module_curr = zoom_blur_2d
        if m_blur_radii[ii]==1: module_curr = ident
        modules.append(module_curr)
    
    blurred_ims = parallel_apply(modules, args)
    blurred_ims = torch.stack(blurred_ims, dim=1)
    return blurred_ims

def get_zoom_stack(rgb, m_blur_radii):
    args = [(rgb[i].unsqueeze(0), m_blur_radii[i]) for i in range(len(rgb))]
    
    modules = [get_zoom_stack_single_image for _ in args]
    outputs = parallel_apply(modules, args)
    return torch.cat(outputs, dim=0)

def zoomblur_image(rgb, depth, n_quantiles, quantile_vals, return_segments=False):
    quantile_vals_squeezed = quantile_vals.squeeze()
    dist_left, dist_right, calculated_quantiles_left, calculated_quantiles = compute_quantile_membership_motion(depth, quantile_vals)
    m_blur_radii = 1. / quantile_vals  #compute_circle_of_confusion_no_magnification(quantile_vals, aperture_size, focus_distance)
    m_blur_radii = m_blur_radii / m_blur_radii.max()
    random_scalar = random.uniform(1.5,4.5) #easier: random.uniform(3.,6.)
    m_blur_radii = m_blur_radii / random_scalar
    m_blur_radii = 1 + m_blur_radii    

    # print(f'Quantiles: {quantile_vals}')
    # print(f'Blurs: {m_blur_radii.shape}')

    blur_stack = get_zoom_stack(rgb, m_blur_radii)
    composited = composite_blur_stack_motion(blur_stack, dist_left, dist_right, calculated_quantiles_left, calculated_quantiles)

    if return_segments:
        return composited, calculated_quantiles_left
    else:
        return composited

def zoomkornia_blur_3d(rgb_batch, depth_batch):
    
    def zoomkornia_blur_3D_image_(rgb, depth):
    
        n_quantiles = np.random.randint(6., 8.) #8 #RANDOMIZE
        return_segments = False

        with torch.no_grad():

            # convert to tensors
            rgb = rgb.unsqueeze(0)#.cuda()
            device = depth.device

            quantiles = torch.arange(0, n_quantiles + 1, device = device) / n_quantiles
            quantiles, quantile_vals = compute_quantiles(depth, quantiles, eps = 0.0001)
            quantile_vals = quantile_vals.permute(1, 0)

            return zoomblur_image(rgb, depth, n_quantiles, quantile_vals, return_segments)[0]
    
    modules = [zoomkornia_blur_3D_image_ for _ in range(len(rgb_batch))]
    args = [(rgb_batch[i], depth_batch[i]) for i in range(len(rgb_batch))]
    outputs = parallel_apply(modules, args)
    
    
    return torch.stack(outputs, dim = 0)
