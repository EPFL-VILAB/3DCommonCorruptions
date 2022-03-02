import random
import numpy as np
import torch
import pdb

import torchvision.transforms as T
from PIL import Image
import random
import torch.nn.functional as F
import torch
from torch.nn.parallel import parallel_apply
from io import BytesIO
from PIL import Image as PILImage

import skimage as sk
from skimage.filters import gaussian
import cv2
from scipy.ndimage import zoom as scizoom

#based on https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
def jpeg_compression(rgb_batch):
    
    def jpeg_compression_(x):
        # c = [25, 18, 15, 10, 7][severity - 1]
        x = T.ToPILImage()(x.squeeze())
        c=random.randint(1,25)

        output = BytesIO()
        x.save(output, 'JPEG', quality=c)
        x = PILImage.open(output)
        x = T.ToTensor()(x)
        return x.unsqueeze(0)       
       
    modules = [jpeg_compression_ for _ in range(len(rgb_batch))]
    args = [(rgb_batch[i]) for i in range(len(rgb_batch))]
    outputs = parallel_apply(modules, args)
    outputs = torch.stack(outputs, dim = 0)  
    return outputs.cuda().squeeze()


def pixelate(rgb_batch):
    
    def pixelate_(x):
        x = T.ToPILImage()(x)
        c=random.uniform(0.1,0.6)

        x = x.resize((int(384 * c), int(384 * c)), PILImage.BOX)
        x = x.resize((384, 384), PILImage.BOX)
        x = Image.fromarray(np.uint8(x))  
        x = T.ToTensor()(x)
        return x.unsqueeze(0)    
    
    modules = [pixelate_ for _ in range(len(rgb_batch))]
    args = [(rgb_batch[i]) for i in range(len(rgb_batch))]
    outputs = parallel_apply(modules, args)
    outputs = torch.stack(outputs, dim = 0)  
    return outputs.cuda().squeeze()

def shot_noise(rgb_batch):
    
    def shot_noise_(x):
        x = T.ToPILImage()(x)
        c=random.randint(1,60)

        x = np.array(x) / 255.
        x = np.clip(np.random.poisson(x * c) / c, 0, 1) * 255

        x = Image.fromarray(np.uint8(x)) 
        x = T.ToTensor()(x)
        return x.unsqueeze(0)   
    
    modules = [shot_noise_ for _ in range(len(rgb_batch))]
    args = [(rgb_batch[i]) for i in range(len(rgb_batch))]
    outputs = parallel_apply(modules, args)
    outputs = torch.stack(outputs, dim = 0)  
    return outputs.cuda().squeeze()

def impulse_noise(rgb_batch):
    
    def impulse_noise_(x):
        x = T.ToPILImage()(x)
        #c = random.random(0,0.27)
        c = random.uniform(0.02,0.27)

        x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
        x = np.clip(x, 0, 1) * 255
        x = Image.fromarray(np.uint8(x))
        x = T.ToTensor()(x)
        return x.unsqueeze(0)    
    
    modules = [impulse_noise_ for _ in range(len(rgb_batch))]
    args = [(rgb_batch[i]) for i in range(len(rgb_batch))]
    outputs = parallel_apply(modules, args)
    outputs = torch.stack(outputs, dim = 0)  
    return outputs.cuda().squeeze()


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

def defocus_blur(rgb_batch):
    
    def defocus_blur_(x):
        x = T.ToPILImage()(x)
        c1=random.randint(0,10)
        c2=random.uniform(0,0.5)
        c=(c1,c2)

        # c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

        x = np.array(x) / 255.
        kernel = disk(radius=c[0], alias_blur=c[1])

        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3
        x = np.clip(channels, 0, 1) * 255

        x = Image.fromarray(np.uint8(x))
        x = T.ToTensor()(x)
        return x.unsqueeze(0)    
    
    modules = [defocus_blur_ for _ in range(len(rgb_batch))]
    args = [(rgb_batch[i]) for i in range(len(rgb_batch))]
    outputs = parallel_apply(modules, args)
    outputs = torch.stack(outputs, dim = 0)  
    return outputs.cuda().squeeze()

# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=512, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()

def fog(rgb_batch):
    
    def fog_(x):
        x = T.ToPILImage()(x)
        c1 = random.uniform(1.5,3.5)
        c2 = random.uniform(2, 1.1)
        c = (c1,c2)
            
        #c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]

        x = np.array(x) / 255.
        max_val = x.max()
        x += c[0] * plasma_fractal(wibbledecay=c[1])[:384, :384][..., np.newaxis]
        x = np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255
        x = Image.fromarray(np.uint8(x))
        x = T.ToTensor()(x)
        return x.unsqueeze(0)    
    
    modules = [fog_ for _ in range(len(rgb_batch))]
    args = [(rgb_batch[i]) for i in range(len(rgb_batch))]
    outputs = parallel_apply(modules, args)
    outputs = torch.stack(outputs, dim = 0)  
    return outputs.cuda().squeeze()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


def zoom_blur(rgb_batch):
    
    def zoom_blur_(x):
        x = T.ToPILImage()(x)
        c1 = 1
        c2 = random.uniform(1.08, 1.32)
        c3 = random.uniform(0.08, 0.32)
        
        c = np.arange(c1, c2, c3)
        #c = [np.arange(1, 1.11, 0.01),
        #     np.arange(1, 1.16, 0.01),
        #     np.arange(1, 1.21, 0.02),
        #     np.arange(1, 1.26, 0.02),
        #     np.arange(1, 1.31, 0.03)][severity - 1]

        x = (np.array(x) / 255.).astype(np.float32)
        out = np.zeros_like(x)
        for zoom_factor in c:
            out += clipped_zoom(x, zoom_factor)

        x = (x + out) / (len(c) + 1)
        x = np.clip(x, 0, 1) * 255
        x = Image.fromarray(np.uint8(x))
        x = T.ToTensor()(x)
        return x.unsqueeze(0)    
    
    modules = [zoom_blur_ for _ in range(len(rgb_batch))]
    args = [(rgb_batch[i]) for i in range(len(rgb_batch))]
    outputs = parallel_apply(modules, args)
    outputs = torch.stack(outputs, dim = 0)  
    return outputs.cuda().squeeze()