import random
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
from refocus_augmentation import RefocusImageAugmentation
from augs_3d_cc import motionkornia_blur_3d, zoomkornia_blur_3d, flash_3d, fog_3d, defocus_blur_3d_random
from augs_2d_cc import jpeg_compression, pixelate, shot_noise, impulse_noise, defocus_blur, fog, zoom_blur

import pdb
try:
    from kornia.augmentation import *
except:
    print("Error importing kornia augmentation")


class Augmentation:
    def __init__(self):
        pass

    def augment_rgb(self, batch):
            rgb = batch['positive']['rgb']
            depth_euclid = batch['positive']['depth_euclidean']
            reshade = batch['positive']['reshading']

            # 2D Kornia augmentations
            # color jitter
            jitter = ColorJitter(0.15, 0.15, 0.15, 0.15, p=1.)

            # unnormalize before applying augs
            augmented_rgb = (rgb + 1. ) / 2.
            
            p = random.random()
            # base omni augmentations
            if p < 0.4: #0.7:
                p = random.random()
                if p < 0.5:
                    aug = RandomSharpness(.3, p=1.)
                    augmented_rgb = aug(augmented_rgb)

                p = random.random()
                if p < 0.5:
                    aug = RandomMotionBlur((3, 7), random.uniform(10., 50.), 0.5, p=1.)
                    augmented_rgb = aug(augmented_rgb)

                p = random.random()
                if p < 0.1:
                    aug = RandomGaussianBlur((7, 7), (0.1, 2.0), p=1.)
                    augmented_rgb = aug(augmented_rgb)
                elif p < 0.4:
                    aug = RandomGaussianBlur((5, 5), (0.1, 2.0), p=1.)
                    augmented_rgb = aug(augmented_rgb)
                elif p < 0.6:
                    aug = RandomGaussianBlur((3, 3), (0.1, 2.0), p=1.)
                    augmented_rgb = aug(augmented_rgb)

            # New 2D augmentations
            elif p < 0.7:
                try:
                    p = random.random()
                    #print(p)
                    if len(augmented_rgb.size())==3:
                        augmented_rgb = augmented_rgb.cuda().unsqueeze(0)
                    if p < 0.10:
                        aug = RandomGaussianNoise(mean=0., std=random.uniform(.1, .6), p=1.)
                        augmented_rgb = aug(augmented_rgb)
                    elif p < 0.20:
                        augmented_rgb = augmented_rgb.cpu()
                        aug = RandomPosterize(3, p=1.)
                        augmented_rgb = aug(augmented_rgb)
                        augmented_rgb = augmented_rgb.cuda()
                    elif p < 0.30:
                        aug = jitter
                        augmented_rgb = aug(augmented_rgb)
                    elif p < 0.40:
                        aug = jpeg_compression
                        augmented_rgb = aug(augmented_rgb)
                    elif p < 0.50:
                        aug = pixelate
                        augmented_rgb = aug(augmented_rgb)
                    elif p < 0.60:
                        aug = shot_noise
                        augmented_rgb = aug(augmented_rgb)
                    elif p < 0.70:
                        aug = impulse_noise
                        augmented_rgb = aug(augmented_rgb)
                    elif p < 0.80:
                        aug = defocus_blur
                        augmented_rgb = aug(augmented_rgb)
                    elif p < 0.90:
                        aug = fog
                        augmented_rgb = aug(augmented_rgb)
                    else:
                        aug = zoom_blur
                        augmented_rgb = aug(augmented_rgb)
                    print(aug)
                except:
                    print("No 2D aug!")
                    augmented_rgb = augmented_rgb

            # New 3D augmentations
            else:
                p = random.random()
                if len(augmented_rgb.size())==3:
                    augmented_rgb = augmented_rgb.cuda().unsqueeze(0)
                    depth_euclid = depth_euclid.cuda().unsqueeze(0)
                    reshade = reshade.cuda().unsqueeze(0)
                if p < 0.2:
                    print('fog3d')
                    augmented_rgb = fog_3d(augmented_rgb, depth_euclid)
                elif p < 0.4:
                    print('flash')
                    augmented_rgb = flash_3d(augmented_rgb, reshade)
                elif p < 0.6:
                    augmented_rgb = defocus_blur_3d_random(augmented_rgb, depth_euclid) 
                elif p < 0.8:
                    print('motion3d')
                    augmented_rgb = motionkornia_blur_3d(augmented_rgb, depth_euclid)
                else:
                    print('zoom3d')
                    augmented_rgb = zoomkornia_blur_3d(augmented_rgb, depth_euclid)


            # normalize back
            augmented_rgb = (augmented_rgb-0.5) / 0.5
            return augmented_rgb

    def resize_augmentation(self, batch, tasks, fixed_size=None):
        p = random.random()

        if p < 0.4:
            resize_method = 'centercrop'
        elif p < 0.7:
            resize_method = 'randomcrop'
        else:
            resize_method = 'resize'

        if fixed_size is not None: 
            h = fixed_size
            w = fixed_size
        else:
            img_sizes = [256, 320, 384, 448, 512]
            while True:
                h = random.choice(img_sizes)
                w = random.choice(img_sizes)
                if resize_method == 'resize':
                    if h < 1.5 * w and w < 1.5 * h: break
                else:   
                    if h < 2 * w and w < 2 * h: break


        if resize_method == 'randomcrop':
            min_x, min_y = 0, 0
            size_x, size_y = batch[tasks[0]].shape[-2], batch[tasks[0]].shape[-1]
            if size_x != h:
                min_x = random.randrange(0, size_x - h - 2)
            if size_y != w:
                min_y = random.randrange(0, size_y - w - 2)

        for task in tasks:
            if len(batch[task].shape) == 3:
                batch[task] = batch[task].unsqueeze(axis=0)

            if resize_method == 'centercrop':
                centercrop = CenterCrop((h, w), p=1.)
                batch[task] = centercrop(batch[task])

            elif resize_method == 'randomcrop':
                batch[task] = batch[task][:, :, min_x:min_x + h, min_y:min_y + w]

            elif resize_method == 'resize':

                if task == 'rgb':
                    batch[task] = F.interpolate(batch[task], (h, w), mode='bilinear')
                else:
                    batch[task] = F.interpolate(batch[task], (h, w), mode='nearest')

        return batch
