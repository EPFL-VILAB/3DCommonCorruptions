from torch.utils import data
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
from PIL import Image, ImageFile, ImageFilter
import numpy as np
import os
import pickle
import random

###
### Depth Estimation
###
def relative_3siw_depth_collate_fn(batch):
  # Order is image, metric depth gt, mask, relative depth array, dimensions tuple
  return (torch.stack([torch.from_numpy(b[0]) for b in batch], 0), torch.stack([torch.from_numpy(b[1]) for b in batch], 0), torch.stack([torch.from_numpy(b[2]) for b in batch], 0),
          [torch.from_numpy(b[3]) for b in batch], [b[4] for b in batch]  )

###
### Surface Normal Estimation
###
class OASISNormalDataset(data.Dataset):
  def __init__(self, csv_filename, data_aug=False, img_size=256):
    super(OASISNormalDataset, self).__init__()
    print("=====================================================")
    print("Using OASISNormalDataset...")
    print("csv file name: %s" % csv_filename)

    img_names = []
    normal_names = []

    with open(csv_filename) as infile:
      next(infile) # skip header
      for line in infile:
        # Filenames are absolute directories
        img_name,_,_,normal_name,_,_,_,_,_,_,_,_,_,_ = line.split(',')
        if len(normal_name) == 0:
          continue
        img_names.append(os.path.join('/scratch/ainaz', img_name.strip()))
        # print(img_name.split('/')[-1].replace('.png', ''))
        normal_names.append(os.path.join('/scratch/ainaz', normal_name.strip()))
    
    self.img_names = img_names
    self.normal_names = normal_names
    self.width = img_size
    self.height = img_size
    # self.width = 320
    # self.height = 240
    self.n_sample = len(self.img_names)
    # self.split = split
    self.data_aug = data_aug
    print("Network input width = %d, height = %d" % (self.width, self.height))
    print("%d samples" % (self.n_sample))
    print("Data augmentation: {}".format(self.data_aug))
    print("=====================================================")

  def __getitem__(self, index):
    color = cv2.imread(self.img_names[index]).astype(np.float32)
    normal_file = open(self.normal_names[index], 'rb')
    normal_dict = pickle.load(normal_file)
    

    h,w,c = color.shape
    mask = np.zeros((h,w))
    normal = np.zeros((h,w,c))

    # Stuff ROI normal into bounding box
    min_y = normal_dict['min_y']
    max_y = normal_dict['max_y']
    min_x = normal_dict['min_x']
    max_x = normal_dict['max_x']
    roi_normal = normal_dict['normal']
    try:
      normal[min_y:max_y+1, min_x:max_x+1, :] = roi_normal

      normal = normal.astype(np.float32)
    except Exception as e:
      print("Error:", self.normal_names[index])
      print(str(e))
      return 

    # Make mask
    roi_mask = np.logical_or(np.logical_or(roi_normal[:,:,0] != 0, roi_normal[:,:,1] != 0), roi_normal[:,:,2] != 0).astype(np.float32)

    mask[min_y:max_y+1, min_x:max_x+1] = roi_mask


    orig_height = color.shape[0]
    orig_width = color.shape[1]
    # Downsample training images
    color = cv2.resize(color, (self.width, self.height))
    mask = cv2.resize(mask, (self.width, self.height))
    normal = cv2.resize(normal, (self.width, self.height))

    # Data augmentation: randomly flip left to right
    if self.data_aug:
      if random.random() < 0.5:
        color = cv2.flip(color, 1)
        normal = cv2.flip(normal, 1)
        # make sure x coordinates of each vector get flipped
        normal[:,:,0] *= -1

    color = np.transpose(color, (2, 0, 1)) / 255.0  # HWC to CHW.
    normal = np.transpose(normal, (2, 0, 1))   # HWC to CHW.
    # Add one channel b/c Pytorch interpolation requires 4D tensor
    mask = mask[np.newaxis, :, :]

    return color, normal, mask, (orig_height, orig_width)

  def __len__(self):
    return self.n_sample


class OASISNormalDatasetVal(OASISNormalDataset):
  def __init__(self, csv_filename, data_aug=False):
    print("+++++++++++++++++++++++++++++++++++++++++++++++")
    print("Using OASISNormalDatasetVal...")
    print("csv file name: %s" % csv_filename)
    OASISNormalDataset.__init__(self, csv_filename, data_aug=data_aug)

  def __getitem__(self, index):
    color = cv2.imread(self.img_names[index]).astype(np.float32)
    normal_file = open(self.normal_names[index], 'rb')
    normal_dict = pickle.load(normal_file)
    

    h,w,c = color.shape
    mask = np.zeros((h,w))
    normal = np.zeros((h,w,c))

    # Stuff ROI normal into bounding box
    min_y = normal_dict['min_y']
    max_y = normal_dict['max_y']
    min_x = normal_dict['min_x']
    max_x = normal_dict['max_x']
    roi_normal = normal_dict['normal']
    try:
      normal[min_y:max_y+1, min_x:max_x+1, :] = roi_normal

      normal = normal.astype(np.float32)
    except Exception as e:
      print("Error:", self.normal_names[index])
      print(str(e))
      return 

    # Make mask
    roi_mask = np.logical_or(np.logical_or(roi_normal[:,:,0] != 0, roi_normal[:,:,1] != 0), roi_normal[:,:,2] != 0).astype(np.float32)

    mask[min_y:max_y+1, min_x:max_x+1] = roi_mask

    orig_height = color.shape[0]
    orig_width = color.shape[1]

    ######
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    h, w = color.shape[0], color.shape[1]
    if h < w:
        if center_x > w // 2: 
            cropped_rgb = color[0:h, w-h:w]
            cropped_mask = mask[0:h, w-h:w]
            cropped_normal = normal[0:h, w-h:w]
        else: 
            cropped_rgb = color[0:h, 0:h]
            cropped_mask = mask[0:h, 0:h]
            cropped_normal = normal[0:h, 0:h]
    else:
        if center_y > h // 2: 
            cropped_rgb = color[h-w:h, 0:w]
            cropped_mask = mask[h-w:h, 0:w]
            cropped_normal = normal[h-w:h, 0:w]
        else: 
            cropped_rgb = color[0:w, 0:w]
            cropped_mask = mask[0:w, 0:w]
            cropped_normal = normal[0:w, 0:w]
    #####


    # Downsample training images
    color = cv2.resize(cropped_rgb, (self.width, self.height))
   

    color = np.transpose(color, (2, 0, 1)) / 255.0  # HWC to CHW.
    normal = np.transpose(cropped_normal, (2, 0, 1))   # HWC to CHW.
    # Add one channel b/c Pytorch interpolation requires 4D tensor
    mask = cropped_mask[np.newaxis, :, :]

    return color, normal, mask, (orig_height, orig_width), self.img_names[index]

  def __len__(self):
    return self.n_sample
