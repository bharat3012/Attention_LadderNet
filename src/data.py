"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V
from PIL import Image

import cv2
import numpy as np
import os
import scipy.misc as misc



def default_Brain_loader(img_path, mask_path):
    image = Image.open(img_path)
    image1 = image.resize((256, 256), Image.ANTIALIAS)
    img = np.array(image1)
    
    Mask = Image.open(mask_path)
    Mask1 = Mask.resize((256, 256), Image.ANTIALIAS).convert('L')
    mask = np.array(Mask1)

    img = np.expand_dims(img, axis=2)
    #print(np.min(img), np.max(img))
    #img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 2.0 -1.0
    #mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask = np.array(mask, np.float32) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <0.5] = 0
    # mask = abs(mask-1)
    return img, mask

def read_Brain_datasets(root_path, mode):
    images = []
    masks = []

    if mode=='train':
        image_root = os.path.join(root_path, 'train/image')
        gt_root = os.path.join(root_path, 'train/mask')

    elif mode == 'valid':
        image_root = os.path.join(root_path, 'valid/image')
        gt_root = os.path.join(root_path, 'valid/mask')

    image_dir = os.listdir(image_root)
    #img_dir = [f.lower() for f in image_dir]
    img_sort_dir = sorted(image_dir)

    for image_name in img_sort_dir:
        
        image_path = os.path.join(image_root, image_name)
        label_path = os.path.join(gt_root, image_name.replace('training','manual1'))
        
        images.append(image_path)
        masks.append(label_path)



    return images, masks

class ImageFolder(data.Dataset):

    def __init__(self,root_path, datasets='Messidor', mode = 'train'):
        self.root = root_path
        self.mode = mode
        self.dataset = datasets
        assert self.dataset in ['RIM-ONE', 'Messidor', 'ORIGA', 'Brain', 'Cell', 'Vessel'], \
            "the dataset should be in 'Messidor', 'ORIGA', 'RIM-ONE', 'Vessel' "
        if self.dataset == 'RIM-ONE':
            self.images, self.labels = read_RIM_ONE_datasets(self.root, self.mode)
        elif self.dataset == 'Messidor':
            self.images, self.labels = read_Messidor_datasets(self.root, self.mode)
        elif self.dataset == 'ORIGA':
            self.images, self.labels = read_ORIGA_datasets(self.root, self.mode)
        elif self.dataset == 'Brain':
            self.images, self.labels = read_Brain_datasets(self.root, self.mode)
        elif self.dataset == 'Cell':
            self.images, self.labels = read_Cell_datasets(self.root, self.mode)
        elif self.dataset == 'GAN_Vessel':
            self.images, self.labels = read_datasets_vessel(self.root, self.mode)
        else:
            print('Default dataset is Messidor')
            self.images, self.labels = read_Messidor_datasets(self.root, self.mode)

    def __getitem__(self, index):

        img, mask = default_Brain_loader(self.images[index], self.labels[index])
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)