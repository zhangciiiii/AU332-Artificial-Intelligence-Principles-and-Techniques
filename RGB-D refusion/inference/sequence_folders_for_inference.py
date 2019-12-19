import torch.utils.data as data
import torch
import numpy as np
import quaternion
from scipy.misc import imread
from path import Path
import random
import os
import math
import matplotlib.pyplot as plt

from PIL import Image



def crawl_folders(root, folders_list, sequence_length):
    '''
    return a list which contains lots of samples : 
    sample = { 'rgb_tgt': rgb_imgs[i], 'rgb_ref_imgs': [], 'depth_tgt': depth_imgs[i], 'depth_ref_imgs': [], 'pose':[3,6]}
    '''
    sequence_set = []
    demi_length = (sequence_length-1)//2
    for folder in folders_list:
        folder = Path(root/folder)
        rgb_folder = folder

        rgb_imgs = sorted(rgb_folder.files('*.png'))

        if len(rgb_imgs) < sequence_length:
            continue

        for i in range(demi_length, len(rgb_imgs)-demi_length):

            sample = { 'rgb_tgt': rgb_imgs[i], 'rgb_ref_imgs': []}
            for j in range(-demi_length, demi_length + 1):
                if j != 0:
                    sample['rgb_ref_imgs'].append(rgb_imgs[i+j])
            sequence_set.append(sample)
    # random.shuffle(sequence_set)
    return sequence_set # 


def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, intrinsics, seed=None, train=True, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.sequence_length = sequence_length
        self.root = Path(root)
        self.scenes = os.listdir(self.root)
        self.samples = crawl_folders(self.root, self.scenes, sequence_length)
        self.transform = transform
        self.intrinsics = intrinsics

    def __getitem__(self, index):
        sample = self.samples[index]
        # print(sample['rgb_tgt'])
        rgb_tgt_img = load_as_float(sample['rgb_tgt'])
        rgb_ref_imgs = [load_as_float(ref_img) for ref_img in sample['rgb_ref_imgs']]

        
        if self.transform is not None:
            imgs, intrinsics = self.transform([rgb_tgt_img] + rgb_ref_imgs, np.copy(self.intrinsics)) # +[mask_tgt_img] + mask_ref_imgs
            rgb_tgt_img = imgs[0]
            rgb_ref_imgs = imgs[1:self.sequence_length]
        

            # print(len(imgs),len(rgb_ref_imgs),len(depth_ref_imgs),)

        
        return rgb_tgt_img, rgb_ref_imgs, intrinsics, np.linalg.inv(intrinsics) # pose_list

    def __len__(self):
        return len(self.samples)
