from __future__ import division
import argparse
import scipy.misc
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from path import Path
from PIL import Image
from PIL.Image import NEAREST, BILINEAR
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", metavar='DIR',
                    help='path to original dataset')
parser.add_argument("--dump-root", type=str, required=False, help="Where to dump the data")


args = parser.parse_args()

def main():
    rgb_img_list = sorted(os.listdir(args.dataset_dir + '/rgb'), key=embedded_numbers)
    mask_img_list = sorted(os.listdir(args.dataset_dir + '/mask'), key=embedded_numbers)
    scene_id = 0

    #---------------------------------

    dump_dir = Path(args.dump_root+'/'+'result')
    dump_dir.makedirs_p()

    # -----------load images --------------------
    general_mask = np.zeros_like(read_img_mask(args.dataset_dir+'/mask/'+mask_img_list[0]))
    general_rgb = np.zeros_like(read_img_rgb(args.dataset_dir+'/rgb/'+rgb_img_list[0]))
    for rgb,mask in zip(rgb_img_list,mask_img_list):
        rgb = read_img_rgb(args.dataset_dir+'/rgb/'+rgb)
        mask = read_img_mask(args.dataset_dir+'/mask/'+mask)
        
        new_mask = (1-general_mask)*mask
        general_rgb = new_mask[...,np.newaxis]*rgb + general_rgb
        general_mask = general_mask + new_mask
        

        Image.fromarray(general_rgb).convert("RGB").save(dump_dir/str(scene_id).zfill(4)+'.png')
        general_rgb
        scene_id += 1

    '''
    
    '''
                
def embedded_numbers(s):
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)           
    pieces[1::2] = map(int, pieces[1::2])    
    return pieces

def read_img_rgb(img_file):
    img = scipy.misc.imread(img_file)
    return img
def read_img_mask(img_file):
    img = scipy.misc.imread(img_file)
    return np.where((img/255)>0.5, np.ones_like(img), np.zeros_like(img))

if __name__ == '__main__':
    main()