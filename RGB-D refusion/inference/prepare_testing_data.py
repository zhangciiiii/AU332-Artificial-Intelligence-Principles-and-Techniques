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
parser.add_argument("--height", type=int, default=384, help="image height") #from 720 to 384 , 
parser.add_argument("--width", type=int, default=512, help="image width")  #from 1280 to 960 , the to 512


args = parser.parse_args()

def main():
    img_list = sorted(os.listdir(args.dataset_dir), key=embedded_numbers)
    scene_id = 0

    #---------------------------------

    rgb_image_dir = Path(args.dump_root+'/'+'rgb')
    rgb_image_dir.makedirs_p()

    # -----------load images --------------------
    for img in img_list:
        img = resize_image_rgb(args.dataset_dir+'/'+img)
        Image.fromarray(img).convert("RGB").save(rgb_image_dir/str(scene_id).zfill(4)+'.png')
        scene_id += 1

    '''
    
    '''
                
def embedded_numbers(s):
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)           
    pieces[1::2] = map(int, pieces[1::2])    
    return pieces

def resize_image_rgb(img_file):
    img = scipy.misc.imread(img_file) [:, 160:-160, :]
    print(img.shape)
    img = scipy.misc.imresize(img, (args.height, args.width))
    return img


if __name__ == '__main__':
    main()
