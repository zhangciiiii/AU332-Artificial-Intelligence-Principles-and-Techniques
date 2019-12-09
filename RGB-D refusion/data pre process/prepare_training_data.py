from __future__ import division
import argparse
import scipy.misc
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from path import Path
from PIL import Image
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", metavar='DIR',
                    help='path to original dataset')
parser.add_argument("--with-gt", action='store_true',
                    help="If available (e.g. with KITTI), will store ground truth along with images, for validation")
parser.add_argument("--dump-root", type=str, required=False, help="Where to dump the data")
parser.add_argument("--height", type=int, default=384, help="image height") #from 480 to 384
parser.add_argument("--width", type=int, default=512, help="image width")  #from 640 to 512


args = parser.parse_args()

def main():
    folder_list = os.listdir(args.dataset_dir)

    for one_scene in tqdm(folder_list):
        rgb_image_dir = Path(args.dump_root+'/'+one_scene+'/'+'rgb')
        rgb_image_dir.makedirs_p()
        depth_image_dir = Path(args.dump_root+'/'+one_scene+'/'+'depth')
        depth_image_dir.makedirs_p()

        rgb_images = os.listdir(args.dataset_dir+'/'+one_scene+'/rgb')
        rgb_images = sorted(rgb_images, key=embedded_numbers)
        depth_images = os.listdir(args.dataset_dir+'/'+one_scene+'/depth')
        depth_images = sorted(depth_images, key=embedded_numbers)

        with open(args.dataset_dir+'/'+one_scene+'/groundtruth.txt') as gt:
            gt_list = gt.readlines()
            # Path(args.dump_root+'/'+one_scene+'/'+'pose.txt').makedirs_p()
            with open ((args.dump_root+'/'+one_scene+'/'+'pose.txt'),'w') as pose:
                for line in gt_list[2:]:
                    pose.write(str(line).lstrip('\n'))
        
        scene_id = 0
        for rgb_image, depth_image in tqdm(zip(rgb_images, depth_images)):

            rgb_image = resize_image(args.dataset_dir+'/'+one_scene+'/rgb/'+rgb_image) # numpy array
            depth_image = resize_image(args.dataset_dir+'/'+one_scene+'/depth/'+depth_image)
            Image.fromarray(rgb_image).convert("RGB").save(rgb_image_dir/str(scene_id).zfill(4)+'.jpg')
            Image.fromarray(depth_image).save(depth_image_dir/str(scene_id).zfill(4)+'.jpg')
            scene_id += 1
        




def embedded_numbers(s):
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)           
    pieces[1::2] = map(int, pieces[1::2])    
    return pieces

def resize_image(img_file):
    img = scipy.misc.imread(img_file)
    img = scipy.misc.imresize(img, (args.height, args.width))
    return img

if __name__ == '__main__':
    main()
