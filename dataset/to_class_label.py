from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.nn.functional import one_hot 
from tqdm import tqdm
from PIL import Image, ImageOps

import albumentations as A
import numpy as np
import torch
import random
import json
import cv2
import os

def to_class_label(mode = 'train'):

    MY_DATA_ROOT = '/mnt/data-hdd/beilei/MSS-TFAL/dataset/endovis18/'
    LABEL_JSON = '/mnt/data-hdd/beilei/MSS-TFAL/dataset/endovis18/train_clean/labels.json'
    class_num = 12
    train = [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16]
    test = [1,2,3,4]

    if mode == 'train':
            train_images = [[s, i]  for s in train for i in range(149)]
            images = train_images

    if mode == 'test':
            test_images = [[s, i] for s in range(1, 2) for i in range(250)] #250
            test_images_2 = [[s, i] for s in range(2,5) for i in range(249)] #249
            images = test_images +test_images_2

    with open(LABEL_JSON, 'r') as f:
        lb_json = json.load(f)
    json_color = [item['color'] for item in lb_json]
    print(json_color)

    if mode == 'train':
        r_lb = os.path.join(MY_DATA_ROOT, 'train_clean/seq_{}/labels/frame{:03d}.png') 
        sav_pt = os.path.join(MY_DATA_ROOT, 'train_clean/seq_{}/labels/grayframe{:03d}.png')
    elif mode == 'test':
        r_lb = os.path.join(MY_DATA_ROOT, 'test_clean/seq_{}/labels/frame{:03d}.png') 
        sav_pt = os.path.join(MY_DATA_ROOT, 'test_clean/seq_{}/labels/grayframe{:03d}.png')

    label_rgb = np.asarray(Image.open(r_lb.format(1, 0)))
    print('label_rgb shape:',label_rgb.shape)
    print('label_rgb unique:',np.unique(label_rgb))

    label_gray = np.zeros(label_rgb.shape[:2])
    for i in range(class_num):
        label_gray[(label_rgb[:,:,:3] == json_color[i]).sum(axis=-1) == 3] = i


    for [ins, frame] in tqdm(images):
            label_rgb = np.asarray(Image.open(r_lb.format(ins, frame)))
            label_gray = np.zeros(label_rgb.shape[:2])
            for i in range(class_num):
                label_gray[(label_rgb[:,:,:3] == json_color[i]).sum(axis=-1) == 3] = i
            cv2.imwrite(sav_pt.format(ins, frame), label_gray)

    label_gray_test1 = np.asarray(Image.open(sav_pt.format(1, 0)))
    print('label_gray shape:',label_gray_test1.shape)
    print('label_gray unique:',np.unique(label_gray_test1))

if __name__ == '__main__':
    to_class_label(mode = 'train')
