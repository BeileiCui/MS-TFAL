import torch
import tqdm
import numpy as np
import cv2
import os
import random,argparse
from random import sample
import shutil
from glob import glob

"""
overlay the separate class id mask from one image together
"""

parser = argparse.ArgumentParser(description='overlay sep mask')
parser.add_argument('--ver', type=int, default=0)
cfg = parser.parse_args()

def mask_paste(mask1, mask2):
    mask1_np, mask2_np = np.array(mask1), np.array(mask2)
    # print('np.unique(mask2_np)', np.unique(mask2_np)) # np.unique(mask2_np) [0 2], np.unique(mask2_np) [0 7]
    
    if len(np.unique(mask2_np)) != 1:
        mask1_np[mask2_np!=0]=np.unique(mask2_np)[1]

    # return Image.fromarray(mask1_np.astype(np.uint8)) # Image.fromarray: 是将array转换成image
    return mask1_np.astype(np.uint8)

    
def masks_paste(masks_list):
    num_masks = len(masks_list)
    assert num_masks != 0
    if num_masks == 1: 
        return masks_list[0]
    else: 
        mask = np.zeros(np.array(masks_list[0]).shape, dtype=np.uint8)
        for i in range(num_masks):
            mask = mask_paste(mask, masks_list[i])
        return mask


noisy_mask_save_dir = './dataset/endovis18/train_noisy_label/noisy_scene_labels_sep_mask_v' + str(cfg.ver)
noisy_final_mask_dir = './dataset/endovis18/train_noisy_label/noisy_scene_labels_final_mask_v' + str(cfg.ver)

all_frame_list = []
for seq in tqdm.tqdm(range(1, 17)):
    print('seq', seq)
    if seq == 8:
        continue
    frame_list = os.listdir(os.path.join(noisy_mask_save_dir, 'seq_'+str(seq)))
    # print('frame_list', frame_list)
    all_frame_list.extend(frame_list)
    
    for frame in tqdm.tqdm(frame_list):
        label_np_list = []
        frame_path = os.path.join(noisy_mask_save_dir, 'seq_'+str(seq), frame)
        frame_separate_cls = os.listdir(frame_path)
        # print('frame_separate_cls', frame_separate_cls)
        for per_class_mask in  frame_separate_cls:
            per_class_mask_path = os.path.join(frame_path, per_class_mask)
            # print('per_class_mask_path', per_class_mask_path)
            label_np = cv2.imread(per_class_mask_path, cv2.IMREAD_GRAYSCALE)
            label_np_list.append(label_np)
        
        # print('label_np_list', label_np_list)

        final_mask = masks_paste(label_np_list)
        noisy_final_mask_dir_2 = os.path.join(noisy_final_mask_dir, 'seq_'+str(seq))
        if not os.path.exists(noisy_final_mask_dir_2):
            os.makedirs(noisy_final_mask_dir_2)
        # print('saved_img_path', os.path.join(noisy_final_mask_dir_2, frame+'.png'))
        cv2.imwrite(os.path.join(noisy_final_mask_dir_2, frame+'.png'), final_mask)

print('Done overlap ver %d.' % cfg.ver)
