import torch
import tqdm
import numpy as np
import cv2
import os
import random,argparse
from random import sample
import shutil
from glob import glob

parser = argparse.ArgumentParser(description='count noisy ratio')
parser.add_argument('--ver', type=int, default=0)
cfg = parser.parse_args()

clean_label_dir = './dataset/endovis18/train_clean/'
noisy_label_dir = './dataset/endovis18/train_noisy_label/noisy_scene_labels_final_mask_v' + str(cfg.ver)

classid = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
cls_noisy_ratio =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

num_clean_pixel = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
num_different_pixel = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

dic_cls_noisy_ratio = dict(zip(classid, cls_noisy_ratio))
dic_num_clean_pixel = dict(zip(classid, num_clean_pixel))
dic_different_pixel = dict(zip(classid, num_different_pixel))

def load_mask(path):
    mask = cv2.imread(str(path), 0)
    return (mask).astype(np.uint8)

Total_frame = 0
Total_noisy_pixel = 0
for seq in tqdm.tqdm(range(1, 17)):
    # print('seq', seq)
    if seq == 8:
        continue

    mask_list = []
    mask_dir = os.path.join(clean_label_dir, 'seq_'+str(seq), 'labels/grayframe{:03d}.png')
    for i in range(149):
        frame_path = mask_dir.format(i)
        mask_list.append(frame_path)

    num_frame = len(mask_list)
    Total_frame += num_frame
    # print('frame_dir', frame_dir)
    for frame in tqdm.tqdm(mask_list):
        # print('frame_id', frame_id)
        clean_frame_path = frame
        noisy_frmae_path = os.path.join(noisy_label_dir, 'seq_'+str(seq), frame[-16:])

        clean_np = load_mask(clean_frame_path)
        height, width  = clean_np.shape[0], clean_np.shape[1]
        noisy_np = load_mask(noisy_frmae_path)

        #### count the dataset level noise ratio
        difference = np.argwhere(noisy_np != clean_np) # difference 得到的是被噪声污染的像素的坐标
        # print('difference', difference)
        num_noisy_pixel = len(difference)
        # print('num_noisy_pixel', num_noisy_pixel)
        Total_noisy_pixel += num_noisy_pixel

        #### count the dataset level class specific ratio
        class_set = np.unique(clean_np)
        for cls in class_set:
            if cls == 0:
                continue             
            # cls_clean = np.argwhere(clean_np == cls)
            # cls_noisy = np.argwhere(noisy_np == cls)

            # # d=[y for y in cls_clean if y in cls_noisy] # 找出既在 cls_clean 中的， 又在 cls_noisy 中的像素
            # set1 = set(cls_clean)
            # set2 = set(cls_noisy)
            # unchanged_pixel = set1 & set2 # TypeError: unhashable type: 'numpy.ndarray'

            # print('unchanged_pixel', unchanged_pixel)
            # num_cls_difference = len(cls_clean)-len(unchanged_pixel)  # 这种算法应该是严格正确的，但是太耗时
            
            num_cls_clean = len(np.argwhere(clean_np == cls))
            dic_num_clean_pixel[cls] += num_cls_clean

            num_cls_noisy = len(np.argwhere(noisy_np == cls))
            num_cls_difference = abs(num_cls_noisy - num_cls_clean)
            dic_different_pixel[cls] += num_cls_difference
            

#### count the dataset level noise ratio
noisy_ratio_dataset_level = Total_noisy_pixel / (height*width*Total_frame)
noisy_ratio_dataset_level *= 100
print('Done count noise ratio ver %d.' % cfg.ver)
print('noisy_ratio_dataset_level: %.2f' % noisy_ratio_dataset_level, '%') # 0.033

#### count the dataset level class specific ratio
for id in classid:
    dic_cls_noisy_ratio[id] = (dic_different_pixel[id] * 100) / (dic_num_clean_pixel[id] + dic_different_pixel[id])
    dic_cls_noisy_ratio[id] = round(dic_cls_noisy_ratio[id],2)

print('dic_cls_noisy_ratio: ', dic_cls_noisy_ratio)


