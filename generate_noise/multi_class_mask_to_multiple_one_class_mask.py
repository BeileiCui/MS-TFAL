import torch
import numpy as np
import cv2
import os
import random
import shutil
from glob import glob

"""_summary_
data structure is:
seq id / frame id / class id
"""

def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path):
    mask = cv2.imread(str(path), 0)
    return (mask).astype(np.uint8)

data_dir = './MSS-TFAL/dataset/endovis18/train_clean/' # seq_1, ..., seq_16, skip seq_8
save_dir = './MSS-TFAL/dataset/endovis18/multiple_one_class_mask'

assert os.path.exists(data_dir), \
    'Source data root dir does not exist: {}.'.format(data_dir)

if not os.path.exists(save_dir):
    os.makedirs(save_dir) 

for seq in range(1, 17):
    print('seq', seq)
    if seq == 8:
        continue

    mask_list = []
    mask_dir = os.path.join(data_dir, 'seq_'+str(seq), 'labels/grayframe{:03d}.png')

    for i in range(149):
        frame_path = mask_dir.format(i)
        mask_list.append(frame_path)
    random.shuffle(mask_list)

    # print('mask_list', mask_list)
    for m in range(len(mask_list)):
        print('mask_list[m]:', mask_list[m])
        file_name = os.path.splitext(os.path.basename(mask_list[m]))[0]
        mask_array = load_mask(mask_list[m])
        print(np.unique(mask_array)) # [0 1 2 4] # 
        # print('mask_array shape', mask_array.shape) # (1024, 1280)
        original_height, original_width= mask_array.shape[0], mask_array.shape[1]
        # num_classes_per_mask = len(np.unique(mask_array))

        for i in np.unique(mask_array):
            # print(mask_array == i)
            class_sep_mask = np.zeros((original_height, original_width))
            class_sep_mask[mask_array == i] = i 
            print(np.unique(class_sep_mask))
            # [0.]
            # [0. 1.]
            # [0. 2.]
            # [0. 4.]

            # save_per_class_dir = os.path.join(save_dir, 'seq_'+str(seq), str(i))
            # if not os.path.exists(save_per_class_dir):
            #     os.makedirs(save_per_class_dir)
            # cv2.imwrite(os.path.join(save_dir, 'seq_'+str(seq), str(i), file_name+'.png'), class_sep_mask)

            save_per_class_dir = os.path.join(save_dir, 'seq_'+str(seq), file_name) 
            if not os.path.exists(save_per_class_dir):
                os.makedirs(save_per_class_dir)
            cv2.imwrite(os.path.join(save_dir, 'seq_'+str(seq), file_name, file_name+'_'+str(i)+'.png'), class_sep_mask)




