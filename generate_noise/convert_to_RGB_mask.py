import PIL.Image
import numpy as np
from skimage import io, data, color
from skimage import img_as_ubyte
from tqdm import tqdm
import os
import cv2
import pathlib
import argparse

parser = argparse.ArgumentParser(description='count noisy ratio')
parser.add_argument('--ver', type=int, default=0)
cfg = parser.parse_args()
noisy_final_mask_dir = '/mnt/data-hdd/beilei/MSS-TFAL/dataset/endovis18/train_noisy_label/noisy_scene_labels_final_mask_v' + str(cfg.ver)
RGB_dir = '/mnt/data-hdd/beilei/MSS-TFAL/dataset/endovis18/train_noisy_label_Visualization/noisy_scene_labels_final_mask_v' + str(cfg.ver) + '_RGB'
# factor = 25
factor = 1
#python convert_to_RGB_mask.py --ver 2

# color  = {
#         0*factor: [0, 0, 0],
#         1*factor: [144, 238, 144], # light green
#         2*factor: [153, 255, 255], # light blue
#         3*factor: [0, 102, 255], # dark blue
#         4*factor: [255, 55, 0], # red
#         5*factor: [0, 153, 51], # dark green
#         6*factor: [187, 155, 25], # khaki
#         7*factor: [255, 204, 255], # pink
#         8*factor: [255, 255, 125], # light yellow
#         9*factor: [123, 15, 175], # purple
#         10*factor: [124, 155, 5],
#         11*factor: [125, 255, 12],
# }

color  = {
        0*factor: [0, 0, 0],
        1*factor: [0, 255, 0], # light green
        2*factor: [0, 255, 255], # light blue
        3*factor: [125, 255, 12], # dark blue
        4*factor: [255, 55, 0], # red
        5*factor: [24, 55, 125], # dark green
        6*factor: [187, 155, 25], # khaki
        7*factor: [0, 255, 125], # pink
        8*factor: [255, 255, 125], # light yellow
        9*factor: [123, 15, 175], # purple
        10*factor: [124, 155, 5],
        11*factor: [12, 255, 141],
}

def visual_mask(src_path, dst_path):
    mask = PIL.Image.open(src_path)
    mask = np.asarray(mask)
    # print('mask', mask.shape) # mask (1024, 1280)
    # print('unique', np.unique(mask)) # unique [ 0  1  2  3  4  6  7 10]
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3))

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            color_mask[i, j] = color[mask[i, j]] # # if the mask does not have  RGB channels

            # color_mask[i, j] = color[mask[i, j, 0]] # if the mask has RGB channelss
            
            # if mask[i, j,0] == 11*factor:
            #     print("predict the tumor")

    mask = PIL.Image.fromarray(color_mask.astype(np.uint8))

    # # ==== Resize it into 1024*1280 === #
    # newsize = (1280, 1024) # (1280, 1024), (1024, 1280)
    # mask = mask.resize(newsize)
    # # ================================= #
    mask.save(dst_path)

# 18 prediction, checkpoint from Real train data
os.environ['CUDA_VISIBLE_DEVICES']='1'
seq_n = [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16]
# seq_n = [2, 3, 4, 9]
# seq_n = [1,5,6,7,10,11,12,13,14,15,16]
src_base_path = []
dst_base_path = []

for seq in seq_n:
    path_src = noisy_final_mask_dir + '/seq_' + str(seq) + '/'
    path_rgb = RGB_dir + '/seq_' + str(seq) + '/'
    src_base_path.append(path_src)
    dst_base_path.append(path_rgb)

# print(src_base_path)
# print(dst_base_path)

print("Start🙂")
for i in range(len(dst_base_path)):
    if not os.path.isdir(dst_base_path[i]):
        pathlib.Path(dst_base_path[i]).mkdir(parents=True, exist_ok=True)
    mask_names = os.listdir(src_base_path[i])
    mask_names.sort()
    print('processing seq',seq_n[i], 'seq length: ', len(mask_names))
    for j in tqdm(range(len(mask_names))):
        src_mask = src_base_path[i] + mask_names[j]
        dst_mask = dst_base_path[i] + mask_names[j]
        visual_mask(src_mask, dst_mask)
print("Done😀")
