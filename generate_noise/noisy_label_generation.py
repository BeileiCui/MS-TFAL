import torch
import tqdm
import numpy as np
import cv2
import os
import random
from random import sample
import shutil, argparse
from glob import glob
from noisy_type import add_noise_batch, add_noise_wo_polygon_batch


# ****************** Training settings ****************** # 
parser = argparse.ArgumentParser(description='Noisy data generation')

parser.add_argument('--ver', type=int, default=0)
parser.add_argument('--noisy_ratio', type=int, default=50)
parser.add_argument('--radius_low', type=int, default=40)
parser.add_argument('--radius_up', type=int, default=50)
parser.add_argument('--num_rays_low', type=int, default=4)
parser.add_argument('--num_rays_up', type=int, default=8)
parser.add_argument('--max_rotation_degree', type=int, default=30)
parser.add_argument('--max_translate', type=float, default=0.2)
parser.add_argument('--max_rescale', type=float, default=0.2)
parser.add_argument('--seed', type=int, default=314)

cfg = parser.parse_args()
print(cfg)
# ****************** Training settings ****************** # 

# ****************** Constant Seed ****************** # 
def seed_everything(seed=314): 
    print('Seed everything') 
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed) 
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True
    
if cfg.seed != 0:
    seed_everything(cfg.seed) 
 # ****************** Constant Seed ****************** # 

# ****************** Parameters of noise type ****************** # 
noisy_ratio = cfg.noisy_ratio / 100  # frame-level noise ratio: 30%， 50%， 80%

# dilate， erode
radius = [cfg.radius_low, cfg.radius_up] # [a,b] increase a, increase data noise rate; increase b, increase data noise rate

# polygon
num_rays = [cfg.num_rays_low, cfg.num_rays_up]

# affine
max_rotation_degree=cfg.max_rotation_degree
max_translate=cfg.max_translate
max_rescale=cfg.max_rescale
# ****************** Parameters of noise type ****************** # 

# ****************** Generate and save noise labels ****************** # 
src_mask_dir = './dataset/endovis18/multiple_one_class_mask'
noisy_mask_save_dir = './dataset/endovis18/train_noisy_label/noisy_scene_labels_sep_mask_v' + str(cfg.ver)
if not os.path.exists(noisy_mask_save_dir):
    os.makedirs(noisy_mask_save_dir) 

filename_list = []
seq_n = [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16]
noisy_vedio_num = int(len(seq_n) * noisy_ratio)

noisy_vedio_list = sample(seq_n, noisy_vedio_num)
noisy_vedio_list.sort()
clean_vedio_list = list(set(seq_n) ^ set(noisy_vedio_list))
print('noisy_vedio_num: ',noisy_vedio_num)  #noisy_vedio_num:  9
print('noisy_vedio_list: ',noisy_vedio_list)    #noisy_vedio_list:  [1, 2, 9, 10, 11, 12, 13, 14, 16]
print('clean_vedio_list: ',clean_vedio_list)    #clean_vedio_list:  [3, 4, 5, 6, 7, 15]

for seq in tqdm.tqdm(noisy_vedio_list):
    frame_dir = os.listdir(os.path.join(src_mask_dir, 'seq_'+str(seq)))
    for frame_id in frame_dir:
        frame_path = os.path.join(src_mask_dir, 'seq_'+str(seq), frame_id)
        filename_list.append(frame_path)
    filename_list.sort()
    # print('filename_list example:   ', filename_list[0],'   filename_list length:  ',len(filename_list)) 
    
    # generate frame length for each vedio
    end = 0
    frame_length_list = []
    
    # 随机将一个视频按照随机长度的分段分为若干段，然后每段的长度储存在frame_length_list
    while end < len(filename_list):
        if end >=   len(filename_list) - 6:
            l = len(filename_list) - end
            frame_length_list.append(l)
            end += l
        else:
            l = random.randint(3, 6)
            frame_length_list.append(l)
            end += l
    print('frame_length_list:   ',frame_length_list)
    print('fram sum:    ',sum(frame_length_list))
    end = 0
    #对于每一个长为l的分段
    for l in tqdm.tqdm(frame_length_list):
        #得到l分段下对应的文件名和seq
        file_name = []
        seq_l = []
        for i in range(l):
            
            noisy_filename = filename_list[end + i]
            file_name.append(noisy_filename.split('/')[-1])
            seq_l.append(noisy_filename.split('/')[-2]) # seq_1
            
        # print(file_name)    #['grayframe000', 'grayframe001', 'grayframe002']
        # print(seq_l)    #['seq_2', 'seq_2', 'seq_2']

        #得到l分段下所有的文件名，储存在class_in_l_length中
        class_in_l_length = []
        for i in file_name:
            class_id = os.listdir(os.path.join(src_mask_dir,'seq_'+str(seq), i))
            class_in_l_length += class_id
        # print(class_in_l_length)
        #['grayframe000_7.png', 'grayframe000_3.png', 'grayframe000_0.png', 'grayframe000_6.png', 'grayframe000_4.png', 
        # 'grayframe000_2.png', 'grayframe000_1.png', 'grayframe001_2.png', 'grayframe001_7.png', 'grayframe001_6.png', 
        # 'grayframe001_1.png', 'grayframe001_0.png', 'grayframe001_3.png', 'grayframe001_4.png', 'grayframe002_2.png', 
        # 'grayframe002_1.png', 'grayframe002_4.png', 'grayframe002_0.png', 'grayframe002_7.png', 'grayframe002_6.png', 'grayframe002_3.png']
        
        #得到l分段下包含的类，储存在class_id_total中
        class_id_total = []
        for class_id in class_in_l_length:
            # print('class_id', class_id)
            cls_id = class_id.split('.')[0].split('_')[-1]
            class_id_total.append(cls_id)
            
        class_id_total = np.unique(class_id_total)
        # print(class_id_total)   #['0' '1' '2' '3' '4' '6' '7' '9']
        
        #对l分段下同一个类的frame进行相同的加噪
        for i in class_id_total:
            
            temp_class = []
            temp_label = []
            for j in class_in_l_length:
                cls_id = j.split('.')[0].split('_')[-1]
                if cls_id == i:
                    temp_class.append(j)
                    
            for k in temp_class:
                noisy_filename_path = os.path.join(src_mask_dir,'seq_'+str(seq), k[:8],k)
                src_label_np = cv2.imread(noisy_filename_path, cv2.IMREAD_GRAYSCALE)
                temp_label.append(src_label_np)
            # print(temp_class,len(temp_label))   #['frame147_9.png', 'frame148_9.png'] 2
            
            #批量处理同一个class的label，做同样的加噪
            if i == '0':
                noise_type = 'no_noise'
                for j in range(len(temp_class)):
                    nosiy_label_dir = os.path.join(noisy_mask_save_dir, 'seq_' + str(seq), temp_class[j][:8])
                    # print('we do not add noise to class 0')
                    # print('nosiy_label_path:    ', nosiy_label_dir)
                    if not os.path.exists(nosiy_label_dir):
                        os.makedirs(nosiy_label_dir) 
                    cv2.imwrite(os.path.join(nosiy_label_dir, temp_class[j][:8]+'_'+i+'_'+noise_type+'.png'), temp_label[j])
            
            elif i == '6': # "classid": 6 "name": "thread"
                dst_label_np, noise_type = add_noise_wo_polygon_batch(temp_label, radius[0], radius[1], 
                                                    max_rotation_degree, max_translate, max_rescale)
                for j in range(len(temp_class)):
                    nosiy_label_dir = os.path.join(noisy_mask_save_dir, 'seq_' + str(seq), temp_class[j][:8])
                    # print('nosiy_label_path:    ', nosiy_label_dir)
                    if not os.path.exists(nosiy_label_dir):
                        os.makedirs(nosiy_label_dir) 
                    cv2.imwrite(os.path.join(nosiy_label_dir, temp_class[j][:8]+'_'+i+'_'+noise_type+'.png'), dst_label_np[j])
            else:
                dst_label_np, noise_type = add_noise_batch(temp_label, radius[0], radius[1], 
                                                    num_rays[0], num_rays[1],
                                                    max_rotation_degree, max_translate, max_rescale)
                for j in range(len(temp_class)):
                    nosiy_label_dir = os.path.join(noisy_mask_save_dir, 'seq_' + str(seq), temp_class[j][:8])
                    # print('nosiy_label_path:    ', nosiy_label_dir)
                    if not os.path.exists(nosiy_label_dir):
                        os.makedirs(nosiy_label_dir) 
                    cv2.imwrite(os.path.join(nosiy_label_dir, temp_class[j][:8]+'_'+i+'_'+noise_type+'.png'), dst_label_np[j])
        end += l    
                        
    filename_list = []
# ****************** Generate and save noise labels ****************** # 

# ****************** Save clean labels ****************** # 
for seq in clean_vedio_list:
    print('processing clean seq_%d' % seq)
    frame_dir = os.listdir(os.path.join(src_mask_dir, 'seq_'+str(seq)))
    frame_dir.sort()
    for frame_id in frame_dir:
        frame_path = os.path.join(src_mask_dir, 'seq_'+str(seq), frame_id)
        file_name = frame_path.split('/')[-1]
        class_id_list = os.listdir(frame_path)
        for class_id in class_id_list:
            no_noisy_filename_path = os.path.join(frame_path, class_id)
            src_label_np = cv2.imread(no_noisy_filename_path, cv2.IMREAD_GRAYSCALE)
            noise_type = 'no_noise'
            non_nosiy_label_dir = os.path.join(noisy_mask_save_dir, 'seq_'+str(seq), file_name)
            if not os.path.exists(non_nosiy_label_dir):
                os.makedirs(non_nosiy_label_dir) 
            cv2.imwrite(os.path.join(non_nosiy_label_dir, class_id.split('.')[0]+'_'+noise_type+'.png'), src_label_np)

print('Done generation ver %d.' % cfg.ver)
