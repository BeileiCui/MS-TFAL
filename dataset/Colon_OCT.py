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

DATA_ROOT = './MSS-TFAL/dataset/Colon_OCT/'

Procedures = {'train':['2T1','2T2','3C1','3C2','3T1','3T2','3T3','3T4','3T5','7C','8C','9C','10C','11C','12C','13C','14C','15C']
                ,'test':['C1','C4','C5','C6','T1']}
Clip_start = {'2T1':13002,'3C1':11002,'3T1':14002,'3T2':14410,'7C':4002,'10C':7002,'13C':20002,'15C':22002,'C1':8002,'C4':1002,'T1':12002}
Procedures_mini = {'train':['2T1','3C1','3T1','3T2','7C','10C','13C','15C']
                ,'test':['C1','C4','T1']}

class Colon_OCT(Dataset):

    def __init__(self, split, t=1, arch='swinPlus', rate=1, global_n=0, data_ver=0, h=256, w=256):
        super(Colon_OCT, self).__init__()

        #self.im_size = {'h': 512, 'w': 640}
        self.class_num = 6

        self.mode = split
        self.t = t
        self.global_n = global_n
        self.rate = rate
        self.arch = arch
        self.data_ver = data_ver
        if self.arch == 'swinPlus':
            self.base_size = {'h': 540, 'w': 672} 
            self.crop_size = {'h': h, 'w': w} 
            self.mask_size = {'h': 1024, 'w': 1024} 
        else:
            self.im_size = {'h': 512, 'w': 640}

        if self.mode == 'train':
            train_images = []
            train_images += [[Procedures['train'][0],i]  for i in range(13005,13251)]
            # train_images += [[Procedures['train'][1],i]  for i in range(13252,13501)]
            train_images += [[Procedures['train'][2],i]  for i in range(11005,11251)]
            # train_images += [[Procedures['train'][3],i]  for i in range(11252,11501)]
            train_images += [[Procedures['train'][4],i]  for i in range(14005,14409)]
            train_images += [[Procedures['train'][5],i]  for i in range(14413,14501)]
            train_images += [[Procedures['train'][5],i]  for i in range(15005,15329)]
            # train_images += [[Procedures['train'][6],i]  for i in range(15330,16225)]
            # train_images += [[Procedures['train'][7],i]  for i in range(16226,17097)]
            # train_images += [[Procedures['train'][8],i]  for i in range(17098,17482)]
            train_images += [[Procedures['train'][9],i]  for i in range(4005,4312)]
            # train_images += [[Procedures['train'][10],i]  for i in range(5002,5388)]
            # train_images += [[Procedures['train'][11],i]  for i in range(6002,6387)]
            train_images += [[Procedures['train'][12],i]  for i in range(7005,7201)]
            train_images += [[Procedures['train'][12],i]  for i in range(7305,7382)]
            # train_images += [[Procedures['train'][13],i]  for i in range(18002,18450)]
            # train_images += [[Procedures['train'][14],i]  for i in range(19002,19450)]
            train_images += [[Procedures['train'][15],i]  for i in range(20005,20201)]
            train_images += [[Procedures['train'][15],i]  for i in range(20305,20381)]
            # train_images += [[Procedures['train'][16],i]  for i in range(21002,21474)]
            train_images += [[Procedures['train'][17],i]  for i in range(22005,22201)]
            train_images += [[Procedures['train'][17],i]  for i in range(22305,22474)]
            self.images = train_images
            print(f'Loaded {len(self.images)} OCT frames. For {self.mode}')
            
        self.test = (split =='test')
        if self.test:
            test_images = []
            test_images += [[Procedures['test'][0],i]  for i in range(8005,8501)]
            test_images += [[Procedures['test'][1],i]  for i in range(1005,1365)]
            # test_images += [[Procedures['test'][2],i]  for i in range(2002,2339)]
            # test_images += [[Procedures['test'][3],i]  for i in range(3002,3312)]
            test_images += [[Procedures['test'][4],i]  for i in range(12005,12501)]
            self.images = test_images
            print(f'Loaded {len(self.images)} OCT frames. For {self.mode}')

        self.test_part = (split =='test_part')
        if self.test_part:
            test_images = []
            test_images += [[Procedures['test'][0],i]  for i in range(8005,8501,5)]
            test_images += [[Procedures['test'][1],i]  for i in range(1005,1365,5)]
            # test_images += [[Procedures['test'][2],i]  for i in range(2002,2339,5)]
            # test_images += [[Procedures['test'][3],i]  for i in range(3002,3312,5)]
            test_images += [[Procedures['test'][4],i]  for i in range(12005,12501,5)]
            self.images = test_images
            print(f'Loaded {len(self.images)} OCT frames. For {self.mode}')
            
        self.num_samples = len(self.images)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        clip,frame  = self.images[idx]
        images, images_1, label, label_1= self._load_data(clip, frame, self.t, self.global_n)
        images = np.array(images).astype('uint8')
        images_1 = np.array(images_1).astype('uint8')
        label = np.array(label).astype('uint8')
        label_1 = np.array(label_1).astype('uint8')
        ###=========augmentation===============
#         if self.mode == 'train':
#             t, h, w, c = images.shape
#             images = images.transpose((1,2,0,3))
#             images = np.ascontiguousarray(images.reshape(h,w,c*t), dtype='uint8')
#             images_1 = images_1.transpose((1,2,0,3))
#             images_1 = np.ascontiguousarray(images_1.reshape(h,w,c*t), dtype='uint8')
#             transf = A.Compose([
# #                 A.HorizontalFlip(p=0.5),
#                 A.VerticalFlip(p=0.5),
#                 A.RandomBrightnessContrast(p=0.5),
#                 A.Rotate()
#             ])
            
#             tsf = transf(image=images, mask=label)
#             tsf_1 = transf(image=images_1, mask=label_1)
#             images = tsf['image'].reshape(self.crop_size['h'], self.crop_size['w'],t,c)
#             images =  np.ascontiguousarray(images.transpose((2,0,1,3)), dtype='float')
#             label = tsf['mask']
#             images_1 = tsf_1['image'].reshape(self.crop_size['h'], self.crop_size['w'],t,c)
#             images_1 =  np.ascontiguousarray(images_1.transpose((2,0,1,3)), dtype='float')
#             label_1 = tsf_1['mask']


        #==========image and label=========
        images = images.astype('float')
        images /= 255.
        images_1 = images_1.astype('float')
        images_1 /= 255.
        if self.t + self.global_n == 1:
            images = torch.from_numpy(images)
            images_1 = torch.from_numpy(images_1)
            images = torch.cat([images,images,images],dim=0) # c * w * h
            images_1 = torch.cat([images_1,images_1,images_1],dim=0) # c * w * h
        else:
            images = torch.from_numpy(images)
            images_1 = torch.from_numpy(images_1)
            images = images.unsqueeze(1)
            images_1 = images_1.unsqueeze(1)
            images = torch.cat([images,images,images],dim=1) # t * c * w * h
            images_1 = torch.cat([images_1,images_1,images_1],dim=1) # t * c * w * h


        # h, w = label.shape[:2]
        label = torch.from_numpy(label)
        label_1 = torch.from_numpy(label_1)

        label = label   # c * w * h
        label_1 = label_1       # c * w * h

        # label = one_hot(label.to(torch.int64), num_classes=self.class_num)
        # label = label.permute(2,0,1) 
        # label_1 = one_hot(label_1.to(torch.int64), num_classes=self.class_num)
        # label_1 = label_1.permute(2,0,1) 
        return {'index':idx, 'path': [clip, frame], 'image': images, 'label': label,'image_1': images_1, 'label_1': label_1}

    def _load_data(self, clip, frame, t=1, global_n=0):
        r_im = os.path.join(DATA_ROOT, 'OCT_train/{}/labelme_json/{}_json/{}_json_img.png')
        r_lb = os.path.join(DATA_ROOT, 'OCT_train/{}/labelme_json/{}_json/{}_json_label.png')
            
        if self.test:
            r_im = os.path.join(DATA_ROOT, 'OCT_test/{}/labelme_json/{}_json/{}_json_img.png')
            r_lb = os.path.join(DATA_ROOT, 'OCT_test/{}/labelme_json/{}_json/{}_json_label.png')

        if self.test_part:
            r_im = os.path.join(DATA_ROOT, 'OCT_test/{}/labelme_json/{}_json/{}_json_img.png')
            r_lb = os.path.join(DATA_ROOT, 'OCT_test/{}/labelme_json/{}_json/{}_json_label.png')

        imgs_t = []
        imgs_t_1 = []
        masks = []
        masks_1 = []
        frame_1 = frame - 1
        frame_diff = frame - Clip_start[clip]
        frame_diff_1 = frame_1 - Clip_start[clip]
        if t > frame_diff: #when t > frame index, use future frame
            imgs_t += [Image.open(r_im.format(clip, i, i))
                     for i in range(frame+t-1, frame-1, -1)]
        else:
            imgs_t += [Image.open(r_im.format(clip, i, i))
                     for i in range(frame-t+1, frame+1)]

        if t > frame_diff_1: #when t > frame index, use future frame
            imgs_t_1 += [Image.open(r_im.format(clip, i, i))
                     for i in range(frame_1+t-1, frame_1-1, -1)]
        else:
            imgs_t_1 += [Image.open(r_im.format(clip, i, i))
                     for i in range(frame_1-t+1, frame_1+1)]

        for i in range(len(imgs_t)):
            imgs_t[i] = imgs_t[i].resize((self.crop_size['w'], self.crop_size['h']), Image.Resampling.BILINEAR)

        for i in range(len(imgs_t_1)):
            imgs_t_1[i] = imgs_t_1[i].resize((self.crop_size['w'], self.crop_size['h']), Image.Resampling.BILINEAR)

        # print(frame,frame_1)
        if (self.mode == 'train'):
            masks = Image.open(r_lb.format(clip, frame, frame))
            masks_1 = Image.open(r_lb.format(clip, frame_1, frame_1))
            masks = masks.resize((self.crop_size['w'], self.crop_size['h']), Image.Resampling.NEAREST)
            masks_1 = masks_1.resize((self.crop_size['w'], self.crop_size['h']), Image.Resampling.NEAREST)

                
        elif (self.mode == 'test') or (self.mode == 'test_part'):
            masks = Image.open(r_lb.format(clip, frame, frame))
            masks_1 = Image.open(r_lb.format(clip, frame_1, frame_1))
            masks = masks.resize((self.mask_size['w'], self.mask_size['h']), Image.Resampling.BILINEAR)
            masks_1 = masks_1.resize((self.mask_size['w'], self.mask_size['h']), Image.Resampling.BILINEAR)
        
        for i in range(len(imgs_t)):
            imgs_t[i] = np.array(imgs_t[i])
            imgs_t_1[i] = np.array(imgs_t_1[i])
        masks = np.array(masks)
        masks_1 = np.array(masks_1)

        return imgs_t, imgs_t_1, masks, masks_1

    def _random_scale(self, imgs, imgs_t, mask, mask_1):
        base_size_w = self.base_size['w']
        crop_size_w = self.crop_size['w']
        crop_size_h = self.crop_size['h']
        # random scale (short edge)

        w, h = imgs[0].size

        long_size = random.randint(int(base_size_w*0.5), int(base_size_w*2.0))
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else: #here
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        for i in range(len(imgs)):
            imgs[i] = imgs[i].resize((ow, oh), Image.Resampling.BILINEAR)
            imgs_t[i] = imgs_t[i].resize((ow, oh), Image.Resampling.BILINEAR)
        mask = mask.resize((ow, oh), Image.Resampling.NEAREST)
        mask_1 = mask.resize((ow, oh), Image.Resampling.NEAREST)
        #print(ow,oh) #926,521
        # pad crop
        if short_size < crop_size_w:
            padh = crop_size_h - oh if oh < crop_size_h else 0
            padw = crop_size_w - ow if ow < crop_size_w else 0
            for i in range(len(imgs)):
                imgs[i] = ImageOps.expand(imgs[i], border=(0, 0, padw, padh), fill=0)
                imgs_t[i] = ImageOps.expand(imgs_t[i], border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            mask_1 = ImageOps.expand(mask_1, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size, if has the previous padding above, then do nothing
        x1 = random.randint(0, w - crop_size_w)
        y1 = random.randint(0, h - crop_size_h)
        for i in range(len(imgs)):
            imgs[i] = np.array(imgs[i].crop((x1, y1, x1+crop_size_w, y1+crop_size_h)))
            imgs_t[i] = np.array(imgs_t[i].crop((x1, y1, x1+crop_size_w, y1+crop_size_h)))
        mask = np.array(mask.crop((x1, y1, x1+crop_size_w, y1+crop_size_h)))
        mask_1 = np.array(mask_1.crop((x1, y1, x1+crop_size_w, y1+crop_size_h)))
        # final transform
        return imgs, imgs_t, mask, mask_1

class Colon_OCT_MSSTFAL(Dataset):

    def __init__(self, split, t=1, arch='swinPlus', rate=1, global_n=0, data_ver=0, h=256, w=256):
        super(Colon_OCT_MSSTFAL, self).__init__()

        #self.im_size = {'h': 512, 'w': 640}
        self.class_num = 6

        self.mode = split
        self.t = t
        self.global_n = global_n
        self.rate = rate
        self.arch = arch
        self.data_ver = data_ver
        if self.arch == 'swinPlus':
            self.base_size = {'h': 540, 'w': 672} 
            self.crop_size = {'h': h, 'w': w} 
            self.mask_size = {'h': 1024, 'w': 1024} 
        else:
            self.im_size = {'h': 512, 'w': 640}

        if self.mode == 'train':
            train_images = []
            train_images += [[Procedures['train'][0],i]  for i in range(13005,13251)]
            # train_images += [[Procedures['train'][1],i]  for i in range(13252,13501)]
            train_images += [[Procedures['train'][2],i]  for i in range(11005,11251)]
            # train_images += [[Procedures['train'][3],i]  for i in range(11252,11501)]
            train_images += [[Procedures['train'][4],i]  for i in range(14005,14409)]
            train_images += [[Procedures['train'][5],i]  for i in range(14413,14501)]
            train_images += [[Procedures['train'][5],i]  for i in range(15005,15329)]
            # train_images += [[Procedures['train'][6],i]  for i in range(15330,16225)]
            # train_images += [[Procedures['train'][7],i]  for i in range(16226,17097)]
            # train_images += [[Procedures['train'][8],i]  for i in range(17098,17482)]
            train_images += [[Procedures['train'][9],i]  for i in range(4005,4312)]
            # train_images += [[Procedures['train'][10],i]  for i in range(5002,5388)]
            # train_images += [[Procedures['train'][11],i]  for i in range(6002,6387)]
            train_images += [[Procedures['train'][12],i]  for i in range(7005,7201)]
            train_images += [[Procedures['train'][12],i]  for i in range(7305,7382)]
            # train_images += [[Procedures['train'][13],i]  for i in range(18002,18450)]
            # train_images += [[Procedures['train'][14],i]  for i in range(19002,19450)]
            train_images += [[Procedures['train'][15],i]  for i in range(20005,20201)]
            train_images += [[Procedures['train'][15],i]  for i in range(20305,20381)]
            # train_images += [[Procedures['train'][16],i]  for i in range(21002,21474)]
            train_images += [[Procedures['train'][17],i]  for i in range(22005,22201)]
            train_images += [[Procedures['train'][17],i]  for i in range(22305,22474)]
            self.images = train_images
            print(f'Loaded {len(self.images)} OCT frames. For {self.mode}')
            
        self.test = (split =='test')
        if self.test:
            test_images = []
            test_images += [[Procedures['test'][0],i]  for i in range(8005,8501)]
            test_images += [[Procedures['test'][1],i]  for i in range(1005,1365)]
            # test_images += [[Procedures['test'][2],i]  for i in range(2002,2339)]
            # test_images += [[Procedures['test'][3],i]  for i in range(3002,3312)]
            test_images += [[Procedures['test'][4],i]  for i in range(12005,12501)]
            self.images = test_images
            print(f'Loaded {len(self.images)} OCT frames. For {self.mode}')

        self.test_part = (split =='test_part')
        if self.test_part:
            test_images = []
            test_images += [[Procedures['test'][0],i]  for i in range(8005,8501,5)]
            test_images += [[Procedures['test'][1],i]  for i in range(1005,1365,5)]
            # test_images += [[Procedures['test'][2],i]  for i in range(2002,2339,5)]
            # test_images += [[Procedures['test'][3],i]  for i in range(3002,3312,5)]
            test_images += [[Procedures['test'][4],i]  for i in range(12005,12501,5)]
            self.images = test_images
            print(f'Loaded {len(self.images)} OCT frames. For {self.mode}')
            
        self.num_samples = len(self.images)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        clip,frame  = self.images[idx]
        images, images_1, label, label_1= self._load_data(clip, frame, self.t, self.global_n)
        images = np.array(images).astype('uint8')
        images_1 = np.array(images_1).astype('uint8')
        label = np.array(label).astype('uint8')
        label_1 = np.array(label_1).astype('uint8')
        ###=========augmentation===============
#         if self.mode == 'train':
#             t, h, w, c = images.shape
#             images = images.transpose((1,2,0,3))
#             images = np.ascontiguousarray(images.reshape(h,w,c*t), dtype='uint8')
#             images_1 = images_1.transpose((1,2,0,3))
#             images_1 = np.ascontiguousarray(images_1.reshape(h,w,c*t), dtype='uint8')
#             transf = A.Compose([
# #                 A.HorizontalFlip(p=0.5),
#                 A.VerticalFlip(p=0.5),
#                 A.RandomBrightnessContrast(p=0.5),
#                 A.Rotate()
#             ])
            
#             tsf = transf(image=images, mask=label)
#             tsf_1 = transf(image=images_1, mask=label_1)
#             images = tsf['image'].reshape(self.crop_size['h'], self.crop_size['w'],t,c)
#             images =  np.ascontiguousarray(images.transpose((2,0,1,3)), dtype='float')
#             label = tsf['mask']
#             images_1 = tsf_1['image'].reshape(self.crop_size['h'], self.crop_size['w'],t,c)
#             images_1 =  np.ascontiguousarray(images_1.transpose((2,0,1,3)), dtype='float')
#             label_1 = tsf_1['mask']


        #==========image and label=========
        images = images.astype('float')
        images /= 255.
        images_1 = images_1.astype('float')
        images_1 /= 255.
        if self.t + self.global_n == 1:
            images = torch.from_numpy(images)
            images_1 = torch.from_numpy(images_1)
            images = torch.cat([images,images,images],dim=0) # c * w * h
            images_1 = torch.cat([images_1,images_1,images_1],dim=0) # c * w * h
        else:
            images = torch.from_numpy(images)
            images_1 = torch.from_numpy(images_1)
            images = images.unsqueeze(1)
            images_1 = images_1.unsqueeze(1)
            images = torch.cat([images,images,images],dim=1) # t * c * w * h
            images_1 = torch.cat([images_1,images_1,images_1],dim=1) # t * c * w * h


        # h, w = label.shape[:2]
        label = torch.from_numpy(label)
        label_1 = torch.from_numpy(label_1)

        label = label   # c * w * h
        label_1 = label_1       # c * w * h

        # label = one_hot(label.to(torch.int64), num_classes=self.class_num)
        # label = label.permute(2,0,1) 
        # label_1 = one_hot(label_1.to(torch.int64), num_classes=self.class_num)
        # label_1 = label_1.permute(2,0,1) 
        return {'index':idx, 'path': [clip, frame], 'image': images, 'label': label,'image_1': images_1, 'label_1': label_1}

    def _load_data(self, clip, frame, t=1, global_n=0):
        r_im = os.path.join(DATA_ROOT, 'OCT_train/{}/labelme_json/{}_json/{}_json_img.png')
        r_lb = os.path.join(DATA_ROOT, 'OCT_train/{}/labelme_json/{}_json/{}_json_label.png')
            
        if self.test:
            r_im = os.path.join(DATA_ROOT, 'OCT_test/{}/labelme_json/{}_json/{}_json_img.png')
            r_lb = os.path.join(DATA_ROOT, 'OCT_test/{}/labelme_json/{}_json/{}_json_label.png')

        if self.test_part:
            r_im = os.path.join(DATA_ROOT, 'OCT_test/{}/labelme_json/{}_json/{}_json_img.png')
            r_lb = os.path.join(DATA_ROOT, 'OCT_test/{}/labelme_json/{}_json/{}_json_label.png')

        imgs_t = []
        imgs_t_1 = []
        masks = []
        masks_1 = []
        frame_1 = frame - 1
        frame_diff = frame - Clip_start[clip]
        frame_diff_1 = frame_1 - Clip_start[clip]
        if (self.mode == 'train'):
            if t > frame_diff: #when t > frame index, use future frame
                imgs_t += [Image.open(r_im.format(clip, i, i))
                        for i in range(frame+t-1, frame-1, -1)]
                masks += [Image.open(r_lb.format(clip, i, i))
                            for i in range(frame+t-1, frame-1, -1)]
            else:
                imgs_t += [Image.open(r_im.format(clip, i, i))
                        for i in range(frame-t+1, frame+1)]
                masks += [Image.open(r_lb.format(clip, i, i))
                            for i in range(frame-t+1, frame+1)]

            if t > frame_diff_1: #when t > frame index, use future frame
                imgs_t_1 += [Image.open(r_im.format(clip, i, i))
                        for i in range(frame_1+t-1, frame_1-1, -1)]
                masks_1 += [Image.open(r_lb.format(clip, i, i))
                            for i in range(frame_1+t-1, frame_1-1, -1)]
            else:
                imgs_t_1 += [Image.open(r_im.format(clip, i, i))
                        for i in range(frame_1-t+1, frame_1+1)]
                masks_1 += [Image.open(r_lb.format(clip, i, i))
                            for i in range(frame_1-t+1, frame_1+1)]

            for i in range(len(imgs_t)):
                imgs_t[i] = imgs_t[i].resize((self.crop_size['w'], self.crop_size['h']), Image.Resampling.BILINEAR)
                masks[i] = masks[i].resize((self.crop_size['w'], self.crop_size['h']), Image.Resampling.BILINEAR)

            for i in range(len(imgs_t_1)):
                imgs_t_1[i] = imgs_t_1[i].resize((self.crop_size['w'], self.crop_size['h']), Image.Resampling.BILINEAR)
                masks_1[i] = masks_1[i].resize((self.crop_size['w'], self.crop_size['h']), Image.Resampling.BILINEAR)
            
            for i in range(len(imgs_t)):
                imgs_t[i] = np.array(imgs_t[i])
                imgs_t_1[i] = np.array(imgs_t_1[i])
                masks[i] = np.array(masks[i])
                masks_1[i] = np.array(masks_1[i])
            

        elif (self.mode == 'test') or (self.mode == 'test_part'):
            imgs_t += [Image.open(r_im.format(clip, frame, frame))]
            imgs_t_1 += [Image.open(r_im.format(clip, frame, frame))]
            masks = Image.open(r_lb.format(clip, frame, frame))
            masks_1 = Image.open(r_lb.format(clip, frame_1, frame_1))
            for i in range(len(imgs_t)):
                imgs_t[i] = imgs_t[i].resize((self.crop_size['w'], self.crop_size['h']), Image.Resampling.BILINEAR)
                imgs_t_1[i] = imgs_t_1[i].resize((self.crop_size['w'], self.crop_size['h']), Image.Resampling.BILINEAR)

            masks = masks.resize((self.mask_size['w'], self.mask_size['h']), Image.Resampling.BILINEAR)
            masks_1 = masks_1.resize((self.mask_size['w'], self.mask_size['h']), Image.Resampling.BILINEAR)

            for i in range(len(imgs_t)):
                imgs_t[i] = np.array(imgs_t[i])
                imgs_t_1[i] = np.array(imgs_t_1[i])
            masks = np.array(masks)
            masks_1 = np.array(masks_1)


        return imgs_t, imgs_t_1, masks, masks_1

    def _random_scale(self, imgs, imgs_t, mask, mask_1):
        base_size_w = self.base_size['w']
        crop_size_w = self.crop_size['w']
        crop_size_h = self.crop_size['h']
        # random scale (short edge)

        w, h = imgs[0].size

        long_size = random.randint(int(base_size_w*0.5), int(base_size_w*2.0))
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else: #here
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        for i in range(len(imgs)):
            imgs[i] = imgs[i].resize((ow, oh), Image.Resampling.BILINEAR)
            imgs_t[i] = imgs_t[i].resize((ow, oh), Image.Resampling.BILINEAR)
        mask = mask.resize((ow, oh), Image.Resampling.NEAREST)
        mask_1 = mask.resize((ow, oh), Image.Resampling.NEAREST)
        #print(ow,oh) #926,521
        # pad crop
        if short_size < crop_size_w:
            padh = crop_size_h - oh if oh < crop_size_h else 0
            padw = crop_size_w - ow if ow < crop_size_w else 0
            for i in range(len(imgs)):
                imgs[i] = ImageOps.expand(imgs[i], border=(0, 0, padw, padh), fill=0)
                imgs_t[i] = ImageOps.expand(imgs_t[i], border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            mask_1 = ImageOps.expand(mask_1, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size, if has the previous padding above, then do nothing
        x1 = random.randint(0, w - crop_size_w)
        y1 = random.randint(0, h - crop_size_h)
        for i in range(len(imgs)):
            imgs[i] = np.array(imgs[i].crop((x1, y1, x1+crop_size_w, y1+crop_size_h)))
            imgs_t[i] = np.array(imgs_t[i].crop((x1, y1, x1+crop_size_w, y1+crop_size_h)))
        mask = np.array(mask.crop((x1, y1, x1+crop_size_w, y1+crop_size_h)))
        mask_1 = np.array(mask_1.crop((x1, y1, x1+crop_size_w, y1+crop_size_h)))
        # final transform
        return imgs, imgs_t, mask, mask_1
# ----------------------------------------------------------------------------


def resize_dataset(src, spt):


    src = [src]
    dst = []
    while src:
        sub = src.pop()
        for item in os.listdir(sub):
            path = os.path.join(sub, item)
            if os.path.isdir(path):
                if item.startswith('seq_'):
                    dst.append(path)
                else:
                    src.append(path)

    for seq in tqdm(dst):
        for key in ['labels', 'left_frames']:
            raw_dir = os.path.join(seq, key)
            sav_dir = os.path.join(spt, os.path.basename(seq), key)
        
            file = [i for i in os.listdir(raw_dir) if i.startswith('frame')]
            print(key, len(file))
            assert len(file) == 149 or len(file) == 249 or len(file) == 250
            
            for item in file:
                raw_pt = os.path.join(raw_dir, item)
                sav_pt = os.path.join(sav_dir, item)
                
                img = cv2.imread(raw_pt)
                assert img.shape == (1024,1280,3)
                
                if key == 'labels':
                    img = img[::2,::2,:]
                else:
                    img = cv2.resize(
                        img, 
                        (1280//2,1024//2), 
                        interpolation=cv2.INTER_LINEAR,  # 双线性插值
                    )
                assert img.shape == (512,640,3)
                
                os.makedirs(os.path.dirname(sav_pt), exist_ok=True)
                cv2.imwrite(sav_pt, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])





if __name__ == '__main__':
    # togray()
    h,w = [256,256]
    train_1 = Colon_OCT('train',t=1,h=h, w=w)
    train_2 = Colon_OCT_MSSTFAL('train',t=4,h=h, w=w)
    test = Colon_OCT('test',t=1,h=h, w=w)
    test_part = Colon_OCT_MSSTFAL('test_part',t=1,h=h, w=w)

    print(train_1.images[0])
    # train_loader = torch.utils.data.DataLoader(train,
    #                              batch_size=1,
    #                              shuffle= True,
    #                              num_workers=3,
    #                              pin_memory=True,
    #                              drop_last=True)

#     for batch_idx, batch in enumerate(train_loader):
#         if batch_idx == 1:
#             print('test break.')
#             break

#         for k in batch:
#             if not k=='path':
# #                     batch[k] = batch[k].to(device=cfg.device, nonw_blocking=True).float()
#                 batch[k] = batch[k].float()
#         batch_index = batch['index'].cpu().numpy().astype(int)
#         print(batch_index)

    print('train 1 image shape:\t', train_1[0]['image'].shape)
    print('train 1 label shape:\t', train_1[0]['label'].shape)
    print('train 1 image shape:\t', train_2[0]['image'].shape)
    print('train 1 label shape:\t', train_2[0]['label'].shape)

    print('test image shape:\t', test[0]['image'].shape)
    print('test label shape:\t', test[0]['label'].shape)
    print('test_part image shape:\t', test_part[0]['image'].shape)
    print('test_part label shape:\t', test_part[0]['label'].shape)
