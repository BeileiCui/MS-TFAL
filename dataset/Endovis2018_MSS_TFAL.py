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

LABEL_JSON = './MSS-TFAL/dataset/endovis18/train_clean/labels.json'
DATA_ROOT = './MSS-TFAL/dataset/endovis18/'

Procedures = {'train':[1,2,3,4,5,6,7,9,10,11,12,13,14,15,16]}

class endovis2018(Dataset):

    def __init__(self, split, t=1, arch='swinPlus', rate=1, global_n=0, data_ver=0, h=512, w=640):
        super(endovis2018, self).__init__()

        self.class_num = 12

        self.mode = split
        self.t = t
        self.global_n = global_n
        self.rate = rate
        self.arch = arch
        self.data_ver = data_ver
        if self.arch == 'swinPlus':
            self.base_size = {'h': 540, 'w': 672} 
            self.crop_size = {'h': h, 'w': w} 
        else:
            self.im_size = {'h': 512, 'w': 640}

        if self.mode == 'train':
            train_images = [[i,f]  for f in Procedures['train'] for i in range(149)]
            self.images = train_images
            print(f'Loaded {len(self.images)} noisy frames. For {self.mode}')

        self.train_clean = (split =='train_clean')
        if self.train_clean:
            train_images = [[i,f] for f in Procedures['train'] for i in range(149)]
            self.images = train_images
            print(f'Loaded {len(self.images)} clean frames. For {self.mode}')
            
        self.test = (split =='test')
        if self.test:
            test_images = [[i,s] for s in range(1, 2) for i in range(250) ] #250
            test_images_2 = [[i,s] for s in range(2,5) for i in range(249)] #249
            self.images = test_images +test_images_2
            print(f'Loaded {len(self.images)} noisy frames. For {self.mode}')

        self.test_part = (split =='test_part')
        if self.test_part:
            test_images = [[i,s] for s in range(1, 2) for i in range(1,250,4)] #250
            test_images_2 = [[i,s] for s in range(2,5) for i in range(1,249,4)] #249
            self.images = test_images +test_images_2
            print(f'Loaded {len(self.images)} noisy frames. For {self.mode}')
            
        self.num_samples = len(self.images)

        with open(LABEL_JSON, 'r') as f:
            self.lb_json = json.load(f)
        self.json_color = [item['color'] for item in self.lb_json]
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        frame,ins,  = self.images[idx]
        images, images_1, label, label_1= self._load_data(ins, frame, self.t, self.global_n)
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
            images = images[0].transpose(2,0,1)     # c * w * h
            images_1 = images_1[0].transpose(2,0,1)     # c * w * h
        else:
            images = images.transpose(0,3,1,2)  # t * c * w * h
            images_1 = images_1.transpose(0,3,1,2)  # t * c * w * h

        images = torch.from_numpy(images)
        images_1 = torch.from_numpy(images_1)

        # h, w = label.shape[:2]
        label = torch.from_numpy(label)
        label_1 = torch.from_numpy(label_1)

        # label = one_hot(label.to(torch.int64), num_classes=self.class_num)
        # label = label.permute(2,0,1) 
        # label_1 = one_hot(label_1.to(torch.int64), num_classes=self.class_num)
        # label_1 = label_1.permute(2,0,1) 
        return {'index':idx, 'path': [ins, frame], 'image': images, 'label': label,'image_1': images_1, 'label_1': label_1}

    def _load_data(self, ins, frame, t=1, global_n=0):
        r_im = os.path.join(DATA_ROOT, 'train_clean/seq_{}/left_frames/frame{:03d}.png')
        r_lb = os.path.join(DATA_ROOT, 'train_noisy_label/noisy_scene_labels_final_mask_v' + str(self.data_ver) +'/seq_{}/frame{:03d}.png') #512*640, class num

        if self.train_clean:
            r_im = os.path.join(DATA_ROOT, 'train_clean/seq_{}/left_frames/frame{:03d}.png')
            r_lb = os.path.join(DATA_ROOT, 'train_clean/seq_{}/labels/grayframe{:03d}.png') #512*640, class num
            
        if self.test:
            r_im = os.path.join(DATA_ROOT, 'test_clean/seq_{}/left_frames/frame{:03d}.png') #resized image :512*640
            r_lb = os.path.join(DATA_ROOT, 'test_clean/seq_{}/labels/frame{:03d}.png') #ori resolution

        if self.test_part:
            r_im = os.path.join(DATA_ROOT, 'test_clean/seq_{}/left_frames/frame{:03d}.png') #resized image :512*640
            r_lb = os.path.join(DATA_ROOT, 'test_clean/seq_{}/labels/frame{:03d}.png') #ori resolution

        imgs_t = []
        imgs_t_1 = []
        masks = []
        masks_1 = []

        if frame == 0:
            frame_1 = 1
        else:
            frame_1 = frame - 1

        if (self.mode == 'train') or (self.mode == 'train_clean'):
            if t > frame: #when t > frame index, use future frame
                imgs_t += [Image.open(r_im.format(ins, i))
                        for i in range(frame+t-1, frame-1, -1)]
                masks += [Image.open(r_lb.format(ins, i))
                        for i in range(frame+t-1, frame-1, -1)]
            else:
                imgs_t += [Image.open(r_im.format(ins, i))
                        for i in range(frame-t+1, frame+1)]
                masks += [Image.open(r_lb.format(ins, i))
                        for i in range(frame-t+1, frame+1)]

            if t > frame_1: #when t > frame index, use future frame
                imgs_t_1 += [Image.open(r_im.format(ins, i))
                        for i in range(frame_1+t-1, frame_1-1, -1)]
                masks_1 += [Image.open(r_lb.format(ins, i))
                        for i in range(frame_1+t-1, frame_1-1, -1)]
            else:
                imgs_t_1 += [Image.open(r_im.format(ins, i))
                        for i in range(frame_1-t+1, frame_1+1)]
                masks_1 += [Image.open(r_lb.format(ins, i))
                        for i in range(frame_1-t+1, frame_1+1)]

            for i in range(len(imgs_t)):
                imgs_t[i] = imgs_t[i].resize((self.crop_size['w'], self.crop_size['h']), Image.Resampling.BILINEAR)
                masks[i] = masks[i].resize((self.crop_size['w'], self.crop_size['h']), Image.Resampling.BILINEAR)

            for i in range(len(imgs_t_1)):
                imgs_t_1[i] = imgs_t_1[i].resize((self.crop_size['w'], self.crop_size['h']), Image.Resampling.BILINEAR)
                masks_1[i] = masks_1[i].resize((self.crop_size['w'], self.crop_size['h']), Image.Resampling.BILINEAR)

                
        elif (self.mode == 'test') or (self.mode == 'test_part'):
            imgs_t += [Image.open(r_im.format(ins, frame))]
            imgs_t_1 += [Image.open(r_im.format(ins, frame_1))]
            masks_color = np.asarray(Image.open(r_lb.format(ins, frame))) #1024,1280,4
            masks_color_1 = np.asarray(Image.open(r_lb.format(ins, frame_1))) #1024,1280,4
            masks = np.zeros(masks_color.shape[:2])
            masks_1 = np.zeros(masks_color_1.shape[:2])
            for i in range(self.class_num):
                masks[(masks_color[:,:,:3] == self.json_color[i]).sum(axis=-1) == 3] = i
                masks_1[(masks_color_1[:,:,:3] == self.json_color[i]).sum(axis=-1) == 3] = i
            
            for i in range(len(imgs_t)):
                imgs_t[i] = imgs_t[i].resize((self.crop_size['w'], self.crop_size['h']), Image.Resampling.BILINEAR)
                imgs_t_1[i] = imgs_t_1[i].resize((self.crop_size['w'], self.crop_size['h']), Image.Resampling.BILINEAR)

        for i in range(len(imgs_t)):
            imgs_t[i] = np.array(imgs_t[i])
            imgs_t_1[i] = np.array(imgs_t_1[i])
            masks[i] = np.array(masks[i])
            masks_1[i] = np.array(masks_1[i])


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
                        interpolation=cv2.INTER_LINEAR, 
                    )
                assert img.shape == (512,640,3)
                
                os.makedirs(os.path.dirname(sav_pt), exist_ok=True)
                cv2.imwrite(sav_pt, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])



if __name__ == '__main__':
    # togray()
    h,w = [256,320]
    train = endovis2018('train',t=1,h=256, w=320)
    test = endovis2018('test',t=1,h=256, w=320)
    test_part = endovis2018('test_part',t=1,h=256, w=320)
    train_clean = endovis2018('train_clean',t=1,h=256, w=320)

    print('train image shape:\t', train[0]['image'].shape)
    print('train label shape:\t', train[0]['label'].shape)
    print('train image_1 shape:\t', train[0]['image_1'].shape)
    print('train label_1 shape:\t', train[0]['label_1'].shape)
    print('train index:\t', train[0]['index'])
    print('test image shape:\t', test[0]['image'].shape)
    print('test label shape:\t', test[0]['label'].shape)
    print('test_part image shape:\t', test_part[0]['image'].shape)
    print('test_part label shape:\t', test_part[0]['label'].shape)
    print('train_clean image shape:\t', train_clean[0]['image'].shape)
    print('train_clean label shape:\t', train_clean[0]['label'].shape)
