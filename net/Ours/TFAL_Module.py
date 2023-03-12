import os
import torch
import sys
import numpy as np
sys.path.insert(0,'../MS-TFAL/')

import torch.nn.functional as F
from torch.nn.functional import one_hot 

def TFAL_get_affinity(feature_1, feature_2, label_1, label_2,class_num , H = 256, W = 320):

    label_1 = one_hot(label_1.to(torch.int64), num_classes=class_num)
    label_2 = one_hot(label_2.to(torch.int64), num_classes=class_num)

    label_1 = label_1.permute(0,3,1,2) 
    label_2 = label_2.permute(0,3,1,2) 
    
    Bl, Cl, Hl, Wl = label_1.shape #4, 12, 32, 40
    Bp, Cp, Hp, Wp = feature_1.shape #4, 304, 32, 40

    # initialize output mask
    mask_p_dle = torch.zeros(Bl,Hl * Wl).float().cuda()
    mask_n_dle = torch.zeros(Bl,Hl * Wl).float().cuda()
    mask_ones = torch.ones(Bl,Hl * Wl).float().cuda()

    # compute the simlarities between feature feature_1 and feature_2
    feature_1 = feature_1.view(Bp, Cp, -1) #N,C,HW: 8, 304, 1280
    feature_2 = feature_2.view(Bp, Cp, -1) #N,C,HW: 8, 304, 1280
    
    # dot product similarity
    logit1comwith2 = torch.bmm(feature_1.transpose(1, 2), feature_2) #N,HW,HW: 8, 1280, 1280

    # get the mask for same class and different class from label_1 and label_2
    label_1 = label_1.permute(0,2,3,1) 
    label_1 = label_1.reshape(Bl, Hl*Wl,-1).float().cuda()#B,HW,C
    label_2 = label_2.permute(0,2,3,1) 
    label_2 = label_2.reshape(Bl, Hl*Wl,-1).float().cuda()#B,HW,C

    mask1comwith2_p = torch.bmm(label_1, label_2.transpose(1, 2)) #B,HWHW, B,i,j = 1 indicates pixel i in pred1 is the same class with pixel j in pred2
    mask1comwith2_n = 1 - mask1comwith2_p

    # get the similarity masks
    masked1comwith2_dist_p = mask1comwith2_p * logit1comwith2 # B,HW,HW
    masked1comwith2_dist_n = mask1comwith2_n * logit1comwith2
    
    # compute the top p_c or n_c number of features
    nonzeronum_p = torch.count_nonzero(masked1comwith2_dist_p,dim=2) # B,HW 8, 5120
    nonzeronum_n = torch.count_nonzero(masked1comwith2_dist_n,dim=2) # B,HW 8, 5120
    
    dist1comwith2_p = torch.sum(masked1comwith2_dist_p,dim=2) / nonzeronum_p # B,HW 4, 1280
    dist1comwith2_p = torch.from_numpy(np.nan_to_num(dist1comwith2_p.cpu().numpy())).cuda()
    nonzeronum_p_count = torch.count_nonzero(dist1comwith2_p,dim=1)
    dist1comwith2_p_average = torch.sum(dist1comwith2_p,dim=1) / nonzeronum_p_count
    
    dist1comwith2_n = torch.sum(masked1comwith2_dist_n,dim=2) / nonzeronum_n
    dist1comwith2_n = torch.from_numpy(np.nan_to_num(dist1comwith2_n.cpu().numpy())).cuda()
    nonzeronum_n_count = torch.count_nonzero(dist1comwith2_n,dim=1)
    dist1comwith2_n_average = torch.sum(dist1comwith2_n,dim=1) / nonzeronum_n_count
    
    return dist1comwith2_p_average, dist1comwith2_n_average

def TFAL_select_Mask(feature_1, feature_2, label_1, label_2,class_num ,p_thershold = 0.5, n_thershold = 0.5, select = 'p', H = 512, W = 640):

    label_1 = one_hot(label_1.to(torch.int64), num_classes=class_num)
    label_2 = one_hot(label_2.to(torch.int64), num_classes=class_num)

    label_1 = label_1.permute(0,3,1,2) 
    label_2 = label_2.permute(0,3,1,2) 
    
    Bl, Cl, Hl, Wl = label_1.shape #4, 12, 32, 40
    Bp, Cp, Hp, Wp = feature_1.shape #4, 304, 32, 40

    p_number = np.int(0.3 * H * W)
    n_number = np.int(0.3 * H * W)

    # initialize output mask
    mask_p_dle = torch.zeros(Bl,H * W).float().cuda()
    mask_n_dle = torch.zeros(Bl,H * W).float().cuda()
    mask_ones = torch.ones(Bl,H * W).float().cuda()

    # compute the simlarities between feature feature_1 and feature_2
    feature_1 = feature_1.view(Bp, Cp, -1) #N,C,HW: 8, 304, 1280
    feature_2 = feature_2.view(Bp, Cp, -1) #N,C,HW: 8, 304, 1280
    # dot product similarity
    logit1comwith2 = torch.bmm(feature_1.transpose(1, 2), feature_2) #N,HW,HW: 8, 1280, 1280 

    # get the mask for same class and different class from label_1 and label_2
    label_1 = label_1.permute(0,2,3,1) 
    label_1 = label_1.reshape(Bl, Hl*Wl,-1).float().cuda()#B,HW,C
    label_2 = label_2.permute(0,2,3,1) 
    label_2 = label_2.reshape(Bl, Hl*Wl,-1).float().cuda()#B,HW,C

    mask1comwith2_p = torch.bmm(label_1, label_2.transpose(1, 2)) #B,HW,HW, B,i,j = 1 indicates pixel i in pred1 is the same class with pixel j in pred2
    mask1comwith2_n = 1 - mask1comwith2_p

    # get the similarity masks
    masked1comwith2_dist_p = mask1comwith2_p * logit1comwith2 # B,HW,HW
    masked1comwith2_dist_n = mask1comwith2_n * logit1comwith2

    nonzeronum_p = torch.count_nonzero(masked1comwith2_dist_p,dim=2) # B,HW 8, 1280
    nonzeronum_n = torch.count_nonzero(masked1comwith2_dist_n,dim=2) # B,HW 8, 1280
    
    dist1comwith2_p = torch.sum(masked1comwith2_dist_p,dim=2) / nonzeronum_p # B,HW 4, 1280 need to find index of pixels smaller than p_thershold
    dist1comwith2_p = torch.from_numpy(np.nan_to_num(dist1comwith2_p.detach().cpu().numpy())).cuda()
    nonzeronum_p_count = torch.count_nonzero(dist1comwith2_p,dim=1)
    dist1comwith2_p_average = torch.sum(dist1comwith2_p,dim=1) / nonzeronum_p_count

    dist1comwith2_n = torch.sum(masked1comwith2_dist_n,dim=2) / nonzeronum_n # B,HW 4, 1280 need to find index of pixels larger than n_thershold
    dist1comwith2_n = torch.from_numpy(np.nan_to_num(dist1comwith2_n.detach().cpu().numpy())).cuda()
    nonzeronum_n_count = torch.count_nonzero(dist1comwith2_n,dim=1)
    dist1comwith2_n_average = torch.sum(dist1comwith2_n,dim=1) / nonzeronum_n_count

    reliability_average = (dist1comwith2_p_average + 1 - dist1comwith2_n_average) / 2

    dist1comwith2_p = dist1comwith2_p.view(Bp , Hp , Wp)
    dist1comwith2_p = F.interpolate(dist1comwith2_p.unsqueeze(0), size=[H,W], mode='bilinear').squeeze(0)
    dist1comwith2_p = dist1comwith2_p.view(Bl,H * W)
    dist1comwith2_n = dist1comwith2_n.view(Bp , Hp , Wp)
    dist1comwith2_n = F.interpolate(dist1comwith2_n.unsqueeze(0), size=[H,W], mode='bilinear').squeeze(0)
    dist1comwith2_n = dist1comwith2_n.view(Bl,H * W)

    confidence_map = (dist1comwith2_p + 1 - dist1comwith2_n).view(Bl,H * W)
    if select == 'union':
        for i in range(Bl):
            # --------------------- use threshold to determine the selected noise map ---------------------#
            # index_p = np.argwhere(dist1comwith2_p[i].cpu().numpy() < p_thershold)
            # index_p = torch.from_numpy(index_p).transpose(0, 1)
            # index_n = np.argwhere(dist1comwith2_n[i].cpu().numpy() > 1 - n_thershold)
            # index_n = torch.from_numpy(index_n).transpose(0, 1)

            # --------------------- use percentage to determine the selected noise map ---------------------#
            _,index_p = torch.topk(dist1comwith2_p[i],p_number,largest=False)
            _,index_n = torch.topk(dist1comwith2_n[i],n_number,largest=True)

            mask_p_dle[i,index_p] = mask_ones[i,index_p]
            mask_p_dle[i,index_n] = mask_ones[i,index_n]

        mask_final = mask_p_dle
        mask_final = mask_final.view(Bl,H,W) #8, 256, 320
    else:        
        for i in range(Bl):
            # --------------------- use threshold to determine the selected noise map ---------------------#
            # index_p = np.argwhere(dist1comwith2_p[i].cpu().numpy() < p_thershold)
            # index_p = torch.from_numpy(index_p).transpose(0, 1)
            # index_n = np.argwhere(dist1comwith2_n[i].cpu().numpy() > 1 - n_thershold)
            # index_n = torch.from_numpy(index_n).transpose(0, 1)

            # --------------------- use percentage to determine the selected noise map ---------------------#
            _,index_p = torch.topk(dist1comwith2_p[i],p_number,largest=False)
            _,index_n = torch.topk(dist1comwith2_n[i],n_number,largest=True)

            mask_p_dle[i,index_p] = mask_ones[i,index_p]
            mask_n_dle[i,index_n] = mask_ones[i,index_n]

        mask_p_dle = mask_p_dle.view(Bl,H,W) #8, 256, 320
        mask_n_dle = mask_n_dle.view(Bl,H,W) #8, 256, 320

        if select == 'p':
            mask_final = mask_p_dle
        elif select == 'n':
            mask_final = mask_n_dle 
        elif select == 'intersection':
            mask_final = mask_p_dle * mask_n_dle
    
    return mask_final,dist1comwith2_p_average, dist1comwith2_n_average,confidence_map

def TFAL_select_Mask_test(feature_1, feature_2, label_1, label_2,class_num ,p_thershold = 0.5, n_thershold = 0.5, select = 'p', H = 256, W = 320):

    label_1 = one_hot(label_1.to(torch.int64), num_classes=class_num)
    label_2 = one_hot(label_2.to(torch.int64), num_classes=class_num)

    label_1 = label_1.permute(0,3,1,2) 
    label_2 = label_2.permute(0,3,1,2) 
    
    Bl, Cl, Hl, Wl = label_1.shape #4, 12, 32, 40
    Bp, Cp, Hp, Wp = feature_1.shape #4, 304, 32, 40

    p_number = np.int(0.4 * H * W)
    n_number = np.int(0.4 * H * W)

    # initialize output mask
    mask_p_dle = torch.zeros(Bl,H * W).float().cuda()
    mask_n_dle = torch.zeros(Bl,H * W).float().cuda()
    mask_ones = torch.ones(Bl,H * W).float().cuda()

    # compute the simlarities between feature feature_1 and feature_2
    feature_1 = feature_1.view(Bp, Cp, -1) #N,C,HW: 4, 304, 1280
    feature_2 = feature_2.view(Bp, Cp, -1) #N,C,HW: 4, 304, 1280

    # dot product similarity
    logit1comwith2 = torch.bmm(feature_1.transpose(1, 2), feature_2) #N,HW,HW: 4, 1280, 1280  

    # get the mask for same class and different class from label_1 and label_2
    label_1 = label_1.permute(0,2,3,1) 
    label_1 = label_1.reshape(Bl, Hl*Wl,-1).float().cuda()#B,HW,C
    label_2 = label_2.permute(0,2,3,1) 
    label_2 = label_2.reshape(Bl, Hl*Wl,-1).float().cuda()#B,HW,C

    mask1comwith2_p = torch.bmm(label_1, label_2.transpose(1, 2)) #B,HW,HW, B,i,j = 1 indicates pixel i in pred1 is the same class with pixel j in pred2
    mask1comwith2_n = 1 - mask1comwith2_p
    # print('mask1comwith2_p:', mask1comwith2_p.shape)

    # get the similarity masks
    masked1comwith2_dist_p = mask1comwith2_p * logit1comwith2 # B,HW,HW
    masked1comwith2_dist_n = mask1comwith2_n * logit1comwith2
    
    nonzeronum_p = torch.count_nonzero(masked1comwith2_dist_p,dim=2) # B,HW 8, 5120
    nonzeronum_n = torch.count_nonzero(masked1comwith2_dist_n,dim=2) # B,HW 8, 5120
    
    dist1comwith2_p = torch.sum(masked1comwith2_dist_p,dim=2) / nonzeronum_p # B,HW 4, 1280 need to find index of pixels smaller than p_thershold
    dist1comwith2_p = torch.from_numpy(np.nan_to_num(dist1comwith2_p.detach().cpu().numpy())).cuda()
    nonzeronum_p_count = torch.count_nonzero(dist1comwith2_p,dim=1)
    dist1comwith2_p_average = torch.sum(dist1comwith2_p,dim=1) / nonzeronum_p_count

    dist1comwith2_n = torch.sum(masked1comwith2_dist_n,dim=2) / nonzeronum_n # B,HW 4, 1280 need to find index of pixels larger than n_thershold
    dist1comwith2_n = torch.from_numpy(np.nan_to_num(dist1comwith2_n.detach().cpu().numpy())).cuda()
    nonzeronum_n_count = torch.count_nonzero(dist1comwith2_n,dim=1)
    dist1comwith2_n_average = torch.sum(dist1comwith2_n,dim=1) / nonzeronum_n_count

    reliability_average = (dist1comwith2_p_average + 1 - dist1comwith2_n_average) / 2

    dist1comwith2_p = dist1comwith2_p.view(Bp , Hp , Wp)
    dist1comwith2_p = F.interpolate(dist1comwith2_p.unsqueeze(0), size=[H,W], mode='bilinear').squeeze(0)
    dist1comwith2_p = dist1comwith2_p.view(Bl,H * W)
    dist1comwith2_n = dist1comwith2_n.view(Bp , Hp , Wp)
    dist1comwith2_n = F.interpolate(dist1comwith2_n.unsqueeze(0), size=[H,W], mode='bilinear').squeeze(0)
    dist1comwith2_n = dist1comwith2_n.view(Bl,H * W)

    confidence_map = (dist1comwith2_p + 1 - dist1comwith2_n).view(Bl,H * W)
    if select == 'union':
        for i in range(Bl):

            # --------------------- use threshold to determine the selected noise map ---------------------#
            # index_p = np.argwhere(dist1comwith2_p[i].cpu().numpy() < p_thershold)
            # index_p = torch.from_numpy(index_p).transpose(0, 1)
            # index_n = np.argwhere(dist1comwith2_n[i].cpu().numpy() > 1 - n_thershold)
            # index_n = torch.from_numpy(index_n).transpose(0, 1)

            # --------------------- use percentage to determine the selected noise map ---------------------#
            _,index_p = torch.topk(dist1comwith2_p[i],p_number,largest=False)
            _,index_n = torch.topk(dist1comwith2_n[i],n_number,largest=True)

            mask_p_dle[i,index_p] = mask_ones[i,index_p]
            mask_p_dle[i,index_n] = mask_ones[i,index_n]

        mask_final = mask_p_dle
        mask_final = mask_final.view(Bl,H,W) #8, 256, 320
    else:        
        for i in range(Bl):
            # --------------------- use threshold to determine the selected noise map ---------------------#
            # index_p = np.argwhere(dist1comwith2_p[i].cpu().numpy() < p_thershold)
            # index_p = torch.from_numpy(index_p).transpose(0, 1)
            # index_n = np.argwhere(dist1comwith2_n[i].cpu().numpy() > 1 - n_thershold)
            # index_n = torch.from_numpy(index_n).transpose(0, 1)

            # --------------------- use percentage to determine the selected noise map ---------------------#
            _,index_p = torch.topk(dist1comwith2_p[i],p_number,largest=False)
            _,index_n = torch.topk(dist1comwith2_n[i],n_number,largest=True)

            mask_p_dle[i,index_p] = mask_ones[i,index_p]
            mask_n_dle[i,index_n] = mask_ones[i,index_n]

        mask_p_dle = mask_p_dle.view(Bl,H,W) #8, 256, 320
        mask_n_dle = mask_n_dle.view(Bl,H,W) #8, 256, 320

        if select == 'p':
            mask_final = mask_p_dle
        elif select == 'n':
            mask_final = mask_n_dle 
        elif select == 'intersection':
            mask_final = mask_p_dle * mask_n_dle
    
    return mask_final,dist1comwith2_p_average, dist1comwith2_n_average,confidence_map,mask1comwith2_p,dist1comwith2_p,dist1comwith2_n,logit1comwith2

if __name__ == '__main__':
    import torch
    from net.Ours.base18 import DeepLabV3Plus
    os.environ['CUDA_VISIBLE_DEVICES']= '0,3'
    net = DeepLabV3Plus(num_classes=12, layers=18).cuda()
    with torch.no_grad():
        y1,feature_1 = net(torch.randn(4, 3, 256, 320).cuda())
        y21,feature_2 = net(torch.randn(4, 3, 256, 320).cuda())
        print('output 1 shape:',y1.shape)
        print('pred_1 shape:',feature_1.shape)

