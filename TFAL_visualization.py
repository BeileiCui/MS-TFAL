import os,tqdm,sys,time,argparse,tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))
import numpy as np
import torch.cuda.amp as amp
scaler = amp.GradScaler()

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot 
import torch.utils.data
import torch.distributed as dist
from net.Ours.TFAL_Module import TFAL_get_affinity,TFAL_select_Mask_test
from utils.summary import DisablePrint
from utils.LoadModel import load_model_full_fortest
from skimage import io
from sklearn.preprocessing import MinMaxScaler

##------------------------------ Training settings ------------------------------##
parser = argparse.ArgumentParser(description='real-time segmentation')

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist', action='store_true')

parser.add_argument('--root_dir', type=str, default='./results/endo18')
parser.add_argument('--dataset', type=str, default='endovis2018')
parser.add_argument('--data_tag', type=str, default='type')
parser.add_argument('--log_name', type=str, default='Uncertainty_test')
parser.add_argument('--data_type', type=str, choices=['clean','noisy'], default='noisy')
parser.add_argument('--data_ver', type=int, default=4 )
parser.add_argument('--arch', type=str, choices=['puredeeplab18','swinPlus'], default='puredeeplab18')
parser.add_argument('--pre_log_name', type=str, default='DLV3PLUS_clean_ver_0')
parser.add_argument('--pre_checkpoint', type=str, default=None) #!!

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--loss', type=str, default='ohem')

parser.add_argument('--gpus', type=str, default='2')
parser.add_argument('--downsample', type=int, default=1)
parser.add_argument('--h', type=int, default=256)
parser.add_argument('--w', type=int, default=320)

parser.add_argument('--log_interval', type=int, default=50)
parser.add_argument('--val_interval', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=3)

parser.add_argument('--t', type=int, default=1)
parser.add_argument('--step', type=int, default=1)

parser.add_argument('--ver', type=int, default=0)
parser.add_argument('--tag', type=int, default=1)

parser.add_argument('--global_n', type=int, default=0)

parser.add_argument('--pretrain_ep', type=int, default=None)
parser.add_argument('--decay', type=int, default=2)

parser.add_argument('--reset', type=str, default=None)
parser.add_argument('--reset_ep', type=int)

cfg = parser.parse_args()

color_map = {
    0: [0,0,0], # background-tissue
    1: [0,255,0], # instrument-shaft
    2: [0,255,255], # instrument-clasper
    3: [125,255,12], # instrument-wrist
    4: [255,55,0], # kidney-parenchyma，
    5: [24,55,125], # covered-kidney，
    6: [187,155,25], # thread，
    7: [0,255,125], # clamps，
    8: [255,255,125], # suturing-needle
    9: [123,15,175], # suction-instrument，
    10: [124,155,5], # small-intestine
    11: [12,255,141] # ultrasound-probe,
}

def label2rgb(ind_im, color_map=color_map):
	rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

	for i, rgb in color_map.items():
		rgb_im[(ind_im==i)] = rgb

	return rgb_im

def main():

    ################################################ def part ################################################

    ##------------------------------ compute feature based affinity confidence ------------------------------##   
    def affinity_confidence():
        print('\n computing affinity confidence test...')
        model.eval()
        Procedures = np.array([1,2,3,4,5,6,7,9,10,11,12,13,14,15,16])
        weight = np.array([0.4,0.4,0.4,0.4,0.4,0.4,0.5,0.6,0.7,0.8,1,1,1,1,1])
        p_sum_for_each_vedio = np.zeros((15,))
        n_sum_for_each_vedio = np.zeros((15,))
        count = np.zeros((15,))
        weight_final = np.zeros((15,))

        tic = time.perf_counter()
        for batch_idx, batch in tqdm.tqdm(enumerate(train_loader)):
            # if batch_idx < 6:
            #     continue
            for k in batch:
                if not k=='path':
                    batch[k] = batch[k].to(device=cfg.device).float()
            #print('shape:', batch['image'].shape) #4, 3, 272, 480
            with torch.no_grad():
                #print(batch['image'].shape)
                outputs , feature = model(batch['image'])
                outputs_1 , feature_1 = model(batch['image_1'])
                B, C, H, W = feature_1.shape
                label_ds = F.interpolate(batch['label'].unsqueeze(0), size=[H,W], mode='nearest').squeeze(0)
                label_1_ds = F.interpolate(batch['label_1'].unsqueeze(0), size=[H,W], mode='nearest').squeeze(0)

                _,p,n,_,_,_,_,_ = TFAL_select_Mask_test(feature, feature_1, label_ds, label_1_ds, class_num =classes , p_thershold = 0.5, n_thershold = 0.5, select = 'intersection', H = cfg.h, W = cfg.w)

                if batch['path'][0] < 9:
                    ins = batch['path'][0].numpy() - 1
                else:
                    ins = batch['path'][0].numpy() - 2

                p_sum_for_each_vedio[ins] += p.cpu().numpy()  
                n_sum_for_each_vedio[ins] += n.cpu().numpy()
                count[ins] += 1

        print('Frame number for each video:')
        print(count)

        AC_pn = (p_sum_for_each_vedio + count - n_sum_for_each_vedio) / count
        sort_p = np.argsort(p_sum_for_each_vedio)
        sort_n = np.argsort(count-n_sum_for_each_vedio)
        sort_pn = np.argsort(AC_pn)

        for i in range(len(weight_final)):
            weight_final[sort_pn[i]] = weight[i]

        print('Positive affinity for each video:')
        print(p_sum_for_each_vedio / count)
        print('Negative affinity for each video:')
        print((count-n_sum_for_each_vedio) / count)
        print('Affinity confidence for each video:')
        print(AC_pn)

        p_thershold = np.mean(p_sum_for_each_vedio / count)
        n_thershold = np.mean((count-n_sum_for_each_vedio) / count)
        print('p_thershold:',p_thershold)
        print('n_thershold:',n_thershold)

        print('Sort according to positive affinity from small to large:',Procedures[sort_p])
        print('Sort according to negative affinity from small to large:',Procedures[sort_n])
        print('Sort according to affinity confidence from small to large:',Procedures[sort_pn])
        print('weight for each video:',weight_final)

        print(' compute uncertainty finished.')
        return      
    
    ##------------------------------ generate samples figures related to temporal affinity ------------------------------##  
    def feature_based_affinity_confidence_test():
        print('\n computing sample affinity confidence test...')
        model.eval()
        Procedures = np.array([1,2,3,4,5,6,7,9,10,11,12,13,14,15,16])
        weight = np.array([0.2,0.2,0.2,0.2,0.2,0.4,0.5,0.6,0.7,0.8,1,1,1,1,1])
        p_sum_for_each_vedio = np.zeros((15,))
        n_sum_for_each_vedio = np.zeros((15,))
        count = np.zeros((15,))
        weight_final = np.zeros((15,))
        label_diff_output = []

        p_thershold = 0.48319209465377816
        n_thershold = 0.78678705171334

        for batch_idx, batch in tqdm.tqdm(enumerate(train_loader)):

            if batch_idx < 0:
                continue

            for k in batch:
                if not k=='path':
#                     batch[k] = batch[k].to(device=cfg.device, nonw_blocking=True).float()
                    batch[k] = batch[k].to(device=cfg.device).float()

            ## get the difference between noisy label and ground truth label ##
            a,b,c = batch['label'].shape
            label_clean = torch.zeros(a,b,c).to(device=cfg.device).float()
            label_diff_output = np.zeros((a,b,c))
            print(label_clean.shape)
            print('batch %d testing' % batch_idx)
            for i in range(cfg.batch_size):
                print(train_clean_dataset[i+cfg.batch_size * batch_idx]['path'])
                label_clean[i] = train_clean_dataset[i+cfg.batch_size * batch_idx]['label']
            
            # get the noise variance map
            for i in range(cfg.batch_size):
                label_diff = one_hot(label_clean[i].to(torch.int64), num_classes=12)* one_hot(batch['label'][i].to(torch.int64), num_classes=12)

                label_diff = 1 - torch.sum(label_diff,dim=2)
                label_diff_output[i] = label_diff.cpu().numpy().astype(np.uint8)
            ## get the difference between noisy label and ground truth label ##

            outputs , feature = model(batch['image'])
            outputs_1 , feature_1 = model(batch['image_1'])

            ins,frame = batch['path']
            B, C, H, W = feature_1.shape
            for i in range(B):
                print('the image is seq_%d frame%03d' %(ins[i],frame[i]))

            output = F.softmax(outputs,dim=1)
            output_output = torch.argmax(output,dim=1).cpu().numpy().astype(np.uint8)
            label_ds = F.interpolate(batch['label'].unsqueeze(0), size=[H,W], mode='nearest').squeeze(0)
            label_1_ds = F.interpolate(batch['label_1'].unsqueeze(0), size=[H,W], mode='nearest').squeeze(0)

            pos_pix_p,p,n,confidence_map,mask1comwith2_p,dist1comwith2_p,dist1comwith2_n,logit1comwith2 = TFAL_select_Mask_test(feature, feature_1, label_ds, label_1_ds, class_num = 12 ,p_thershold = p_thershold, n_thershold = n_thershold, select = 'p',H=h,W=w)
            pos_pix_n,_,_,_,_,_,_,_= TFAL_select_Mask_test(feature, feature_1, label_ds, label_1_ds, class_num = 12 ,p_thershold = p_thershold, n_thershold = n_thershold, select = 'n',H=h,W=w)
            pos_pix_i,_,_,_,_,_,_,_ = TFAL_select_Mask_test(feature, feature_1, label_ds, label_1_ds, class_num = 12 ,p_thershold = p_thershold, n_thershold = n_thershold, select = 'intersection',H=h,W=w)
            pos_pix_u,_,_,_,_,_,_,_ = TFAL_select_Mask_test(feature, feature_1, label_ds, label_1_ds, class_num = 12 ,p_thershold = p_thershold, n_thershold = n_thershold, select = 'union',H=h,W=w)

            mask1comwith2_n = 1 - mask1comwith2_p
            pos_pix_p_output = pos_pix_p.cpu().numpy().astype(np.uint8)
            pos_pix_n_output = pos_pix_n.cpu().numpy().astype(np.uint8)
            pos_pix_i_output = pos_pix_i.cpu().numpy().astype(np.uint8)
            pos_pix_u_output = pos_pix_u.cpu().numpy().astype(np.uint8) # 1, 256, 320
            pos_pix_i_n_output = 1 - pos_pix_i_output

            mask1comwith2_p_output = mask1comwith2_p.cpu().numpy().astype(np.uint8)
            mask1comwith2_n_output = mask1comwith2_n.cpu().numpy().astype(np.uint8)
            min_max_scaler = MinMaxScaler()
            confidence_map = confidence_map.cpu().numpy()
            dist1comwith2_p = dist1comwith2_p.cpu().numpy()
            dist1comwith2_n = dist1comwith2_n.cpu().numpy()
            logit1comwith2 = logit1comwith2.cpu().detach().numpy()

            for i in range(B):
                confidence_map[i] = min_max_scaler.fit_transform(confidence_map[i].reshape(-1, 1)).squeeze(1)
                dist1comwith2_p[i] = min_max_scaler.fit_transform(dist1comwith2_p[i].reshape(-1, 1)).squeeze(1)
                dist1comwith2_n[i] = min_max_scaler.fit_transform(dist1comwith2_n[i].reshape(-1, 1)).squeeze(1)
            confidence_map = confidence_map.reshape(B, h, w)
            dist1comwith2_p = dist1comwith2_p.reshape(B, h, w)
            dist1comwith2_n = dist1comwith2_n.reshape(B, h, w)

            label_gt_output = batch['label'].cpu().numpy().astype(np.uint8) # 1, 256, 320
            label_corrected = pos_pix_i_n_output * batch['label'].cpu().detach().numpy() + pos_pix_i_output * output_output
            image_output = batch['image'].permute(0,2,3,1).cpu().numpy() # 256, 320, 3

            # --------------------------------- save the images ---------------------------------- $
            cfg.pix_p_vis_path = os.path.join(cfg.test_dir,'pos_pix_p_seq_{}_frame{:03d}.png')
            cfg.pix_n_vis_path = os.path.join(cfg.test_dir,'pos_pix_n_seq_{}_frame{:03d}.png')
            cfg.pix_i_vis_path = os.path.join(cfg.test_dir,'pos_pix_i_seq_{}_frame{:03d}.png')
            cfg.pix_u_vis_path = os.path.join(cfg.test_dir,'pos_pix_u_seq_{}_frame{:03d}.png')
            cfg.labelgt_vis_path = os.path.join(cfg.test_dir,'labelnoisy_seq_{}_frame{:03d}.png')
            cfg.image_vis_path = os.path.join(cfg.test_dir,'image_seq_{}_frame{:03d}.png')
            cfg.labeldiff_vis_path = os.path.join(cfg.test_dir,'labeldiff_seq_{}_frame{:03d}.png')
            cfg.modelpred_vis_path = os.path.join(cfg.test_dir,'modelpred_seq_{}_frame{:03d}.png')
            cfg.labelcorrected_vis_path = os.path.join(cfg.test_dir,'labelcorrected_seq_{}_frame{:03d}.png')
            cfg.confidence_map_vis_path = os.path.join(cfg.test_dir,'affinity_confidence_map_seq_{}_frame{:03d}.png')
            cfg.p_map_vis_path = os.path.join(cfg.test_dir,'p_map_seq_{}_frame{:03d}.png')
            cfg.n_map_vis_path = os.path.join(cfg.test_dir,'n_map_seq_{}_frame{:03d}.png')
            cfg.p_affinity_map_vis_path = os.path.join(cfg.test_dir,'p_affinity_map_seq_{}_frame{:03d}.png')
            cfg.n_affinity_map_vis_path = os.path.join(cfg.test_dir,'n_affinity_map_seq_{}_frame{:03d}.png')
            cfg.cos_sim_map_vis_path = os.path.join(cfg.test_dir,'cos_sim__map_seq_{}_frame{:03d}.png')

            for i in range(B):
                save_pix_p_pth = cfg.pix_p_vis_path.format(ins[i],frame[i])
                save_pix_n_pth = cfg.pix_n_vis_path.format(ins[i],frame[i])
                save_pix_i_pth = cfg.pix_i_vis_path.format(ins[i],frame[i])
                save_pix_u_pth = cfg.pix_u_vis_path.format(ins[i],frame[i])
                save_labelgt_pth = cfg.labelgt_vis_path.format(ins[i],frame[i])
                save_image_pth = cfg.image_vis_path.format(ins[i],frame[i])
                save_labeldiff_pth = cfg.labeldiff_vis_path.format(ins[i],frame[i])
                save_modelpred_pth = cfg.modelpred_vis_path.format(ins[i],frame[i])
                save_labelcorrected_pth = cfg.labelcorrected_vis_path.format(ins[i],frame[i])
                save_confidence_map_path = cfg.confidence_map_vis_path.format(ins[i],frame[i])
                save_p_map_path = cfg.p_map_vis_path.format(ins[i],frame[i])
                save_n_map_path = cfg.n_map_vis_path.format(ins[i],frame[i])
                save_p_affinity_map_path = cfg.p_affinity_map_vis_path.format(ins[i],frame[i])
                save_n_affinity_map_path = cfg.n_affinity_map_vis_path.format(ins[i],frame[i])
                save_cos_sim_map_path = cfg.cos_sim_map_vis_path.format(ins[i],frame[i])

                predict = label2rgb(label_gt_output[i]).astype(np.uint8)
                predict_model = label2rgb(output_output[i]).astype(np.uint8)
                predict_corrected = label2rgb(label_corrected[i]).astype(np.uint8)

                io.imsave(save_pix_p_pth, pos_pix_p_output[i] * 255)
                io.imsave(save_pix_n_pth, pos_pix_n_output[i] * 255)
                io.imsave(save_pix_i_pth, pos_pix_i_output[i] * 255)
                io.imsave(save_pix_u_pth, pos_pix_u_output[i] * 255)
                io.imsave(save_labelgt_pth, predict)
                io.imsave(save_image_pth, image_output[i])
                io.imsave(save_labeldiff_pth, label_diff_output[i] * 255)
                io.imsave(save_modelpred_pth, predict_model)
                io.imsave(save_labelcorrected_pth, predict_corrected)
                io.imsave(save_confidence_map_path, confidence_map[i])
                io.imsave(save_p_map_path, mask1comwith2_p_output[i] * 255)
                io.imsave(save_n_map_path, mask1comwith2_n_output[i] * 255)
                io.imsave(save_p_affinity_map_path, dist1comwith2_p[i])
                io.imsave(save_n_affinity_map_path, dist1comwith2_n[i])
                io.imsave(save_cos_sim_map_path, logit1comwith2[i] * 255)
            print('feature based uncertainty test finished.')

            if batch_idx >= 0:
                print('testing break.')
                break
        return  

    ################################################ def part ################################################   
    
    ################################################ main part ################################################
    ##------------------------------ Enviroment ------------------------------##

    os.environ['CUDA_VISIBLE_DEVICES']=cfg.gpus
    torch.backends.cudnn.benchmark = True  # disable this if OOM at beginning of training
    num_gpus = torch.cuda.device_count()
    
    if cfg.dist:
        cfg.device = torch.device('cuda:%d' % cfg.local_rank)
        torch.cuda.set_device(cfg.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=num_gpus, rank=cfg.local_rank)
    else:
        cfg.device = torch.device('cuda')

    cfg.log_name += '_ver_' + str(cfg.ver)

        # logger
    cfg.log_dir = os.path.join(cfg.root_dir, cfg.log_name, 'logs')
    cfg.ckpt_dir = os.path.join(cfg.root_dir, cfg.log_name, 'ckpt')
    cfg.test_dir = os.path.join(cfg.root_dir, cfg.log_name, 'sample_test')
    os.makedirs(cfg.test_dir, exist_ok=True)

    print(cfg)
    ##------------------------------ dataset ------------------------------##
    print('Setting up data...')
    
    if cfg.dataset=='endovis2018':
        h,w = [cfg.h,cfg.w]
        ori_h, ori_w = [1024, 1280]
        print('size of data %d, %d.' %(h,w))
        if cfg.data_type=='clean':
            from dataset.Endovis2018_backbone import endovis2018
            train_dataset = endovis2018('train_clean', t=cfg.t, arch='swinPlus',rate=1, global_n=cfg.global_n,h = h, w = w)
            val_dataset = endovis2018('test_part', t=cfg.t,arch='swinPlus', rate=1, global_n=cfg.global_n,h = h, w = w)
            classes = train_dataset.class_num
        elif cfg.data_type=='noisy':
            from dataset.Endovis2018_backbone import endovis2018
            train_dataset = endovis2018('train', t=cfg.t, arch='swinPlus',rate=1, global_n=cfg.global_n, data_ver=cfg.data_ver,h = h, w = w)
            train_clean_dataset = endovis2018('train_clean', t=cfg.t, arch='swinPlus',rate=1, global_n=cfg.global_n, data_ver=cfg.data_ver,h = h, w = w)
            val_dataset = endovis2018('test_part', t=cfg.t,arch='swinPlus', rate=1, global_n=cfg.global_n, data_ver=cfg.data_ver,h = h, w = w)
            classes = train_dataset.class_num
    ##------------------------------ build model ------------------------------##        
    if 'puredeeplab' in cfg.arch:
        from net.Ours.base18 import DeepLabV3Plus
        model = DeepLabV3Plus(train_dataset.class_num, 18)
    elif 'swin' in cfg.arch:
        from net.Ours.base18 import TswinPlus
        model = TswinPlus(train_dataset.class_num,h,w)

    else:
        raise NotImplementedError
    # load pretrain model
    if cfg.pre_log_name is not None:
        cfg.pre_ckpt_path = os.path.join(cfg.root_dir, cfg.pre_log_name, 'ckpt', 'epoch_1_checkpoint.t7')
        print('initialize the model from:', cfg.pre_ckpt_path)
        model = load_model_full_fortest(model, cfg.pre_ckpt_path)
    ##------------------------------ combile model ------------------------------## 
    torch.cuda.empty_cache()
    print('Starting computing...')

    gpus = cfg.gpus.split(',')
    if len(cfg.gpus)>1:
        model = nn.DataParallel(model, device_ids=list(map(int,gpus))).cuda()
    else:
        model = model.to(cfg.device)
    ##------------------------------ dataloader ------------------------------## 
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                batch_size=cfg.batch_size,
                                shuffle= False,
                                num_workers=cfg.num_workers,
                                pin_memory=True,
                                drop_last=True)
    
    ##------------------------------ compute feature based affinity confidence ------------------------------##
    # enable this esction if you want to compute the affinity confidence for the dataset
    affinity_confidence()

    ##------------------------------ generate samples figures related to temporal affinity ------------------------------## 
    # enable this esction if you want to generate samples figures related to temporal affinity
    feature_based_affinity_confidence_test()

    ################################################ main part ################################################

if __name__ == '__main__':
    with DisablePrint(local_rank=cfg.local_rank):
        main()
