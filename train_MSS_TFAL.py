import os,tqdm,sys,time,argparse,math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

import numpy as np
import torch.cuda.amp as amp
scaler = amp.GradScaler()

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.distributed as dist
from net.Ours.TFAL_Module import TFAL_select_Mask,TFAL_get_affinity

from utils.losses import BCELoss,OhemCELoss2D,myOhemCELoss2D,DiceLoss

from utils.EndoMetric import general_dice, general_jaccard
from utils.summary import create_summary, create_logger, create_saver, DisablePrint
from utils.LoadModel import load_model_full,load_model


##------------------------------ Training settings ------------------------------##
parser = argparse.ArgumentParser(description='real-time segmentation')

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist', action='store_true')

parser.add_argument('--root_dir', type=str, default='./results/endo18')
parser.add_argument('--dataset', type=str, choices=['endovis2018','colon_oct'],default='endovis2018')
parser.add_argument('--data_tag', type=str, default='type')
parser.add_argument('--log_name', type=str, default='MSS_TFAL')
parser.add_argument('--data_type', type=str, choices=['clean','noisy'], default='noisy')
parser.add_argument('--data_ver', type=int, default=0)
parser.add_argument('--arch', type=str, default='puredeeplab18')
parser.add_argument('--pre_log_name', type=str, default=None)
parser.add_argument('--pre_checkpoint', type=str, default=None) #!!

parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--iterations', type=int, default=60000)
parser.add_argument('--loss', type=str, default='ohem')

parser.add_argument('--gpus', type=str, default='3')
parser.add_argument('--downsample', type=int, default=1)
parser.add_argument('--h', type=int, default=256)
parser.add_argument('--w', type=int, default=320)

parser.add_argument('--log_interval', type=int, default=50)
parser.add_argument('--val_interval', type=int, default=500)
parser.add_argument('--num_workers', type=int, default=3)

parser.add_argument('--t', type=int, default=1)
parser.add_argument('--step', type=int, default=1)

parser.add_argument('--ver', type=int, default=1)
parser.add_argument('--tag', type=int, default=1)
parser.add_argument('--T1', type=int, default=3)
parser.add_argument('--T2', type=int, default=6)
parser.add_argument('--T3', type=int, default=10)
parser.add_argument('--lambda_corrected', type=float, default=0.5)
parser.add_argument('--ratio_small_loss', type=float, default=0.4)
parser.add_argument('--small_loss_compute_interval', type=int, default=5)
parser.add_argument('--global_n', type=int, default=0)

parser.add_argument('--pretrain_ep', type=int, default=20)
parser.add_argument('--decay', type=int, default=2)

parser.add_argument('--reset', type=str, default=None)
parser.add_argument('--reset_ep', type=int)

cfg = parser.parse_args()
##------------------------------ Training settings ------------------------------##

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

    ##------------------------------ train model for endovis 18 dataset------------------------------##   
    def train_endo(epoch,stage,val_interval,best,best_ep,weight_for_vedio,p_thershold,n_thershold,video_uncertain):
        epoch_real = (epoch - 1) * 4
        ##------------------------- stage 1 -------------------------##
        if stage == 1:
            model.train()
            tic = time.perf_counter()
            tr_loss = []
            for batch_idx, batch in enumerate(train_loader):
                for k in batch:
                    if not k=='path':
                        batch[k] = batch[k].to(device=cfg.device).float()
                #print('shape:', batch['image'].shape) #4, 3, 272, 480
                with amp.autocast():
                    #print(batch['image'].shape)
                    outputs , _ = model(batch['image'].squeeze())
                    loss = compute_loss(outputs, batch['label'].squeeze().long())

                tr_loss.append(loss.detach().cpu().numpy())
                    
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                #loss.backward()
                #optimizer.step()

                if batch_idx % cfg.log_interval == 0:
                    duration = time.perf_counter() - tic
                    tic = time.perf_counter()
                    print('[%d/%d-%d/%d]' % (epoch, cfg.num_epochs, batch_idx, len(train_loader))+
                        'loss:{:.4f} Time:{:.4f}'.format(loss.item(),duration))
                
                if val_interval > 0 and (batch_idx+1) % val_interval == 0:
                    epoch_real += 1
                    save_map = val_map_endo(epoch,epoch_real)
                    if save_map > best:
                        best = save_map
                        best_ep = epoch_real
                        print(saver.save(model.state_dict(), 'epoch_{}_checkpoint'.format(epoch_real)))

                    else:
                        if epoch - best_ep > 200:
                            break
                    print(saver.save(model.state_dict(), 'latestcheckpoint'))
                    summary_writer.add_scalar('Tr_loss', np.mean(tr_loss), epoch_real)
                    tr_loss = []
        ##------------------------- stage 2 Video-level supervision involved-------------------------##            
        elif stage == 2:
            weight_for_vedio = torch.from_numpy(weight_for_vedio).to(device=cfg.device)
            model.train()
            tic = time.perf_counter()
            tr_loss = []
            for batch_idx, batch in enumerate(train_loader):
                for k in batch:
                    if not k=='path':
                        batch[k] = batch[k].to(device=cfg.device).float()

                if batch['path'][0] < 9:
                    ins_weight = batch['path'][0].numpy() - 1
                else:
                    ins_weight = batch['path'][0].numpy() - 2
                b = weight_for_vedio[ins_weight]
                
                with amp.autocast():
                    #print(batch['image'].shape)
                    outputs , _ = model(batch['image'].squeeze())
                    loss = b * compute_loss(outputs, batch['label'].squeeze().long())

                tr_loss.append(loss.detach().cpu().numpy())
                    
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if batch_idx % cfg.log_interval == 0:
                    duration = time.perf_counter() - tic
                    tic = time.perf_counter()
                    print('[%d/%d-%d/%d]' % (epoch, cfg.num_epochs, batch_idx, len(train_loader))+
                        'loss:{:.4f} Time:{:.4f}'.format(loss.item(),duration))
                
                if val_interval > 0 and (batch_idx+1) % val_interval == 0:
                    epoch_real += 1
                    save_map = val_map_endo(epoch,epoch_real)
                    if save_map > best:
                        best = save_map
                        best_ep = epoch_real
                        print(saver.save(model.state_dict(), 'epoch_{}_checkpoint'.format(epoch_real)))
                    else:
                        if epoch - best_ep > 200:
                            break
                    print(saver.save(model.state_dict(), 'latestcheckpoint'))
                    summary_writer.add_scalar('Tr_loss', np.mean(tr_loss), epoch_real)
                    tr_loss = []
        ##------------------------- stage 3 Video and Image-level supervision involved-------------------------##
        elif stage == 3:
            weight_for_vedio = torch.from_numpy(weight_for_vedio).to(device=cfg.device)
            pn_thershold = p_thershold + n_thershold
            model.train()
            tic = time.perf_counter()
            tr_loss = []
            for batch_idx, batch in enumerate(train_loader):
                for k in batch:
                    if not k=='path':
                        batch[k] = batch[k].to(device=cfg.device).float()

                if batch['path'][0] < 9:
                    ins_weight = batch['path'][0].numpy() - 1
                else:
                    ins_weight = batch['path'][0].numpy() - 2
                b = weight_for_vedio[ins_weight]

                with amp.autocast():
                    outputs , feature = model(batch['image'].squeeze())

                with torch.no_grad():
                    _ , feature_1 = model(batch['image_1'].squeeze())
                    B, C, H, W = feature_1.shape
                    label_ds = F.interpolate(batch['label'], size=[H,W], mode='nearest').squeeze(0)
                    label_1_ds = F.interpolate(batch['label_1'], size=[H,W], mode='nearest').squeeze(0)
                    p,n = TFAL_get_affinity(feature, feature_1, label_ds, label_1_ds, class_num =classes , H = h, W = w)
                    a = 1 / (math.exp( -(torch.mean(p + 1 - n) - pn_thershold) ) ) ** 2

                with amp.autocast():    
                    loss = a * b * compute_loss(outputs, batch['label'].squeeze().long())

                tr_loss.append(loss.detach().cpu().numpy())
                    
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if batch_idx % cfg.log_interval == 0:
                    duration = time.perf_counter() - tic
                    tic = time.perf_counter()
                    print('[%d/%d-%d/%d]' % (epoch, cfg.num_epochs, batch_idx, len(train_loader))+
                        'loss:{:.4f} Time:{:.4f}'.format(loss.item(),duration))
                
                if val_interval > 0 and (batch_idx+1) % val_interval == 0:
                    epoch_real += 1
                    save_map = val_map_endo(epoch,epoch_real)

                    if save_map > best:
                        best = save_map
                        best_ep = epoch_real
                        print(saver.save(model.state_dict(), 'epoch_{}_checkpoint'.format(epoch_real)))
                    else:
                        if epoch - best_ep > 200:
                            break
                    print(saver.save(model.state_dict(), 'latestcheckpoint'))
                    summary_writer.add_scalar('Tr_loss', np.mean(tr_loss), epoch_real)
                    tr_loss = []
        ##------------------------- stage 4 Video,Image and Pixel-level supervision involved-------------------------##            
        elif stage == 4:
            weight_for_vedio = torch.from_numpy(weight_for_vedio).to(device=cfg.device)
            pn_thershold = p_thershold + n_thershold
            model.train()
            model_pred.eval()
            tic = time.perf_counter()
            tr_loss = []
            for batch_idx, batch in enumerate(train_loader):
                for k in batch:
                    if not k=='path':
                        batch[k] = batch[k].to(device=cfg.device).float()

                if batch['path'][0] < 9:
                    ins_weight = batch['path'][0].numpy() - 1
                else:
                    ins_weight = batch['path'][0].numpy() - 2
                b = weight_for_vedio[ins_weight]

                with amp.autocast():
                    outputs , feature = model(batch['image'].squeeze())

                with torch.no_grad():
                    _ , feature_1 = model(batch['image_1'].squeeze())
                    B, C, H, W = feature_1.shape
                    label_ds = F.interpolate(batch['label'], size=[H,W], mode='nearest').squeeze(0)
                    label_1_ds = F.interpolate(batch['label_1'], size=[H,W], mode='nearest').squeeze(0)

                    if batch['path'][0].numpy() in video_uncertain:
                        outputs_pred , _ = model_pred(batch['image'].squeeze())
                        output = F.softmax(outputs_pred,dim=1)
                        output = torch.argmax(output,dim=1)
                        pos_pix_i,p,n,_ = TFAL_select_Mask(feature, feature_1, label_ds, label_1_ds, class_num = classes ,p_thershold = p_thershold, n_thershold = n_thershold, select = 'intersection', H = h, W = w)
                        pos_pix_i_n = 1 - pos_pix_i
                        label = (pos_pix_i_n * batch['label'].squeeze().long() + pos_pix_i * output).long()
                        a = 1
                        b = 0.7
                    else:
                        p,n = TFAL_get_affinity(feature, feature_1, label_ds, label_1_ds, class_num =classes , H = h, W = w)
                        label = batch['label'].squeeze().long()
                        a = 1 / (math.exp( -(torch.mean(p + 1 - n) - pn_thershold) ) ) ** 2

                with amp.autocast():    
                    loss = a * b * compute_loss(outputs, label)

                tr_loss.append(loss.detach().cpu().numpy())
                    
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if batch_idx % cfg.log_interval == 0:
                    duration = time.perf_counter() - tic
                    tic = time.perf_counter()
                    print('[%d/%d-%d/%d]' % (epoch, cfg.num_epochs, batch_idx, len(train_loader))+
                        'loss:{:.4f} Time:{:.4f}'.format(loss.item(),duration))
                
                if val_interval > 0 and (batch_idx+1) % val_interval == 0:
                    epoch_real += 1
                    save_map = val_map_endo(epoch,epoch_real)

                    if save_map > best:
                        best = save_map
                        best_ep = epoch_real
                        print(saver.save(model.state_dict(), 'epoch_{}_checkpoint'.format(epoch_real)))
                    else:
                        if epoch - best_ep > 200:
                            break
                    print(saver.save(model.state_dict(), 'latestcheckpoint'))
                    summary_writer.add_scalar('Tr_loss', np.mean(tr_loss), epoch_real)
                    tr_loss = []

        return best,best_ep
    
    ##------------------------------ train model for rat colon dataset------------------------------##   
    def train_oct(epoch,stage,val_interval,best,best_ep,weight_for_vedio,p_thershold,n_thershold,video_uncertain):
        Procedures_mini = ['2T1','3C1','3T1','3T2','7C','10C','13C','15C']
        epoch_real = (epoch - 1) * 4
        ##------------------------- stage 1 -------------------------##
        if stage == 1:
            model.train()
            tic = time.perf_counter()
            tr_loss = []
            for batch_idx, batch in enumerate(train_loader):
                for k in batch:
                    if not k=='path':
                        batch[k] = batch[k].to(device=cfg.device).float()
                #print('shape:', batch['image'].shape) #4, 3, 272, 480
                with amp.autocast():
                    #print(batch['image'].shape)
                    outputs , _ = model(batch['image'].squeeze())
                    loss = compute_loss(outputs, batch['label'].squeeze().long())

                tr_loss.append(loss.detach().cpu().numpy())
                    
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                #loss.backward()
                #optimizer.step()

                if batch_idx % cfg.log_interval == 0:
                    duration = time.perf_counter() - tic
                    tic = time.perf_counter()
                    print('[%d/%d-%d/%d]' % (epoch, cfg.num_epochs, batch_idx, len(train_loader))+
                        'loss:{:.4f} Time:{:.4f}'.format(loss.item(),duration))

                if val_interval > 0 and (batch_idx+1) % val_interval == 0:
                    epoch_real += 1

                    save_map = val_map_oct(epoch,epoch_real)
                    if save_map > best:
                        best = save_map
                        best_ep = epoch_real
                        print(saver.save(model.state_dict(), 'epoch_{}_checkpoint'.format(epoch_real)))

                    else:
                        if epoch - best_ep > 200:
                            break
                    print(saver.save(model.state_dict(), 'latestcheckpoint'))
                    summary_writer.add_scalar('Tr_loss', np.mean(tr_loss), epoch_real)
                    tr_loss = []
        ##------------------------- stage 2 Video-level supervision involved-------------------------##       
        elif stage == 2:
            weight_for_vedio = torch.from_numpy(weight_for_vedio).to(device=cfg.device)
            model.train()
            tic = time.perf_counter()
            tr_loss = []
            for batch_idx, batch in enumerate(train_loader):
                for k in batch:
                    if not k=='path':
                        batch[k] = batch[k].to(device=cfg.device).float()

                ins = batch['path'][0][0]
                seq_ins = Procedures_mini.index(ins)
                b = weight_for_vedio[seq_ins]
                
                with amp.autocast():
                    # print(batch['image'].shape)
                    outputs , _ = model(batch['image'].squeeze())
                    loss = b * compute_loss(outputs, batch['label'].squeeze().long())

                tr_loss.append(loss.detach().cpu().numpy())
                    
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if batch_idx % cfg.log_interval == 0:
                    duration = time.perf_counter() - tic
                    tic = time.perf_counter()
                    print('[%d/%d-%d/%d]' % (epoch, cfg.num_epochs, batch_idx, len(train_loader))+
                        'loss:{:.4f} Time:{:.4f}'.format(loss.item(),duration))
                
                if val_interval > 0 and (batch_idx+1) % val_interval == 0:
                    epoch_real += 1
                    save_map = val_map_oct(epoch,epoch_real)
                    if save_map > best:
                        best = save_map
                        best_ep = epoch_real
                        print(saver.save(model.state_dict(), 'epoch_{}_checkpoint'.format(epoch_real)))
                    else:
                        if epoch - best_ep > 200:
                            break
                    print(saver.save(model.state_dict(), 'latestcheckpoint'))
                    summary_writer.add_scalar('Tr_loss', np.mean(tr_loss), epoch_real)
                    tr_loss = []

        ##------------------------- stage 3 Video and Image-level supervision involved-------------------------##    
        elif stage == 3:
            weight_for_vedio = torch.from_numpy(weight_for_vedio).to(device=cfg.device)
            pn_thershold = p_thershold + n_thershold
            model.train()
            tic = time.perf_counter()
            tr_loss = []
            for batch_idx, batch in enumerate(train_loader):
                for k in batch:
                    if not k=='path':
                        batch[k] = batch[k].to(device=cfg.device).float()

                ins = batch['path'][0][0]
                seq_ins = Procedures_mini.index(ins)
                b = weight_for_vedio[seq_ins]

                with amp.autocast():
                    outputs , feature = model(batch['image'].squeeze())

                with torch.no_grad():
                    _ , feature_1 = model(batch['image_1'].squeeze())
                    B, C, H, W = feature_1.shape
                    label_ds = F.interpolate(batch['label'], size=[H,W], mode='nearest').squeeze(0)
                    label_1_ds = F.interpolate(batch['label_1'], size=[H,W], mode='nearest').squeeze(0)
                    p,n = TFAL_get_affinity(feature, feature_1, label_ds, label_1_ds, class_num =classes , H = h, W = w)
                    a = 1 / (math.exp( -(torch.mean(p + 1 - n) - pn_thershold) ) ) ** 2

                with amp.autocast():    
                    loss = a * b * compute_loss(outputs, batch['label'].squeeze().long())

                tr_loss.append(loss.detach().cpu().numpy())
                    
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                #loss.backward()
                #optimizer.step()

                if batch_idx % cfg.log_interval == 0:
                    duration = time.perf_counter() - tic
                    tic = time.perf_counter()
                    print('[%d/%d-%d/%d]' % (epoch, cfg.num_epochs, batch_idx, len(train_loader))+
                        'loss:{:.4f} Time:{:.4f}'.format(loss.item(),duration))
                
                if val_interval > 0 and (batch_idx+1) % val_interval == 0:
                    epoch_real += 1
                    save_map = val_map_oct(epoch,epoch_real)
                    if save_map > best:
                        best = save_map
                        best_ep = epoch_real
                        print(saver.save(model.state_dict(), 'epoch_{}_checkpoint'.format(epoch_real)))
                    else:
                        if epoch - best_ep > 200:
                            break
                    print(saver.save(model.state_dict(), 'latestcheckpoint'))
                    summary_writer.add_scalar('Tr_loss', np.mean(tr_loss), epoch_real)
                    tr_loss = []

        ##------------------------- stage 4 Video ,Image and Pixel-level supervision involved-------------------------##              
        elif stage == 4:
            weight_for_vedio = torch.from_numpy(weight_for_vedio).to(device=cfg.device)
            pn_thershold = p_thershold + n_thershold
            model.train()
            model_pred.eval()
            tic = time.perf_counter()
            tr_loss = []
            for batch_idx, batch in enumerate(train_loader):
                for k in batch:
                    if not k=='path':
                        batch[k] = batch[k].to(device=cfg.device).float()

                ins = batch['path'][0][0]
                seq_ins = Procedures_mini.index(ins)
                b = weight_for_vedio[seq_ins]

                with amp.autocast():
                    outputs , feature = model(batch['image'].squeeze())

                with torch.no_grad():
                    _ , feature_1 = model(batch['image_1'].squeeze())
                    B, C, H, W = feature_1.shape
                    label_ds = F.interpolate(batch['label'], size=[H,W], mode='nearest').squeeze(0)
                    label_1_ds = F.interpolate(batch['label_1'], size=[H,W], mode='nearest').squeeze(0)

                    if seq_ins in video_uncertain:
                        outputs_pred , _ = model_pred(batch['image'].squeeze())
                        output = F.softmax(outputs_pred,dim=1)
                        output = torch.argmax(output,dim=1)
                        pos_pix_i,p,n,_ = TFAL_select_Mask(feature, feature_1, label_ds, label_1_ds, class_num = classes ,p_thershold = p_thershold, n_thershold = n_thershold, select = 'intersection', H = h, W = w)
                        pos_pix_i_n = 1 - pos_pix_i
                        label = (pos_pix_i_n * batch['label'].squeeze().long() + pos_pix_i * output).long()
                        a = 1
                        b = 0.7
                    else:
                        p,n = TFAL_get_affinity(feature, feature_1, label_ds, label_1_ds, class_num =classes , H = h, W = w)
                        label = batch['label'].squeeze().long()
                        a = 1 / (math.exp( -(torch.mean(p + 1 - n) - pn_thershold) ) ) ** 2

                with amp.autocast():    
                    loss = a * b * compute_loss(outputs, label)

                tr_loss.append(loss.detach().cpu().numpy())
                    
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if batch_idx % cfg.log_interval == 0:
                    duration = time.perf_counter() - tic
                    tic = time.perf_counter()
                    print('[%d/%d-%d/%d]' % (epoch, cfg.num_epochs, batch_idx, len(train_loader))+
                        'loss:{:.4f} Time:{:.4f}'.format(loss.item(),duration))
                
                if val_interval > 0 and (batch_idx+1) % val_interval == 0:
                    epoch_real += 1
                    save_map = val_map_oct(epoch,epoch_real)
                    if save_map > best:
                        best = save_map
                        best_ep = epoch_real
                        print(saver.save(model.state_dict(), 'epoch_{}_checkpoint'.format(epoch_real)))
                    else:
                        if epoch - best_ep > 200:
                            break
                    print(saver.save(model.state_dict(), 'latestcheckpoint'))
                    summary_writer.add_scalar('Tr_loss', np.mean(tr_loss), epoch_real)
                    tr_loss = []

        return best,best_ep

    ##------------------------------ validation model for endovis18 ------------------------------##   
    def val_map_endo(epoch,epoch_real):
        print('\n Val Endo18 @ Epoch: %d, Real Epoch: %d' % (epoch,epoch_real))
        model.eval()
        torch.cuda.empty_cache()
        metrics = np.zeros((2,))
        metrics_seq = np.zeros((2, 4))
        count_seq = np.zeros((4,))
        dice_each = np.zeros((12,))
        iou_each = np.zeros((12,))
        tool_each = np.zeros((12,))
        count = 0

        with torch.no_grad():
            for inputs in tqdm.tqdm(val_loader):
                inputs['image'] = inputs['image'].to(cfg.device).float()
                #print('shape:', inputs['image'].shape) #1,3,256,480
                tic = time.perf_counter()
                # print('imput size: ',inputs['image'].squeeze(0).shape)
                output,_ = model(inputs['image'].squeeze(0))

                output = F.interpolate(output, (ori_h,ori_w), mode='bilinear', align_corners=True)
                output = F.softmax(output,dim=1)
                output = torch.argmax(output,dim=1)
                output = output.cpu().numpy()
                duration = time.perf_counter() - tic

                dice = general_dice(inputs['label'].numpy(),output)  # dice containing each tool class
                iou = general_jaccard(inputs['label'].numpy(), output)
                for i in range(len(dice)):
                    tool_id = dice[i][0]
                    dice_each[tool_id] += dice[i][1]
                    iou_each[tool_id] += iou[i][1]
                    tool_each[tool_id] += 1
                frame_dice = np.mean([dice[i][1] for i in range(len(dice))])
                frame_iou = np.mean([iou[i][1] for i in range(len(dice))])
                #overall
                metrics[0] += frame_dice # dice of each frame
                metrics[1] += frame_iou
                count += 1
                #----for seq
                seq_ind = int(inputs['path'][0]) - 1 #seq: 0-3
                metrics_seq[0][seq_ind] += frame_dice
                metrics_seq[1][seq_ind] += frame_iou
                count_seq[seq_ind] += 1

            for j in range(12):
                dice_each[j] /= tool_each[j]
                iou_each[j] /= tool_each[j]
            dice_each_f = [float('{:.4f}'.format(i)) for i in dice_each]
            iou_each_f = [float('{:.4f}'.format(i)) for i in iou_each]

            print(count)
            metrics[0] /= count
            metrics[1] /= count
            print(metrics)
            dc, jc = metrics[0], metrics[1]

            metrics_seq[0] /= count_seq
            dice_seq = [float('{:.4f}'.format(i)) for i in metrics_seq[0]]
            metrics_seq[1] /= count_seq
            iou_seq = [float('{:.4f}'.format(i)) for i in metrics_seq[1]]

        print('Dice:{:.4f} IoU:{:.4f} Time:{:.4f}'.format(dc, jc, duration))
        print('Dice_seq1:{:.4f}, seq2:{:.4f}, seq3:{:.4f}, seq4:{:.4f}'.format(dice_seq[0], dice_seq[1], dice_seq[2],dice_seq[3]))
        print('IOU_seq1:{:.4f}, seq2:{:.4f}, seq3:{:.4f}, seq4:{:.4f}'.format(iou_seq[0], iou_seq[1], iou_seq[2],iou_seq[3]))
        summary_writer.add_scalar('Dice', dc, epoch_real)
        summary_writer.add_scalar('IoU', jc, epoch_real)
        return jc
    
    ##------------------------------ validation model for colon oct ------------------------------## 
    def val_map_oct(epoch,epoch_real):

        print('\n Val OCT @ Epoch: %d, Real Epoch: %d' % (epoch,epoch_real))
        model.eval()
        torch.cuda.empty_cache()
        metrics = np.zeros((2,))
        count = 0

        with torch.no_grad():
            for inputs in tqdm.tqdm(val_loader):
                inputs['image'] = inputs['image'].to(cfg.device).float()
                # print('shape:', inputs['image'].shape) #1,3,256,480
                tic = time.perf_counter()
                output,_ = model(inputs['image'].squeeze(0))

                output = F.interpolate(output, (ori_h,ori_w), mode='bilinear', align_corners=True)
                output = F.softmax(output,dim=1)
                output = torch.argmax(output,dim=1)
                output = output.cpu().numpy()
                duration = time.perf_counter() - tic

                dice = general_dice(inputs['label'].numpy(),
                                    output)  # dice containing each tool class
                iou = general_jaccard(inputs['label'].numpy(), output)

                frame_dice = np.mean([dice[i][1] for i in range(len(dice))])
                frame_iou = np.mean([iou[i][1] for i in range(len(dice))])
                #overall
                metrics[0] += frame_dice # dice of each frame
                metrics[1] += frame_iou
                count += 1


            print(count)
            metrics[0] /= count
            metrics[1] /= count
            print(metrics)
            dc, jc = metrics[0], metrics[1]

        print('Dice:{:.4f} IoU:{:.4f} Time:{:.4f}'.format(dc, jc, duration))
        summary_writer.add_scalar('Dice', dc, epoch)
        summary_writer.add_scalar('IoU', jc, epoch)
        return jc

    ##------------------------------ get the weight for vedios based on features for endovis18 dataset ------------------------------##   
    def get_weight_for_vedio_endo(epoch):
            
        print('\n computing affinity confidence for videos Endo, Epoch %d' % epoch)
        model.eval()
        Procedures = np.array([1,2,3,4,5,6,7,9,10,11,12,13,14,15,16])
        weight = np.array([0.4,0.4,0.4,0.4,0.4,0.5,0.6,0.7,0.8,0.9,1,1,1,1,1])
        p_sum_for_each_vedio = np.zeros((15,))
        n_sum_for_each_vedio = np.zeros((15,))
        count = np.zeros((15,))
        weight_final = np.zeros((15,))

        tic = time.perf_counter()
        for batch_idx, batch in tqdm.tqdm(enumerate(train_loader_1)):
            for k in batch:
                if not k=='path':
                    batch[k] = batch[k].to(device=cfg.device).float()
            #print('shape:', batch['image'].shape) #4, 3, 272, 480
            with torch.no_grad():
                #print(batch['image'].shape)
                _ , feature = model(batch['image'])
                _ , feature_1 = model(batch['image_1'])
                B, C, H, W = feature_1.shape
                label_ds = F.interpolate(batch['label'], size=[H,W], mode='nearest').squeeze(0)
                label_1_ds = F.interpolate(batch['label_1'], size=[H,W], mode='nearest').squeeze(0)
                
                p,n = TFAL_get_affinity(feature, feature_1, label_ds, label_1_ds, class_num =classes , H = h, W = w)

                if batch['path'][0] < 9:
                    ins = batch['path'][0].numpy() - 1
                else:
                    ins = batch['path'][0].numpy() - 2

                p_sum_for_each_vedio[ins] += p.cpu().numpy()  
                n_sum_for_each_vedio[ins] += n.cpu().numpy()
                count[ins] += 1

        reliability_pn = (p_sum_for_each_vedio + count - n_sum_for_each_vedio) / count
        p_thershold = np.mean(p_sum_for_each_vedio / count)
        n_thershold = np.mean((count-n_sum_for_each_vedio) / count)

        sort_p = np.argsort(p_sum_for_each_vedio)
        sort_n = np.argsort(count-n_sum_for_each_vedio)
        sort_pn = np.argsort(reliability_pn)

        video_uncertain = Procedures[sort_pn]
        video_uncertain = video_uncertain[:3]
        for i in range(len(weight_final)):
            weight_final[sort_pn[i]] = weight[i]

        print(Procedures[sort_pn])
        print(weight_final)

        return weight_final,p_thershold,n_thershold,video_uncertain
    
    ##------------------------------ get the weight for vedios based on features rat colon dataset ------------------------------##   
    def get_weight_for_vedio_oct(epoch):
            
        print('\n computing affinity confidence for videos OCT, Epoch %d' % epoch)
        model.eval()
        Procedures_mini = ['2T1','3C1','3T1','3T2','7C','10C','13C','15C']
        Procedures_mini_int = np.array([0,1,2,3,4,5,6,7])
        weight = np.array([0.4,0.4,0.4,0.6,0.8,1,1,1])
        p_sum_for_each_vedio = np.zeros((8,))
        n_sum_for_each_vedio = np.zeros((8,))
        count = np.zeros((8,))
        weight_final = np.zeros((8,))

        tic = time.perf_counter()
        for batch_idx, batch in tqdm.tqdm(enumerate(train_loader_1)):
            for k in batch:
                if not k=='path':
                    batch[k] = batch[k].to(device=cfg.device).float()
            #print('shape:', batch['image'].shape) #4, 3, 272, 480
            with torch.no_grad():
                #print(batch['image'].shape)
                _ , feature = model(batch['image'])
                _ , feature_1 = model(batch['image_1'])
                B, C, H, W = feature_1.shape
                label_ds = F.interpolate(batch['label'], size=[H,W], mode='nearest').squeeze(0)
                label_1_ds = F.interpolate(batch['label_1'], size=[H,W], mode='nearest').squeeze(0)
                
                p,n = TFAL_get_affinity(feature, feature_1, label_ds, label_1_ds, class_num =classes , H = h, W = w)
                ins = batch['path'][0][0]
                
                seq_ins = Procedures_mini.index(ins)

                p_sum_for_each_vedio[seq_ins] += p.cpu().numpy()  
                n_sum_for_each_vedio[seq_ins] += n.cpu().numpy()
                count[seq_ins] += 1

        reliability_pn = (p_sum_for_each_vedio + count - n_sum_for_each_vedio) / (count + 1e-14)
        p_thershold = np.mean(p_sum_for_each_vedio / (count + 1e-14))
        n_thershold = np.mean((count-n_sum_for_each_vedio) / (count + 1e-14))

        sort_p = np.argsort(p_sum_for_each_vedio)
        sort_n = np.argsort(count-n_sum_for_each_vedio)
        sort_pn = np.argsort(reliability_pn)

        video_uncertain = Procedures_mini_int[sort_pn]
        video_uncertain = video_uncertain[:6]
        for i in range(len(weight_final)):
            weight_final[sort_pn[i]] = weight[i]

        print(Procedures_mini_int[sort_pn])
        print(weight_final)

        return weight_final,p_thershold,n_thershold,video_uncertain
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
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.test_dir, exist_ok=True)
    saver = create_saver(cfg.local_rank, save_dir=cfg.ckpt_dir)
    logger = create_logger(cfg.local_rank, save_dir=cfg.log_dir)
    summary_writer = create_summary(cfg.local_rank, log_dir=cfg.log_dir)

    print = logger.info
    print(cfg)
    ##------------------------------ dataset ------------------------------##
    print('Setting up data...')
    
    if cfg.dataset=='endovis2018':
        h,w = [cfg.h,cfg.w]
        ori_h, ori_w = [1024, 1280]
        print('size of endovis2018 data %d, %d.' %(h,w))
        if cfg.data_type=='clean':
            from dataset.Endovis2018_MSS_TFAL import endovis2018
            train_dataset = endovis2018('train_clean', t=cfg.t, arch='swinPlus',rate=1, global_n=cfg.global_n,h = h, w = w)
            val_dataset = endovis2018('test_part', t=cfg.t,arch='swinPlus', rate=1, global_n=cfg.global_n,h = h, w = w)
            classes = train_dataset.class_num
        elif cfg.data_type=='noisy':
            from dataset.Endovis2018_MSS_TFAL import endovis2018
            train_dataset = endovis2018('train', t=cfg.batch_size, arch='swinPlus',rate=1, global_n=cfg.global_n, data_ver=cfg.data_ver,h = h, w = w)
            train_dataset_1 = endovis2018('train', t=1, arch='swinPlus',rate=1, global_n=cfg.global_n, data_ver=cfg.data_ver,h = h, w = w)
            val_dataset = endovis2018('test_part', t=cfg.batch_size,arch='swinPlus', rate=1, global_n=cfg.global_n, data_ver=cfg.data_ver,h = h, w = w)
            classes = train_dataset.class_num
    elif cfg.dataset=='colon_oct':
        h,w = [cfg.h,cfg.w]
        ori_h, ori_w = [1024, 1024]
        print('size of colon_oct data %d, %d.' %(h,w))
        from dataset.Colon_OCT import Colon_OCT_MSSTFAL
        train_dataset = Colon_OCT_MSSTFAL('train', t=cfg.batch_size, arch='swinPlus',rate=1, global_n=cfg.global_n,h = h, w = w)
        train_dataset_1 = Colon_OCT_MSSTFAL('train', t=1, arch='swinPlus',rate=1, global_n=cfg.global_n,h = h, w = w)
        val_dataset = Colon_OCT_MSSTFAL('test_part', t=cfg.batch_size,arch='swinPlus', rate=1, global_n=cfg.global_n,h = h, w = w)
        classes = train_dataset.class_num
    ##------------------------------ build model ------------------------------##        
    if 'puredeeplab' in cfg.arch:
        from net.Ours.base18 import DeepLabV3Plus
        model = DeepLabV3Plus(classes, 18)
        model_pred = DeepLabV3Plus(classes, 18)
    else:
        raise NotImplementedError
    ##------------------------------ combile model ------------------------------## 
    optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
    loss_functions = {'bce': BCELoss(), 'ohem':OhemCELoss2D(w*h//16//(cfg.downsample**2)),'myohem':myOhemCELoss2D(w*h//16//(cfg.downsample**2)), 'dice':DiceLoss}
    compute_loss = loss_functions[cfg.loss] 
    
    torch.cuda.empty_cache()
    print('Starting training...')
    cfg.num_epochs = int(cfg.iterations / len(train_dataset)) # 26
    cfg.val_interval = int(len(train_dataset) / cfg.batch_size) # 558
    best = 0
    best_ep = 0
    T1 = cfg.T1
    T2 = cfg.T2
    T3 = cfg.T3
    if cfg.dataset=='endovis2018':
        weight_for_vedio = np.zeros((15,))
    else:
        weight_for_vedio = np.zeros((8,))

    gpus = cfg.gpus.split(',')
    if len(cfg.gpus)>1:
        model = nn.DataParallel(model, device_ids=list(map(int,gpus))).cuda()
        model_pred = nn.DataParallel(model_pred, device_ids=list(map(int,gpus))).cuda()
    else:
        model = model.to(cfg.device)
        model_pred = model.to(cfg.device)
    ##------------------------------ dataloader ------------------------------## 
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                 batch_size=1,
                                 shuffle= True,
                                 num_workers=cfg.num_workers,
                                 pin_memory=True,
                                 drop_last=True)
    train_loader_1 = torch.utils.data.DataLoader(train_dataset_1,
                                 batch_size=1,
                                 shuffle= False,
                                 num_workers=cfg.num_workers,
                                 pin_memory=True,
                                 drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                       shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    ##------------------------------ resume training ------------------------------## 
    if cfg.reset:
        pre_ckpt_path = os.path.join(cfg.root_dir, cfg.log_name, 'ckpt', 'latestcheckpoint.t7')
        print('initialize the model from: %s' % pre_ckpt_path)
        model = load_model_full(model, pre_ckpt_path)
        best_ep = cfg.reset_ep
        if cfg.dataset=='endovis2018':
            best = val_map_endo(best_ep,(best_ep-1)*4)
        else:
            best = val_map_oct(best_ep,(best_ep-1)*4)
    ##------------------------------ choose the train function for different dataset ------------------------------##
    def train(epoch,stage,val_interval,best,best_ep,weight_for_vedio,p_thershold,n_thershold,video_uncertain):
        if cfg.dataset=='endovis2018':
            return train_endo(epoch,stage,val_interval,best,best_ep,weight_for_vedio,p_thershold,n_thershold,video_uncertain)
        elif cfg.dataset=='colon_oct':
            return train_oct(epoch,stage,val_interval,best,best_ep,weight_for_vedio,p_thershold,n_thershold,video_uncertain)
        
    ##------------------------------ choose the weight calculation function for different dataset ------------------------------##
    def weight(epoch):
        if cfg.dataset=='endovis2018':
            return get_weight_for_vedio_endo(epoch)
        elif cfg.dataset=='colon_oct':
            return get_weight_for_vedio_oct(epoch)
    ##------------------------------ training section ------------------------------## 
    torch.cuda.empty_cache()
    for epoch in range(best_ep + 1, cfg.num_epochs + 1):
        
        if epoch < T1:
            stage = 1
            print('\n Stage 1, Epoch: %d' % epoch)
            best,best_ep = train(epoch,stage,cfg.val_interval,best,best_ep,weight_for_vedio,[],[],[])
        elif epoch < T2 :
            stage = 2
            print('\n Stage 2, Epoch: %d' % epoch)
            weight_for_vedio, _, _, _ = weight(epoch)
            best,best_ep = train(epoch,stage,cfg.val_interval,best,best_ep,weight_for_vedio,[],[],[])
        elif epoch < T3 :
            stage = 3
            print('\n Stage 3, Epoch: %d' % epoch)
            weight_for_vedio,p_thershold,n_thershold, _ = weight(epoch)
            print('\n p_thershold: %f, n_thershold: %f' % (p_thershold,n_thershold))
            best,best_ep = train(epoch,stage,cfg.val_interval,best,best_ep,weight_for_vedio,p_thershold,n_thershold,[])
        else :
            stage = 4
            print('\n Stage 4, Epoch: %d' % epoch)
            weight_for_vedio,p_thershold,n_thershold, video_uncertain = weight(epoch)
            print('\n p_thershold: %f, n_thershold: %f' % (p_thershold,n_thershold))

            pred_ckpt_path = os.path.join(cfg.ckpt_dir, 'epoch_{}_checkpoint.t7'.format(best_ep))
            print('model for correction path: %s' % pred_ckpt_path)
            model_pred = load_model_full(model_pred, pred_ckpt_path)
            best,best_ep = train(epoch,stage,cfg.val_interval,best,best_ep,weight_for_vedio,p_thershold,n_thershold,video_uncertain)
        
    summary_writer.close()
    ################################################ main part ################################################

if __name__ == '__main__':
    with DisablePrint(local_rank=cfg.local_rank):
        main()
