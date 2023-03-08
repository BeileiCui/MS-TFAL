import os,tqdm,sys,time,argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

import numpy as np
import torch.cuda.amp as amp
scaler = amp.GradScaler()

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.distributed as dist

from utils.losses import BCELoss,OhemCELoss2D,DiceLoss

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
parser.add_argument('--log_name', type=str, default='DLV3PLUS_clean')
parser.add_argument('--data_type', type=str, choices=['clean','noisy'], default='noisy')
parser.add_argument('--data_ver', type=int, default=0)
parser.add_argument('--arch', type=str, choices=['puredeeplab18','swinPlus','RAUNet'], default='swinPlus')
parser.add_argument('--pre_log_name', type=str, default=None)
parser.add_argument('--pre_checkpoint', type=str, default=None)

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--loss', type=str, default='ohem')

parser.add_argument('--gpus', type=str, default='3')
parser.add_argument('--downsample', type=int, default=1)
parser.add_argument('--h', type=int, default=512)
parser.add_argument('--w', type=int, default=640)

parser.add_argument('--log_interval', type=int, default=50)
parser.add_argument('--val_interval', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)

parser.add_argument('--t', type=int, default=1)
parser.add_argument('--step', type=int, default=1)

parser.add_argument('--ver', type=int, default=1)
parser.add_argument('--tag', type=int, default=1)


# parser.add_argument('--freeze_name', type=str, )
# parser.add_argument('--spatial_layer', type=int, )
parser.add_argument('--global_n', type=int, default=0)

parser.add_argument('--pretrain_ep', type=int, default=20)
parser.add_argument('--decay', type=int, default=2)

parser.add_argument('--reset', type=str, default=None)
parser.add_argument('--reset_ep', type=int)

cfg = parser.parse_args()
##------------------------------ Training settings ------------------------------##

def main():

    ################################################ def part ################################################

    ##------------------------------ train model ------------------------------##   
    def train(epoch):
        print('\n Epoch: %d' % epoch)
        model.train()
        tic = time.perf_counter()
        tr_loss = []
        for batch_idx, batch in enumerate(train_loader):
            for k in batch:
                if not k=='path':
                    batch[k] = batch[k].to(device=cfg.device).float()
            # print('shape of input image:', batch['image'].shape) #4, 3, 272, 480
            with amp.autocast():
                #print(batch['image'].shape)
                outputs , _ = model(batch['image'])
                if cfg.loss == 'ohem':
                    loss = compute_loss(outputs, batch['label'].long())
                else:
                    loss = compute_loss(outputs, batch['label'])
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
        summary_writer.add_scalar('Tr_loss', np.mean(tr_loss), epoch)
        return      
    ##------------------------------ validation model ------------------------------##   
    def val_map_endo(epoch):

        print('\n Val@Epoch: %d' % epoch)
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
                # print('shape:', inputs['image'].shape) #1,3,256,480
                tic = time.perf_counter()
                output,_ = model(inputs['image'])

                output = F.interpolate(output, (ori_h,ori_w), mode='bilinear', align_corners=True)
                output = F.softmax(output,dim=1)
                output = torch.argmax(output,dim=1)
                output = output.cpu().numpy()
                duration = time.perf_counter() - tic

                dice = general_dice(inputs['label'].numpy(),
                                    output)  # dice containing each tool class
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
        summary_writer.add_scalar('Dice', dc, epoch)
        summary_writer.add_scalar('IoU', jc, epoch)
        return jc
    
    def val_map_oct(epoch):

        print('\n Val@Epoch: %d' % epoch)
        model.eval()
        torch.cuda.empty_cache()
        metrics = np.zeros((2,))
        count = 0

        with torch.no_grad():
            for inputs in tqdm.tqdm(val_loader):
                inputs['image'] = inputs['image'].to(cfg.device).float()
                # print('shape:', inputs['image'].shape) #1,3,256,480
                tic = time.perf_counter()
                output,_ = model(inputs['image'])

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

    cfg.log_dir = os.path.join(cfg.root_dir, cfg.log_name, 'logs')
    cfg.ckpt_dir = os.path.join(cfg.root_dir, cfg.log_name, 'ckpt')
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
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
            from dataset.Endovis2018_backbone import endovis2018
            train_dataset = endovis2018('train_clean', t=cfg.t, arch='swinPlus',rate=1, global_n=cfg.global_n,h = h, w = w)
            val_dataset = endovis2018('test_part', t=cfg.t,arch='swinPlus', rate=1, global_n=cfg.global_n,h = h, w = w)
            classes = train_dataset.class_num
        elif cfg.data_type=='noisy':
            from dataset.Endovis2018_backbone import endovis2018
            train_dataset = endovis2018('train', t=cfg.t, arch='swinPlus',rate=1, global_n=cfg.global_n, data_ver=cfg.data_ver,h = h, w = w)
            val_dataset = endovis2018('test_part', t=cfg.t,arch='swinPlus', rate=1, global_n=cfg.global_n, data_ver=cfg.data_ver,h = h, w = w)
            classes = train_dataset.class_num
    elif cfg.dataset=='colon_oct':
        h,w = [cfg.h,cfg.w]
        ori_h, ori_w = [1024, 1024]
        print('size of colon_oct data %d, %d.' %(h,w))
        from dataset.Colon_OCT import Colon_OCT
        train_dataset = Colon_OCT('train', t=cfg.t, arch='swinPlus',rate=1, global_n=cfg.global_n,h = h, w = w)
        val_dataset = Colon_OCT('test_part', t=cfg.t,arch='swinPlus', rate=1, global_n=cfg.global_n,h = h, w = w)
        classes = train_dataset.class_num
    ##------------------------------ build model ------------------------------##        
    if 'puredeeplab' in cfg.arch:
        from net.Ours.base18 import DeepLabV3Plus
        model = DeepLabV3Plus(classes, 18)
    elif 'swin' in cfg.arch:
        from net.Ours.base18 import TswinPlus
        model = TswinPlus(classes,h,w)
    elif 'RAUNet' in cfg.arch:
        from net.Ours.RAUNet import RAUNet
        model = RAUNet(classes)
    else:
        raise NotImplementedError
    # load pretrain model
    if cfg.pre_log_name is not None:
        cfg.pre_ckpt_path = os.path.join(cfg.root_dir, cfg.pre_log_name, 'ckpt', 'checkpoint.t7')
        print('initialize the model from:', cfg.pre_ckpt_path)
        model = load_model(model, cfg.pre_ckpt_path)
    ##------------------------------ combile model ------------------------------## 
    optimizer = torch.optim.Adam(model.parameters(), cfg.lr,weight_decay=cfg.weight_decay)
    loss_functions = {'bce': BCELoss(), 'ohem':OhemCELoss2D(w*h//16//(cfg.downsample**2)), 'dice':DiceLoss}
    compute_loss = loss_functions[cfg.loss] 
    
    torch.cuda.empty_cache()
    print('Starting training...')
    best = 0
    best_ep = 0
    
    gpus = cfg.gpus.split(',')
    if len(cfg.gpus)>1:
        model = nn.DataParallel(model, device_ids=list(map(int,gpus))).cuda()
    else:
        model = model.to(cfg.device)
    ##------------------------------ dataloader ------------------------------## 
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                 batch_size=cfg.batch_size,
                                 shuffle= True,
                                 num_workers=cfg.num_workers,
                                 pin_memory=True,
                                 drop_last=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                       shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    ##------------------------------ resume training ------------------------------## 
    if cfg.reset:
        pre_ckpt_path = os.path.join(cfg.root_dir, cfg.log_name, 'ckpt', 'latestcheckpoint.t7')
        print('initialize the model from: %s' % pre_ckpt_path)
        model = load_model_full(model, pre_ckpt_path)
        best_ep = cfg.reset_ep
        
        if cfg.dataset=='endovis2018':
            best = val_map_endo(best_ep)
        else:
            best = val_map_oct(best_ep)
    ##------------------------------ training section ------------------------------## 
    for epoch in range(best_ep + 1, cfg.num_epochs + 1):
        train(epoch)
        if cfg.val_interval > 0 and epoch % cfg.val_interval == 0:
            if cfg.dataset=='endovis2018':
                save_map = val_map_endo(epoch)
            else:
                save_map = val_map_oct(epoch)
            if save_map > best:
                best = save_map
                best_ep = epoch
                print(saver.save(model.state_dict(), 'epoch_{}_checkpoint'.format(epoch)))
            else:
                if epoch - best_ep > 100:
                    break
            print(saver.save(model.state_dict(), 'latestcheckpoint'))
    summary_writer.close()

    ################################################ main part ################################################

if __name__ == '__main__':
    with DisablePrint(local_rank=cfg.local_rank):
        # tensorboard --logdir '/mnt/data-hdd/beilei/nll_new/results/endo18/STswin_clean_ver_0/logs'
        main()
