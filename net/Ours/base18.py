import os
import torch
import torch.nn as nn
import sys,time

from net.utils.helpers import maybe_download
from net.utils.layer_factory import conv1x1, conv3x3, convbnrelu, CRPBlock
from net.Ours.Module import *
from net.Ours.resnet import *
from net.Ours.ASPP import *
from net.Ours.swin_512 import SwinTransformerLayerv5


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, layers=18):
        super(DeepLabV3Plus, self).__init__()

        self.num_classes = num_classes
        if layers == 18:
            self.resnet = ResNet18_OS8()
            self.aspp = ASPP(num_classes=256,num_channel = 512)
        elif layers == 50:
            # self.resnet = ResNet50_OS16()
            self.aspp = ASPP_Bottleneck(num_classes=256)
        self.project = nn.Sequential(
            nn.Conv2d(512, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1))
        print('network architecture: v3PLUS')

    def forward(self, x):

        # (x has shape (batch_size, 3, h, w))
        h = x.size()[2]
        w = x.size()[3]
        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/8, w/8)) 2,512,34,60
        aspp_output = self.aspp(feature_map)  # (shape: (batch_size, num_classes, h/16, w/16)) 2,256,34,60

        low_level_feature = self.project(feature_map) #2,48,34,60
        aspp_output = F.interpolate(aspp_output, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False) #2,26,34,60
        output = self.classifier(torch.cat([low_level_feature, aspp_output], dim=1))
        feature_out = F.normalize(torch.cat([low_level_feature, aspp_output], dim=1), dim=1)
        output = F.interpolate(output, size=(h, w), mode="bilinear")  # (shape: (batch_size, num_classes, h, w))
        return output,feature_out


class TswinPlus(nn.Module):
    ## input size = 256*448; t=3; no temporal swin
    def __init__(self, num_classes,h , w):
        super(TswinPlus, self).__init__()
        #self.memory = Memory(512)
        self.swin = SwinTransformerLayerv5(input_resolution=(int(h / 8),int(w / 8)))
        self.resnet = ResNet18_OS8()
        self.aspp = ASPP(num_classes=256)
        self.project1 = nn.Sequential(
            nn.Conv2d(512, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))
        self.project2 = nn.Sequential(
            nn.Conv2d(512, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))
        self.project3 = nn.Sequential(
            nn.Conv2d(1024, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))

        self.classifier = nn.Sequential(
            nn.Conv2d(400, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1))
        print('network architecture: TswinPlus')

    def forward(self, x):
        tic = time.perf_counter()
        b, t, _, w, h = x.size()  #
        # 2, 3, 3, 512, 640
        seq = []

        for i in range(t):
            tensor = self.resnet(x[:, i])
            # x[:, i]: 2,3,512,640 = > tensor: 2,512,64,80
            seq.append(tensor.unsqueeze(1))  # 2,1,512,64,80

        tem = torch.cat(seq, dim=1)  # b,t,c,w,h #2,3,512,64,80
        res_output = tem[:, -1, :, :, :] #2,512,64,80

        tem1, tem2 = self.swin(tem)
        temporal_output1 = tem1[:, -1, :, :, :] #2,512,64,80
        temporal_output2 = tem2[:, -1, :, :, :] #2,512,32,40
        aspp_output = self.aspp(temporal_output2) #2,512,32,40

        res_output_pro = self.project1(res_output)
        temporal_output1_pro = self.project2(temporal_output1)
        temporal_output2_pro = self.project3(temporal_output2)
        temporal_output2_pro = F.interpolate(temporal_output2_pro, size=res_output_pro.shape[2:], mode="bilinear", align_corners=False)
        aspp_output = F.interpolate(aspp_output, size=res_output_pro.shape[2:], mode='bilinear', align_corners=False)
        
        output = self.classifier(torch.cat([res_output_pro, temporal_output1_pro, temporal_output2_pro, aspp_output], dim=1)) #2,12,64,80
        feature = torch.cat([res_output_pro, temporal_output1_pro, temporal_output2_pro, aspp_output], dim=1)
        output = nn.functional.interpolate(output, (w, h), mode="bilinear")

        return output,feature


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
if __name__ == '__main__':
    import torch
    os.environ['CUDA_VISIBLE_DEVICES']= '0,3'
    net = DeepLabV3Plus(num_classes=12, layers=18).cuda()
    # net = TswinPlus(num_classes=12).cuda()
    with torch.no_grad():
        y1,pred_1 = net(torch.randn(4, 3, 256, 320).cuda())
        y21,pred_2 = net(torch.randn(4, 3, 256, 320).cuda())
        print('output 1 shape:',y1.shape)
        print('pred_1 shape:',pred_1.shape)

    
