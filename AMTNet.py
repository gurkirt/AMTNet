import os, pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from data import v2
from layers import *
from layers.modules.feat_pooling import FeatPooling
from torch.nn.parameter import Parameter

# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(), conv7, nn.ReLU()]
    return layers


def add_extras(cfg, i):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers




basemodel = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}


class SSD_CORE(nn.Module):

        def __init__(self, input_frames, size, seq_len =2, kd=3, featmap_fusion='cat'):
            super(SSD_CORE, self).__init__()

            self.vgg = nn.ModuleList(vgg(basemodel[str(size)], input_frames*3))
            # Layer learns to scale the l2 normalized features from conv4_3
            self.L2Norm = L2Norm(512, 20)
            self.extras = nn.ModuleList(add_extras(extras[str(size)], 1024))
            self.seq_len = seq_len
            self.size = size
            self.kd = kd
            self.scale = kd*kd*1.0

            self.fmd = [512, 1024, 512, 256, 256, 256]
            self.feature_maps = [38, 19, 10, 5, 3, 1]

            self.featPool0 = FeatPooling(self.fmd[0], np.identity(38 ** 2), afthresh=0.9, kd=kd, fusion_type=featmap_fusion, seq_len=seq_len)
            self.featPool1 = FeatPooling(self.fmd[1], np.identity(19 ** 2), afthresh=0.9, kd=kd, fusion_type=featmap_fusion, seq_len=seq_len)
            self.featPool2 = FeatPooling(self.fmd[2], np.identity(10 ** 2), afthresh=0.9, kd=kd, fusion_type=featmap_fusion, seq_len=seq_len)
            self.featPool3 = FeatPooling(self.fmd[3], np.identity(5 ** 2), afthresh=0.9, kd=kd, fusion_type=featmap_fusion, seq_len=seq_len)
            self.featPool4 = FeatPooling(self.fmd[4], np.identity(3 ** 2), afthresh=0.9, kd=kd, fusion_type=featmap_fusion, seq_len=seq_len)
            self.featPool5 = FeatPooling(self.fmd[5], np.identity(1 ** 2), afthresh=0.9, kd=kd, fusion_type=featmap_fusion, seq_len=seq_len)

        def forward(self, x):

            x = x.view(-1, x.size(2), x.size(3), x.size(4))
            _sources = self.baseforward(x)
            sources = [0,1,2,3,4,5]
            for i, s in enumerate(_sources):
                sources[i] = s/self.scale
            pooled_source = []
            # print(sources[0].size())
            pooled_source.append(self.featPool0(sources[0]))
            pooled_source.append(self.featPool1(sources[1]))
            pooled_source.append(self.featPool2(sources[2]))
            pooled_source.append(self.featPool3(sources[3]))
            pooled_source.append(self.featPool4(sources[4]))
            pooled_source.append(self.featPool5(sources[5]))
            #print('pooled_source size', pooled_source[0].size())

            return pooled_source

        def baseforward(self, x):

            sources = list()
            pooled_source = []
            # apply vgg up to conv4_3 relu
            for k in range(23):
                x = self.vgg[k](x)

            s = self.L2Norm(x)
            sources.append(s)

            # apply vgg up to fc7
            for k in range(23, len(self.vgg)):
                x = self.vgg[k](x)
            sources.append(x)

            # apply extra layers and cache source layer outputs
            for k, v in enumerate(self.extras):
                x = F.relu(v(x))
                if k % 2 == 1:
                    sources.append(x)

            return sources

        def load_my_state_dict(self, state_dict, input_frames=1):

            own_state = self.state_dict()
            # print('\n\n input_frames {:d}\n\n'.format(input_frames))
            # print('OWN KEYS: ', own_state.keys())
            # print('Loaded KEYS: ', state_dict.keys())
            # pdb.set_trace()
            
            for name, param in state_dict.items():
                name1 = name.split('.')
                name2 = '.'.join(name1[2:])
                # pdb.set_trace()
                if name in own_state.keys() or name2 in own_state.keys():
                    if name2 in own_state.keys():
                        name = name2
                    # print(name)
                    match = False
                    own_size = own_state[name].size()
                    if isinstance(param, Parameter):
                        param = param.data
                    param_size = param.size()
                    try:
                        if len(param_size)>2 and  param_size[1] != own_size[1]:
                            param = param.repeat(1, int(own_size[1]/param_size[1]), 1, 1)/float(own_size[1]/param_size[1])
                            own_state[name].copy_(param)
                        else:
                            own_state[name].copy_(param)
                    except Exception:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
                else:
                    print('NAME IS NOT IN OWN STATE::>' + name)
            # pdb.set_trace()


mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}

def multibox(cfg, num_classes, fusion_num_muliplier, seq_len=2, kd=3):
    loc_layers = []
    conf_layers = []
    fmd = [512, 1024, 512, 256, 256, 256]  # feature map depth/channel size
    fmd_mul = fusion_num_muliplier
    for i in range(len(fmd)):
        inpd = fmd[i]*seq_len*kd*kd*fmd_mul
        print('Feature map size', inpd)
        out_dim_reg = cfg[i] * 4 * seq_len
        out_dim_cls = cfg[i] * num_classes
        head_reg = nn.Linear(inpd, out_dim_reg)
        head_conf = nn.Linear(inpd, out_dim_cls)
        head_reg.bias.data.fill_(0)
        head_conf.bias.data.fill_(0)
        loc_layers += [head_reg]
        conf_layers += [head_conf]

    return loc_layers, conf_layers

class AMTNet(nn.Module):
    def __init__(self, args):
        #num_classes, seq_len=2, fusion_type='cat', kd=3):
        super(AMTNet, self).__init__()
        self.fusion = args.fusion
        self.core_base = SSD_CORE(args.input_frames_base, args.ssd_dim, args.seq_len, kd=args.kd)
        if self.fusion:
            self.core_extra = SSD_CORE(args.input_frames_extra, args.ssd_dim, args.seq_len, kd=args.kd)
        self.fusion_type = args.fusion_type
        self.num_classes = args.num_classes
        self.seq_len = args.seq_len
        head = multibox(mbox[str(args.ssd_dim)], args.num_classes, args.fusion_num_muliplier, args.seq_len, args.kd)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

    def forward(self, x):

        pooled_base = self.core_base(x[0])
        loc = list()
        conf = list()

        if self.fusion:
            pooled_extra = self.core_extra(x[1])
            # apply multibox head to source layers
            # pdb.set_trace()
            for (x1, x2, l, c) in zip(pooled_base, pooled_extra, self.loc, self.conf):
                # print('x_norm', x.norm())
                # pdb.set_trace()
                if self.fusion_type == 'cat':
                    x = torch.cat((x1, x2), 2)
                elif self.fusion_type == 'sum':
                    x = x1 + x2
                elif self.fusion_type == 'mean':
                    x = (x1 + x2) / 2.0
                else:
                    raise Exception('Supply correct fusion type')
                locs = l(x)
                locs = locs.view(locs.size(0), -1)
                loc.append(locs)
                
                confs = c(x)
                confs  = confs.view(confs.size(0), -1)
                conf.append(confs)  # .contiguous())
        else:
            # apply multibox head to source layers
            for (x, l, c) in zip(pooled_base, self.loc, self.conf):
                locs = l(x)
                locs = locs.view(locs.size(0), -1)
                loc.append(locs)
                
                confs = c(x)
                confs  = confs.view(confs.size(0), -1)
                conf.append(confs) # .contiguous())
        # pdb.set_trace()
        loc = torch.cat(loc, 1)
        conf = torch.cat(conf, 1)

        
        return conf.view(conf.size(0), -1, self.num_classes), loc.view(loc.size(0), -1, 4*self.seq_len),
