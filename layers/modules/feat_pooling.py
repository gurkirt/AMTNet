import torch
import torch.nn as nn
from torch.autograd import Variable
from layers.modules.spatial_pooling import SpatialPool

# fusion here means something different than two stream fusion
class FeatPooling(nn.Module):
    def __init__(self, fd, afnMat, afthresh=0.6, kd=3, fusion_type='cat', seq_len=2):
        super(FeatPooling, self).__init__()
        print('*** feat_pooling.py : __init__() *** fustiontype',fusion_type, ' kd ', kd, ' afthresh ', afthresh,' feature size ',fd)
        # fusion_type type here mean fusion accross frame; 'cat' fusion concatenate the features from frames in t sequnce 
        # fusion here means something different than two stream fusion
        self.afnMat = afnMat
        self.amt = afthresh # affintiy matrix threshold
        self.fd = fd
        self.fusion_type = fusion_type
        self.seq_len = seq_len
        amd = afnMat.shape  # am dim. eg. 1444 x 1444 for fm 38 x 38
        with torch.no_grad():
            ffm_index = torch.LongTensor(seq_len,amd[0]*amd[0])
            amc = 0 # affinity matrix element count
            for c1 in range(amd[0]):
                for c2 in range(amd[1]):
                    if self.afnMat[c1, c2] >= self.amt:
                        for s in range(seq_len):
                            ffm_index[s,amc] = c1
                        amc += 1
            ffm_index = ffm_index[:,:amc]
            ffm_index = ffm_index.cuda()
        ffm_index.requires_grad = False   
        #print('print(self.ffm_index_c1', ffm_index_c1.size(), afnMat.shape)
        self.register_buffer("ffm_index", ffm_index)  # , requires_grad=False)

        self.amd = amd
        self.amc = amc
        self.fdd = self.fd * kd * kd
        self.kd = kd

        if self.kd == 1:
            self.spatial_pool = nn.AvgPool2d(3, stride=1, padding=1)
            self.spatial_pool = self.spatial_pool.cuda()
        else:
            self.spatial_pool = SpatialPool(self.amd[0], kd)


    def forward(self, fm):
        #print('Len of input feature maps ', len(fm))
        #print('INput feature map size', fm.size())
        pfm = self.spatial_pool(fm)  # spatially poolling feature maps

        if self.kd == 1:
            pfm = pfm.permute(0,2,3,1).contiguous()
            pfm = pfm.view(pfm.size(0), -1, pfm.size(3))

        pfm = pfm.unsqueeze(0)
        #print('INput feature map size 1:', pfm.size())
        pfm = pfm.view(-1, self.seq_len, pfm.size(2), pfm.size(3))
        #print('INput feature map size 2:', pfm.size())
        pfm = pfm.permute(1, 0, 2, 3).contiguous()
        #print('INput feature map size 3:', pfm.size())
        pfm1 = pfm[0]
        pfm2 = pfm[1]

        final_fm = None
        if self.fusion_type == 'cat':
            final_fm = torch.cat((pfm1.index_select(1, self.ffm_index[0]), pfm2.index_select(1, self.ffm_index[1])), 2)
            for s in range(2, self.seq_len):
                    pfm2 = pfm[s]
                    final_fm = torch.cat((final_fm,pfm2.index_select(1, Variable(self.ffm_index[s]))), 2)
        elif self.fusion_type == 'mul':
            final_fm = (pfm1.index_select(1, Variable(self.ffm_index[0]))) * (pfm2.index_select(1, Variable(self.ffm_index[1])))
            for s in range(2, self.seq_len):
                    pfm2 = pfm[s]
                    final_fm *= pfm2.index_select(1, Variable(self.ffm_index[s]))
        elif self.fusion_type == 'mean':
            final_fm = ((pfm1.index_select(1, Variable(self.ffm_index[0]))) + (pfm2.index_select(1, Variable(self.ffm_index[1]))))
            for s in range(2, self.seq_len):
                    pfm2 = pfm[s]
                    final_fm += pfm2.index_select(1, Variable(self.ffm_index[s]))
            final_fm /= float(self.seq_len)

        elif self.fusion_type == 'sum':
            final_fm = (
            (pfm1.index_select(1, Variable(self.ffm_index[0]))) + (pfm2.index_select(1, Variable(self.ffm_index[1]))))
            for s in range(2, self.seq_len):
                pfm2 = pfm[s]
                final_fm += pfm2.index_select(1, Variable(self.ffm_index[s]))

        else:
            raise Exception('Supply correct fusion type ')
        # print('pooled feature size', final_fm.size())
        return final_fm




