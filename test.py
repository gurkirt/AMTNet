
from layers.modules.feat_pooling import FeatPooling

from layers.modules.spatial_pooling import SpatialPool

import numpy as np
import torch, pdb

for k in [1, 3, 5, 10, 19, 38]:
    print(k)
    x = torch.randn(8,256,k,k, device='cuda:0')
    cc = torch.nn.Conv2d(256,256,kernel_size=1, stride=1).cuda()
    x = cc(x)
    # ft = SpatialPool(25).cuda()
    ft = FeatPooling(256, np.identity(k**2)).cuda()
    # lin = torch.nn.Linear(4608,1).cuda()
    y = torch.rand(4, device='cuda:0')

    x = ft(x)
    print(x.size())
    loss = x[:, 0, 0] - y
    loss = loss.sum()
    print(loss)

    loss.backward()