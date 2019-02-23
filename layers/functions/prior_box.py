import torch
from math import sqrt as sqrt
from itertools import product as product

def stack_4_seq_len(x1,y1,x2,y2,seq_len):
    box = [x1,y1,x2,y2]
    for i in range(seq_len-1):
        box.append(x1)
        box.append(y1)
        box.append(x2)
        box.append(y2)

    return box

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.
    """
    def __init__(self, cfg, seq_len=2):
        super(PriorBox, self).__init__()
        self.seq_len = seq_len
        # self.type = cfg.name
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        # TODO merge these
        if self.version == 'v2':
            for k, f in enumerate(self.feature_maps):
                for i, j in product(range(f), repeat=2):
                    f_k = self.image_size / self.steps[k]
                    # unit center x,y
                    cx = (j + 0.5) / f_k
                    cy = (i + 0.5) / f_k

                    # aspect_ratio: 1
                    # rel size: min_size
                    s_k = self.min_sizes[k]/self.image_size
                    mean += stack_4_seq_len(cx, cy, s_k, s_k, self.seq_len)

                    # aspect_ratio: 1
                    # rel size: sqrt(s_k * s_(k+1))
                    s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                    mean += stack_4_seq_len(cx, cy, s_k_prime, s_k_prime, self.seq_len)

                    # rest of aspect ratios
                    for ar in self.aspect_ratios[k]:
                        mean += stack_4_seq_len(cx, cy, s_k*sqrt(ar), s_k/sqrt(ar), self.seq_len)
                        mean += stack_4_seq_len(cx, cy, s_k/sqrt(ar), s_k*sqrt(ar), self.seq_len)

        else:
            raise 'wrong version'

        output = torch.cuda.FloatTensor(mean).view(-1, 4*self.seq_len)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
