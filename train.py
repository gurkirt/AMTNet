""" 

Author: Gurkirt Singh 
Modified from https://github.com/gurkirt/realtime-action-detection
Licensed under The MIT License [see LICENSE for details]

"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import v2, ActionDetection, NormliseBoxes, detection_collate, CLASSES, BaseTransform
from utils.augmentations import SSDAugmentation
from layers.modules import MultiboxLoss
from layers.functions import PriorBox
from layers import MatchPrior
from AMTNet import AMTNet
import numpy as np
import time, pdb
from utils.evaluation import evaluate_detections
from layers.box_utils import nms, decode_seq
from utils import  AverageMeter
from torch.optim.lr_scheduler import MultiStepLR
# from torchviz import make_dot, make_dot_from_trace

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='AMTNet detection training script')
parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--dataset', default='ucf24', help='pretrained base model')
parser.add_argument('--train_split', default=1, type=int, help='Split id')
parser.add_argument('--ssd_dim', default=300, type=int, help='Input Size for SSD') # only support 300 now
parser.add_argument('--seq_len', default=2, type=int, help='Input sequence length ')
parser.add_argument('--seq_gap', default=0, type=int, help='Gap between the frame of sequence')
parser.add_argument('--fusion_type', default='cat', type=str, 
                    help='Fusion type to fuse from sequence of frames; options are SUM, CAT and NONE')
                    # 
parser.add_argument('--input_type_base', default='rgb', type=str, help='INput tyep default rgb can take flow (brox or fastOF) as well')
parser.add_argument('--input_type_extra', default='brox', type=str, help='INput tyep default brox can take flow (brox or fastOF) as well')
parser.add_argument('--input_frames_base', default=1, type=int, help='Number of input frame, default for rgb is 1')
parser.add_argument('--input_frames_extra', default=5, type=int, help='Number of input frame, default for flow is 5')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--max_iter', default=300000, type=int, help='Number of training iterations')
parser.add_argument('--val_step', default=10000, type=int, help='Number of training iterations')
parser.add_argument('--cuda', default=1, type=str2bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=2, type=int, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--stepvalues', default='10000,30000', type=str, help='step points for learning rate drop')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD at for stepwise schedule')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--vis_port', default=8095, type=int, help='Port for Visdom Server')
parser.add_argument('--data_root', default='/mnt/sun-gamma/', help='Location of where in data is located like images and annotation file')
parser.add_argument('--save_root', default='/mnt/sun-gamma/', help='Location to where we wanr save the checkpoints of models')
parser.add_argument('--iou_thresh', default=0.5, type=float, help='Evaluation threshold')
parser.add_argument('--conf_thresh', default=0.01, type=float, help='Confidence threshold for evaluation')
parser.add_argument('--nms_thresh', default=0.45, type=float, help='NMS threshold')
parser.add_argument('--default_mult', default=1.0, type=float, help='NMS threshold')
parser.add_argument('--topk', default=50, type=int, help='topk for evaluation')
parser.add_argument('--man_seed', default=123, type=int, help='manula seed')
args = parser.parse_args()


import socket
hostname = socket.gethostname()

if hostname == 'mars':
    args.data_root = '/mnt/mars-fast/datasets/'
    args.save_root = '/mnt/mars-gamma/'
    args.vis_port = 8097
elif hostname in ['sun']:
    args.data_root = '/mnt/sun-gamma/'
    args.save_root = '/mnt/sun-gamma/'
    args.vis_port = 8096
elif hostname == 'mercury':
    args.data_root = '/mnt/mercury-fast/datasets/'
    args.save_root = '/mnt/mercury-beta/'
    args.vis_port = 8098
else:
    args.data_root = '/home/gurkirt/datasets/'
    args.save_root = '/home/gurkirt/cache/'
    # args.vis_port = 8098
    visdom=False
# python train.py --seq_len=2 --num_workers=4 --batch_size=16 --ngpu=2 --fusion_type=NONE --input_type_base=brox --input_frames_base=5 --stepvalues=30000,50000 --max_iter=60000 --val_step=10000 --lr=0.001 

torch.set_default_tensor_type('torch.FloatTensor')
np.random.seed(args.man_seed)
torch.manual_seed(args.man_seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.man_seed)

def print_node(gdf):
    node_fns = gdf.next_functions
    for fn in node_fns:
        print(fn)
        print_node(fn[0][0])
    

def main():
    args.cfg = v2
    args.train_sets = 'train'
    args.test_sets = 'test'
    kd = 3
    args.means = (104, 117, 123)  
    num_classes = len(CLASSES[args.dataset]) + 1 # only support multiclass datasets, not multilabel
    args.num_classes = num_classes
    args.stepvalues = [int(val) for val in args.stepvalues.split(',')]
    args.loss_reset_step = 30
    # args.val_step = 30000
    args.print_step = 10
    args.fusion_type = args.fusion_type.upper()
    args.fusion = args.fusion_type in ['SUM','CAT','MEAN']
    ## Define the experiment Name will used for save directory and ENV for visdom
    if not args.fusion:
        args.exp_name = 'AMTNet-{}-s{:d}-{}-sl{:02d}sg{:02d}-bs{:02d}-lr{:05d}'.format(args.dataset, args.train_split,
                                                                                args.input_type_base.upper(),
                                                                                args.seq_len, args.seq_gap, 
                                                                                args.batch_size, int(args.lr * 100000))
    else:
        args.exp_name = 'AMTNet-{}-s{:d}-{}-{}-{}-sl{:02d}sg{:02d}-bs{:02d}-lr{:05d}'.format(args.dataset, args.train_split,
                                                                                args.fusion_type, args.input_type_base,
                                                                                args.input_type_extra,
                                                                                args.seq_len, args.seq_gap, 
                                                                                args.batch_size,int(args.lr * 100000))

    
    

    num_feat_multiplier = {'CAT': 2, 'SUM': 1, 'MEAN': 1, 'NONE': 1}
    # fusion type can one of the above keys
    args.fmd = [512, 1024, 512, 256, 256, 256]
    args.kd = 3
    args.fusion_num_muliplier = num_feat_multiplier[args.fusion_type]

    ## DEFINE THE NETWORK
    net = AMTNet(args)

    
    if args.fusion:
        base_weights = torch.load(args.data_root +'/weights/AMTNet_single_stream_{}_s{}.pth'.format(args.input_type_base, args.train_split))
        extra_weights = torch.load(args.data_root + '/weights/AMTNet_single_stream_{}_s{}.pth'.format(args.input_type_extra, args.train_split))
        print('Loading base network...')
        net.core_base.load_my_state_dict(base_weights, input_frames=args.input_frames_base)
        net.core_extra.load_my_state_dict(extra_weights, input_frames=args.input_frames_extra)
    else:
        base_weights = torch.load(args.data_root +'/weights/vgg_ucf24_{}_s{}.pth'.format(args.input_type_base, args.train_split))
        net.core_base.load_my_state_dict(base_weights, input_frames=args.input_frames_base)
    
    args.data_root += args.dataset + '/'
    args.save_root += args.dataset + '/'

    net = net.cuda()

    def xavier(param):
        init.xavier_uniform(param)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            xavier(m.weight.data)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            xavier(m.weight.data)
            m.bias.data.zero_()

    print('Initializing weights for HEADs...')
    net.loc.apply(weights_init)
    net.conf.apply(weights_init)

    args.save_root = args.save_root + 'cache/' + args.exp_name + '/'
    if not os.path.isdir(args.save_root):
        os.makedirs(args.save_root)

    if args.ngpu>1:
        print('\nLets do dataparallel\n\n')
        net = torch.nn.DataParallel(net)

    parameter_dict = dict(net.named_parameters()) # Get parmeter of network in dictionary format wtih name being key
    params = []

    #Set different learning rate to bias layers and set their weight_decay to 0
    mult = 1; decay = 0

    for name, param in parameter_dict.items():
        if name.find('bias') > -1:
            mult = 2.0; decay = 0
        else:
            mult = 1.0;  decay = 1
        if name.find('vgg')> -1 or name.find('extra')>-1 or name.find('L2Norm')>-1:
            mult = mult/args.seq_len

        # print(name, 'layer parameters will be trained @ {}'.format(args.lr*mult))
        params += [{'params':[param], 'lr': args.lr*mult, 'weight_decay':args.weight_decay*decay}]

    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = MultiboxLoss()

    scheduler = MultiStepLR(optimizer, milestones=args.stepvalues, gamma=args.gamma)
    # Get proior or anchor boxes
    with torch.no_grad():
        priorbox = PriorBox(v2, args.seq_len)
        priors = priorbox.forward()
    train(args, net, priors, optimizer, criterion, scheduler)


def train(args, net, priors, optimizer, criterion, scheduler):
    log_file = open(args.save_root+"training.log", "w", 1)
    log_file.write(args.exp_name+'\n')
    for arg in sorted(vars(args)):
        print(arg, getattr(args, arg))
        log_file.write(str(arg)+': '+str(getattr(args, arg))+'\n')

    net.train()
    # loss counters
    batch_time = AverageMeter()
    losses = AverageMeter()
    loc_losses = AverageMeter()
    cls_losses = AverageMeter()

    print('Loading Dataset...')
    train_dataset = ActionDetection(args, args.train_sets, SSDAugmentation(args.ssd_dim, args.means),
                                    NormliseBoxes(), anno_transform=MatchPrior(priors, args.cfg['variance']))
    log_file.write(train_dataset.print_str)
    print(train_dataset.print_str)
    val_dataset = ActionDetection(args, args.test_sets, BaseTransform(args.ssd_dim, args.means),
                                  NormliseBoxes(), full_test=False)
    log_file.write(val_dataset.print_str)
    # print(val_dataset.print_str)
    epoch_size = len(train_dataset) // args.batch_size

    print('Training SSD on', train_dataset.name)

    if args.visdom:

        import visdom
        viz = visdom.Visdom(env=args.exp_name, port=args.vis_port)
        # initialize visdom loss plot
        lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 6)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='Current SSD Training Loss',
                legend=['REG', 'CLS', 'AVG', 'S-REG', ' S-CLS', ' S-AVG']
            )
        )
        # initialize visdom meanAP and class APs plot
        legends = ['meanAP']
        for cls in CLASSES[args.dataset]:
            legends.append(cls)
        val_lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1,args.num_classes)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Mean AP',
                title='Current SSD Validation mean AP',
                legend=legends
            )
        )


    batch_iterator = None
    train_data_loader = data.DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
    val_data_loader = data.DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers,
                                 shuffle=False, collate_fn=detection_collate, pin_memory=True)
    itr_count = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for iteration in range(args.max_iter + 1):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(train_data_loader)

        # load train data
        images, _ , prior_gt_labels, prior_gt_locations, _, _ = next(batch_iterator)
        # images, ground_truths, _ , _, num_mt, img_indexs
        # pdb.set_trace()
        images = [img.cuda(0, non_blocking=True) for img in images if not isinstance(img, list)]
        prior_gt_labels = prior_gt_labels.cuda(0, non_blocking=True)
        prior_gt_locations = prior_gt_locations.cuda(0, non_blocking=True)
        # forward
        cls_out, reg_out = net(images)

        optimizer.zero_grad()
        loss_l, loss_c = criterion(cls_out, reg_out, prior_gt_labels, prior_gt_locations)
        loss = loss_l + loss_c

        loss.backward()
        optimizer.step()
        scheduler.step()

        # pdb.set_trace()
        loc_loss = loss_l.item()
        conf_loss = loss_c.item()
        # print('Loss data type ',type(loc_loss))
        loc_losses.update(loc_loss)
        cls_losses.update(conf_loss)
        losses.update((loc_loss + conf_loss)/2.0)

        if iteration == 103:
            loc_losses.reset()
            cls_losses.reset()
            losses.reset()
            batch_time.reset()

        if iteration % args.print_step == 0:
            if args.visdom and iteration>100:
                losses_list = [loc_losses.val, cls_losses.val, losses.val, loc_losses.avg, cls_losses.avg, losses.avg]
                viz.line(X=torch.ones((1, 6)).cpu() * iteration,
                    Y=torch.from_numpy(np.asarray(losses_list)).unsqueeze(0).cpu(),
                    win=lot,
                    update='append')


            torch.cuda.synchronize()
            t1 = time.perf_counter()
            batch_time.update(t1 - t0)

            print_line = 'Itration {:02d}/{:06d}/{:06d} loc-loss {:.3f}({:.3f}) cls-loss {:.3f}({:.3f}) ' \
                         'average-loss {:.3f}({:.3f}) Timer {:0.3f}({:0.3f})'.format(iteration//epoch_size,
                          iteration, args.max_iter, loc_losses.val, loc_losses.avg, cls_losses.val,
                          cls_losses.avg, losses.val, losses.avg, batch_time.val, batch_time.avg)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            log_file.write(print_line+'\n')
            print(print_line)

            itr_count += 1

            if itr_count % args.loss_reset_step == 0 and itr_count > 0:
                loc_losses.reset()
                cls_losses.reset()
                losses.reset()
                batch_time.reset()
                print('Reset ', args.exp_name,' after', itr_count*args.print_step)
                itr_count = 0


        if (iteration % args.val_step == 0 or iteration in [1000, args.max_iter]) and iteration>0:
            torch.cuda.synchronize()
            tvs = time.perf_counter()
            print('Saving state, iter:', iteration)
            torch.save(net.state_dict(), args.save_root + 'AMTNet_' +
                       repr(iteration) + '.pth')

            net.eval() # switch net to evaluation mode
            mAP, ap_all, ap_strs = validate(args, net, priors, val_data_loader, val_dataset, iteration, iou_thresh=args.iou_thresh)

            for ap_str in ap_strs:
                print(ap_str)
                log_file.write(ap_str+'\n')
            ptr_str = '\nMEANAP:::=>'+str(mAP)+'\n'
            print(ptr_str)
            log_file.write(ptr_str)

            if args.visdom:
                aps = [mAP]
                for ap in ap_all:
                    aps.append(ap)
                viz.line(
                    X=torch.ones((1, args.num_classes)).cpu() * iteration,
                    Y=torch.from_numpy(np.asarray(aps)).unsqueeze(0).cpu(),
                    win=val_lot,
                    update='append'
                        )
            net.train() # Switch net back to training mode
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            prt_str = '\nValidation TIME::: {:0.3f}\n\n'.format(t0-tvs)
            print(prt_str)
            log_file.write(ptr_str)

    log_file.close()


def validate(args, net, priors, val_data_loader, val_dataset, iteration_num, iou_thresh=0.5):
    """Test a SSD network on an image database."""
    print('Validating at ', iteration_num)
    num_images = len(val_dataset)
    num_classes = args.num_classes
    priors = priors.cuda()
    det_boxes = [[] for _ in range(len(CLASSES[args.dataset]))]
    gt_boxes = []
    print_time = True
    batch_iterator = None
    val_step = 100
    count = 0
    torch.cuda.synchronize()
    ts = time.perf_counter()
    softmax = nn.Softmax(dim=2).cuda()
    with torch.no_grad():
        for val_itr in range(len(val_data_loader)):
            if not batch_iterator:
                batch_iterator = iter(val_data_loader)

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            images, ground_truths, _ , _, num_mt, img_indexs = next(batch_iterator)
            batch_size = images[0].size(0)
            #images = images.permute(1, 0, 2, 3, 4)
            height, width = images[0].size(3), images[0].size(4)

            images = [img.cuda(0, non_blocking=True) for img in images if not isinstance(img, list)]

            conf_preds, loc_data = net(images)
            
            # pdb.set_trace()
            conf_scores_all = softmax(conf_preds).clone()
            

            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                tf = time.perf_counter()
                print('Forward Time {:0.3f}'.format(tf-t1))
            
            for b in range(batch_size):
                # pdb.set_trace()
                inds = np.asarray([m*args.seq_len for m in range(num_mt[b])])
                # pdb.set_trace()
                gt = ground_truths[b].numpy()
                gt = gt[inds]
                gt[:,0] *= width
                gt[:,2] *= width
                gt[:,1] *= height
                gt[:,3] *= height
                gt_boxes.append(gt)
                decoded_boxes = decode_seq(loc_data[b], priors, args.cfg['variance'], args.seq_len)
                decoded_boxes = decoded_boxes[:,:4].clone()
                conf_scores = conf_scores_all[b].cpu().clone()
                #Apply nms per class and obtain the results
                for cl_ind in range(1, num_classes):
                    pdb.set_trace()
                    scores = conf_scores[:, cl_ind].squeeze()
                    c_mask = scores.gt(args.conf_thresh)  # greater than minmum threshold
                    scores = scores[c_mask].squeeze() # reduce the dimension so if no element then # of dim is 0
                    if scores.dim() == 0:
                        det_boxes[cl_ind - 1].append(np.asarray([]))
                        continue
                    boxes = decoded_boxes.clone()
                    l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                    boxes = boxes[l_mask].view(-1, 4)
                    # idx of highest scoring and non-overlapping boxes per class
                    ids, counts = nms(boxes, scores, args.nms_thresh, args.topk)  # idsn - ids after nms
                    scores = scores[ids[:counts]].cpu().numpy()
                    boxes = boxes[ids[:counts]].cpu().numpy()
                    # print('boxes sahpe',boxes.shape)
                    boxes[:,0] *= width
                    boxes[:,2] *= width
                    boxes[:,1] *= height
                    boxes[:,3] *= height

                    for ik in range(boxes.shape[0]):
                        boxes[ik, 0] = max(0, boxes[ik, 0])
                        boxes[ik, 2] = min(width, boxes[ik, 2])
                        boxes[ik, 1] = max(0, boxes[ik, 1])
                        boxes[ik, 3] = min(height, boxes[ik, 3])
                    cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)
                    det_boxes[cl_ind-1].append(cls_dets)
                count += 1

            if val_itr%val_step == 0:
                torch.cuda.synchronize()
                te = time.perf_counter()
                print('im_detect: {:d}/{:d} time taken {:0.3f}'.format(count, num_images, te-ts))
                torch.cuda.synchronize()
                ts = time.perf_counter()
            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                te = time.perf_counter()
                print('NMS stuff Time {:0.3f}'.format(te - tf))
    print('Evaluating detections for itration number ', iteration_num)
    return evaluate_detections(gt_boxes, det_boxes, CLASSES[args.dataset], iou_thresh=iou_thresh)


if __name__ == '__main__':
    main()
