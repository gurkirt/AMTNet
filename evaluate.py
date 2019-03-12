"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Which was adopated by: Ellis Brown, Max deGroot
    https://github.com/amdegroot/ssd.pytorch

    Further:
    Updated by Gurkirt Singh for ucf101-24 dataset
    Licensed under The MIT License [see LICENSE for details]

"""

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from data import v2, ActionDetection, NormliseBoxes, detection_collate, CLASSES, BaseTransform
from layers.functions import PriorBox
from AMTNet import AMTNet
import torch.utils.data as data
from layers.box_utils import nms, decode_seq
from utils.evaluation import evaluate_detections
import os, time
import argparse
import numpy as np
import pickle
import scipy.io as sio # to save detection as mat files
cfg = v2


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--dataset', default='ucf24', help='pretrained base model')
parser.add_argument('--ssd_dim', default=300, type=int, help='Input Size for SSD') # only support 300 now
parser.add_argument('--seq_len', default=2, type=int, help='Input sequence length ')
parser.add_argument('--seq_gap', default=0, type=int, help='Gap between the frame of sequence')
parser.add_argument('--eval_gaps', default='1', type=str, help='Gap between the frame of sequence at evaluation time')
parser.add_argument('--train_split', default=1, type=int, help='Split id')
parser.add_argument('--fusion_type', default='cat', type=str, 
                    help='Fusion type to fuse from sequence of frames; options are SUM, CAT and NONE')
                    # 
parser.add_argument('--input_type_base', default='rgb', type=str, help='INput tyep default rgb can take flow (brox or fastOF) as well')
parser.add_argument('--input_type_extra', default='brox', type=str, help='INput tyep default brox can take flow (brox or fastOF) as well')
parser.add_argument('--input_frames_base', default=1, type=int, help='Number of input frame, default for rgb is 1')
parser.add_argument('--input_frames_extra', default=5, type=int, help='Number of input frame, default for flow is 5')

parser.add_argument('--eval_iters', default='5000', type=str, help='evaluation iterations type')
parser.add_argument('--input_frames', default=1, type=int, help='Number of input frame, default for rgb is 1')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--max_iter', default=150000, type=int, help='Number of training iterations')
parser.add_argument('--eval_iter', default=40000, type=int, help='Evaluate on iterations')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=2, type=int, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float, help='initial learning rate')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--data_root', default='/mnt/mars-fast/datasets/', help='Location of VOC root directory')
parser.add_argument('--save_root', default='/mnt/mars-alpha/', help='Location to save checkpoint models')
parser.add_argument('--iou_thresh', default=0.5, type=float, help='Evaluation threshold')
parser.add_argument('--conf_thresh', default=0.05, type=float, help='Confidence threshold for evaluation')
parser.add_argument('--nms_thresh', default=0.45, type=float, help='NMS threshold')
parser.add_argument('--topk', default=20, type=int, help='topk for evaluation')
parser.add_argument('--man_seed', default=123, type=int, help='manualseed for reproduction')
args = parser.parse_args()
import socket
hostname = socket.gethostname()

if hostname == 'mars':
    args.data_root = '/mnt/mars-fast/datasets/'
    args.save_root = '/mnt/mars-gamma/'
    args.vis_port = 8097
elif hostname == 'sun':
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
## set random seeds
np.random.seed(args.man_seed)
torch.manual_seed(args.man_seed)

if args.cuda:
    torch.cuda.manual_seed_all(args.man_seed)

torch.set_default_tensor_type('torch.FloatTensor')
# CLASSES = CLASSES[args.dataset]


def test_net(net, priors, args, dataset, iteration, thresh=0.5 ):
    """ Test a SSD network on an Action image database. """
    print('Test a SSD network on an Action image database')
    val_data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                            shuffle=False, collate_fn=detection_collate, pin_memory=True)
    print('Done making val dataset')
    image_ids = dataset.ids
    save_ids = []
    val_step = 250
    num_images = len(dataset)
    video_list = dataset.video_list
    det_boxes = [[] for _ in range(len(CLASSES[args.dataset]))]
    gt_boxes = []
    print_time = True
    batch_iterator = None
    count = 0
    torch.cuda.synchronize()
    ts = time.perf_counter()
    num_batches = len(val_data_loader)
    frame_save_dir = '{}detections/{:s}-eg{:02d}/'.format(args.save_root, args.exp_name, args.eval_gap)
    softmax = nn.Softmax(dim=2).cuda()
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
            print('Forward Time {:0.3f}'.format(tf - t1))
        for b in range(batch_size):
            inds = np.asarray([m * args.seq_len for m in range(num_mt[b])])
            gt = ground_truths[b].numpy()
            gt = gt[inds]
            gt[:, 0] *= width
            gt[:, 2] *= width
            gt[:, 1] *= height
            gt[:, 3] *= height
            gt_boxes.append(gt)
            bloc_data = loc_data[b]
            #print(bloc_data.size(), prior_data.size())
            decoded_boxes = decode_seq(bloc_data, priors, args.cfg['variance'], args.seq_len)
            decoded_boxes = decoded_boxes.cpu()
            conf_scores = conf_scores_all[b].cpu().clone()
            index = img_indexs[b]
            annot_info = image_ids[index]

            frame_num = annot_info[1][0]+1; video_id = annot_info[0]; videoname = video_list[video_id]
            output_dir = frame_save_dir+videoname
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            # for s in range(args.seq_len):
            output_file_name_tmp = output_dir + '/{:06d}.mat'.format(int(frame_num))
            # save_ids.append(output_file_name_tmp)
            decoded_boxes_tmp = decoded_boxes.numpy()
            #print(output_file_name_tmp)
            sio.savemat(output_file_name_tmp,
                    mdict={'scores': conf_scores.numpy(), 'loc': decoded_boxes_tmp})

            decoded_boxes = decoded_boxes[:, :4].clone()
            
            for cl_ind in range(1, args.num_classes):
                scores = conf_scores[:, cl_ind].squeeze()
                c_mask = scores.gt(args.conf_thresh)  # greater than minmum threshold
                scores = scores[c_mask].squeeze()
                # print('scores size',scores.size())
                if scores.dim() == 0:
                    # print(len(''), ' dim ==0 ')
                    det_boxes[cl_ind - 1].append(np.asarray([]))
                    continue
                boxes = decoded_boxes.clone()
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes = boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, counts = nms(boxes, scores, args.nms_thresh, args.topk)  # idsn - ids after nms
                scores = scores[ids[:counts]].numpy()
                boxes = boxes[ids[:counts]].numpy()
                # print('boxes sahpe',boxes.shape)
                boxes[:, 0] *= width
                boxes[:, 2] *= width
                boxes[:, 1] *= height
                boxes[:, 3] *= height

                for ik in range(boxes.shape[0]):
                    boxes[ik, 0] = max(0, boxes[ik, 0])
                    boxes[ik, 2] = min(width, boxes[ik, 2])
                    boxes[ik, 1] = max(0, boxes[ik, 1])
                    boxes[ik, 3] = min(height, boxes[ik, 3])

                cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)
                det_boxes[cl_ind - 1].append(cls_dets)

            count += 1
        if val_itr%val_step == 0:
            torch.cuda.synchronize()
            te = time.perf_counter()
            print('im_detect: {:d}/{:d} time taken {:0.3f}'.format(count, num_images, te - ts))
            torch.cuda.synchronize()
            ts = time.perf_counter()
        if print_time and val_itr%val_step == 0:
            torch.cuda.synchronize()
            te = time.perf_counter()
            print('NMS stuff Time {:0.3f}'.format(te - tf))
    print('Evaluating detections for itration number ', iteration)

    #Save detection after NMS along with GT
    # with open(det_file, 'wb') as f:
    #     pickle.dump([gt_boxes, det_boxes, save_ids], f, pickle.HIGHEST_PROTOCOL)
    # if args.dataset != 'daly00000000000000000000000000':
    #     return 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '\n\n\n AP is not COMPUTED for any of the classes in dataset \n\n\n'
    # else:
    return evaluate_detections(gt_boxes, det_boxes, CLASSES[args.dataset], iou_thresh=thresh)


def main():

    means = (104, 117, 123)  # only support voc now
    args.save_root += args.dataset + '/'
    args.data_root += args.dataset + '/'
    for eval_gap in [int(g) for g in args.eval_gaps.split(',')]:
        args.eval_gap = eval_gap
        

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
        print(args.exp_name, ' eg::=> ', eval_gap)
    

        args.cfg = v2
        args.num_classes = len(CLASSES[args.dataset]) + 1  # 7 +1 background
        
        # Get proior or anchor boxes
        with torch.no_grad():
            priorbox = PriorBox(v2, args.seq_len)
            priors = priorbox.forward()
            priors = priors.cuda()
            num_feat_multiplier = {'CAT': 2, 'SUM': 1, 'MEAN': 1, 'NONE': 1}
            # fusion type can one of the above keys
            args.fmd = [512, 1024, 512, 256, 256, 256]
            args.kd = 3
            args.fusion_num_muliplier = num_feat_multiplier[args.fusion_type]

            dataset = ActionDetection(args, 'test', BaseTransform(args.ssd_dim, means), NormliseBoxes(), full_test=False)

            ## DEFINE THE NETWORK
            net = AMTNet(args)
            if args.ngpu>1:
                print('\nLets do dataparallel\n\n')
                net = torch.nn.DataParallel(net)
        
                # Load dataset

            for iteration in [int(it) for it in args.eval_iters.split(',')]:
                fname = args.save_root + 'cache/' + args.exp_name + "/testing-{:d}-eg{:d}.log".format(iteration, eval_gap)
                log_file = open(fname, "w", 1)
                log_file.write(args.exp_name + '\n')
                print(fname)
                trained_model_path = args.save_root + 'cache/' + args.exp_name + '/AMTNet_' + repr(iteration) + '.pth'
                log_file.write(trained_model_path+'\n')
                # trained_model_path = '/mnt/sun-alpha/ss-workspace/CVPR2018_WORK/ssd.pytorch_exp/UCF24/guru_ssd_pipeline_weights/ssd300_ucf24_90000.pth'

                net.load_state_dict(torch.load(trained_model_path))
                print('Finished loading model %d !' % iteration)
                net.eval()
                net = net.cuda()
                
                # evaluation
                torch.cuda.synchronize()
                tt0 = time.perf_counter()
                log_file.write('Testing net \n')
                
                mAP, ap_all, ap_strs = test_net(net, priors, args, dataset, iteration)
                for ap_str in ap_strs:
                    print(ap_str)
                    log_file.write(ap_str + '\n')
                ptr_str = '\nMEANAP:::=>' + str(mAP) + '\n'
                print(ptr_str)
                log_file.write(ptr_str)
                torch.cuda.synchronize()
                print('Complete set time {:0.2f}'.format(time.perf_counter() - tt0))
                log_file.close()


if __name__ == '__main__':
    main()