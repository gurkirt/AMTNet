"""UCF24 Dataset Classes

Original author: Francisco Massa for VOC dataset
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
Updated by: Ellis Brown, Max deGroot for VOC dataset

Updated by Gurkirt Singh for ucf101-24 dataset
"""

import os, pdb
import os.path
import torch,json
import torch.utils.data as data
import cv2, pickle
import numpy as np

np.random.seed(123)

CLASSES = dict()
CLASSES['ucf24'] = ['Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving', 'Fencing',
                    'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing',
                    'SalsaSpin','SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling',
                    'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog']

CLASSES['daly'] = ['ApplyingMakeUpOnLips', 'BrushingTeeth', 'CleaningFloor', 'CleaningWindows', 'Drinking',
                   'FoldingTextile', 'Ironing', 'Phoning', 'PlayingHarmonica', 'TakingPhotosOrVideos']

CLASSES['jhmdb21'] = ['brush_hair','catch','clap','climb_stairs','golf','jump','kick_ball',
                      'pick','pour','pullup','push','run','shoot_ball','shoot_bow',
                      'shoot_gun','sit','stand','swing_baseball','throw','walk','wave']



def make_lists_jhmdb(rootpath, imgtype, seq_len=2, seq_gap=0, split=1, fulltest=False):

    print(fulltest)
    dataset = 'jhmdb21'
    imagesDir = rootpath + imgtype + '/'
    splitfile = rootpath + 'splitfiles/trainlist{:02d}.txt'.format(split)
    trainvideos = readsplitfile(splitfile)
    trainlist = []
    testlist = []

    with open(rootpath + 'splitfiles/pyannot.pkl','rb') as fff:
        database = pickle.load(fff)

    train_action_counts = np.zeros(len(CLASSES[dataset]), dtype=np.int32)
    test_action_counts = np.zeros(len(CLASSES[dataset]), dtype=np.int32)

    ratios = [1.0, 1.1, 1.1, 0.9, 1.1, 0.8, 0.7, 0.8, 1.1, 1.4, 1.0, 0.8, 0.7, 1.2, 1.0, 0.8, 0.7, 1.2, 1.2, 1.0, 0.9]
    # ratios = np.ones(len(CLASSES[dataset]))
    video_list = []
    numf_list = []
    for vid, videoname in enumerate(sorted(database.keys())):
        video_list.append(videoname)
        actidx = database[videoname]['label']
        istrain = True
        step = ratios[actidx]
        if videoname not in trainvideos:
            istrain = False
            step = max(1, ratios[actidx]) * 3

        if fulltest:
            step = (seq_len - 1) * (seq_gap + 1)

        # print('step ', step)
        annotations = database[videoname]['annotations']
        numf = annotations[0]['boxes'].shape[0] #database[videoname]['numf']
        numf_list.append(numf)
        num_tubes = len(annotations)

        possible_frame_nums = np.arange(0, numf - ((seq_len - 1) * seq_gap + seq_len), step)

        if numf-step-1 not in possible_frame_nums and fulltest:
            possible_frame_nums = np.append(possible_frame_nums, numf-step-1)
        # print('numf',numf,possible_frame_nums[-1])
        for frame_id, frame_num in enumerate(possible_frame_nums): # loop from start to last possible frame which can make a legit sequence
            frame_num = np.int(frame_num)
            sq_frames = np.arange(frame_num, frame_num + (seq_len - 1) * (seq_gap + 1) + 1, seq_gap + 1,
                                  dtype=np.int16)  # make a sequence starting from current frame_num

            # print(sq_frames,numf,annotations[0]['boxes'].shape, videoname)
            all_boxes = []
            labels = []
            image_name = imagesDir + videoname + '/{:05d}.jpg'.format(frame_num + 1)
            assert os.path.isfile(image_name), 'Image does not exist' + image_name
            for tubeid, tube in enumerate(annotations):
                boxes = []
                for l in sq_frames:
                    boxes.append(tube['boxes'][int(l)])
                boxes = np.asarray(boxes)
                boxes[:,2] += boxes[:,0]  # convert width to xmax
                boxes[:,3] += boxes[:,1]
                all_boxes.append(boxes)
                labels.append(tube['label'])

                if istrain: # if it is training video
                    trainlist.append([vid, sq_frames, labels, all_boxes])
                    train_action_counts[actidx] += 1 #len(labels)
                else: # if test video and has micro-tubes with GT
                    testlist.append([vid, sq_frames, labels, all_boxes])
                    test_action_counts[actidx] += 1 #len(labels)

    print_str = ''
    for actidx, act_count in enumerate(train_action_counts): # just to see the distribution of train and test sets
        tmp_str = 'train {:05d} test {:05d} action {:02d} {:s}'.format(act_count, test_action_counts[actidx] , int(actidx), CLASSES[dataset][actidx])
        # print(tmp_str)
        print_str += tmp_str+'\n'

    tmp_str = 'Trainlistlen ' + str(len(trainlist)) + ' testlist ' + str(len(testlist))
    # print(tmp_str)
    print_str += tmp_str + '\n'

    return trainlist, testlist, video_list, numf_list, print_str


def make_lists_ucf(rootpath, imgtype, seq_len=2, seq_gap=0, split=1, fulltest=False):
    print(fulltest)
    imagesDir = rootpath + imgtype + '/'
    splitfile = rootpath + 'splitfiles/trainlist{:02d}.txt'.format(split)
    trainvideos = readsplitfile(splitfile)
    trainlist = []
    testlist = []

    with open(rootpath + 'splitfiles/pyannot.pkl','rb') as fff:
        database = pickle.load(fff)

    train_action_counts = np.zeros(len(CLASSES['ucf24']), dtype=np.int32)
    test_action_counts = np.zeros(len(CLASSES['ucf24']), dtype=np.int32)

    #ratios = np.asarray([1.1,0.8,4.7,1.4,0.9,2.6,2.2,3.0,3.0,5.0,6.2,2.7,3.5,3.1,4.3,2.5,4.5,3.4,6.7,3.6,1.6,3.4,0.6,4.3])
    all_ratios = dict()
    all_ratios['2'] = np.asarray([1.00, 0.80, 4.19, 1.30, 0.85, 2.34, 1.98, 2.64, 2.66, 4.49, 5.53, 2.44,
                         3.15, 2.74, 3.77, 2.26, 4.00, 3.06, 6.04, 3.25, 1.47, 3.02, 0.70, 3.82]) ## 5000 as base number

    all_ratios['6'] = np.asarray([1.11, 0.85, 5.10, 1.53, 1.0, 2.82, 2.38, 3.21, 3.21, 5.49, 6.78, 2.95,
                                  3.82, 3.34, 4.16, 2.74, 4.90, 3.75, 7.44, 3.92, 1.64, 3.61, 0.76, 4.68]) ##4000 as base number
    ratios = all_ratios[str(seq_len)]
    #ratios = np.ones_like(ratios)
    video_list = []
    numf_list = []
    for vid, videoname in enumerate(sorted(database.keys())):
        video_list.append(videoname)
        actidx = database[videoname]['label']

        istrain = True
        step = ratios[actidx]

        if videoname not in trainvideos:
            istrain = False
            step = max(1, ratios[actidx]) * 3

        if fulltest:
            step = (seq_len - 1) * (seq_gap + 1)

        # print('step ', step)
        annotations = database[videoname]['annotations']
        numf = database[videoname]['numf']
        numf_list.append(numf)
        num_tubes = len(annotations)

        tube_labels = np.zeros((numf,num_tubes),dtype=np.int16) # check for each tube if present in
        tube_boxes = [[[] for _ in range(num_tubes)] for _ in range(numf)]
        for tubeid, tube in enumerate(annotations):
            # print('numf00', numf, tube['sf'], tube['ef'])
            for frame_id, frame_num in enumerate(np.arange(tube['sf'], tube['ef'], 1)): # start of the tube to end frame of the tube
                label = tube['label']
                assert actidx == label, 'Tube label and video label should be same'
                box = tube['boxes'][frame_id, :]  # get the box as an array
                box = box.astype(np.float32)
                box -= 1  #[Suman] I am doing this in  AnnotationTransform()
                box[2] += box[0]  #convert width to xmax
                box[3] += box[1]  #converst height to ymax
                tube_labels[frame_num, tubeid] = 1  # change label in tube_labels matrix to 1 form 0
                tube_boxes[frame_num][tubeid] = box  # put the box in matrix of lists

        possible_frame_nums = np.arange(0, numf - ((seq_len - 1) * seq_gap + seq_len), step)

        if numf-step-1 not in possible_frame_nums and fulltest:
            possible_frame_nums = np.append(possible_frame_nums, numf-step-1)
        # print('numf',numf,possible_frame_nums[-1])
        for frame_id, frame_num in enumerate(possible_frame_nums): # loop from start to last possible frame which can make a legit sequence
            frame_num = np.int(frame_num)
            sq_frames = np.arange(frame_num, frame_num + (seq_len - 1) * (seq_gap + 1) + 1, seq_gap + 1,
                                  dtype=np.int16)  # make a sequence starting from current frame_num
            check_tubes = np.zeros(num_tubes)
            tl_counts = np.zeros(num_tubes)
            # BELOW: Start to check if there a sequence of frame from same tube exsits
            for tubeid, tube in enumerate(annotations):
                tl_count = 0
                for l in sq_frames:
                    tl_count += tube_labels[int(l), tubeid]
                if tl_count % seq_len == 0:  # either tube should be present for all frames in sequence or none
                    check_tubes[tubeid] = 1
                tl_counts[tubeid] = tl_count


            #check_tubes = tube_labels[frame_num,:]
            # np.sum(tl_counts == sq_len)>2 -- how many micro tubes it should retrun
            if np.sum(check_tubes) == num_tubes and np.sum(tl_counts == seq_len)>0:  # check if there aren't any semi overlapping tubes
                all_boxes = []
                labels = []
                image_name = imagesDir + videoname + '/{:05d}.jpg'.format(frame_num + 1)
                assert os.path.isfile(image_name), 'Image does not exist' + image_name
                for tubeid, tube in enumerate(annotations):
                    if tl_counts[tubeid] == seq_len:
                        boxes = []
                        for l in sq_frames:
                            boxes.append(tube_boxes[int(l)][tubeid])
                        boxes = np.asarray(boxes)
                        all_boxes.append(boxes)
                        labels.append(tube['label'])

                if istrain: # if it is training video
                    trainlist.append([vid, sq_frames, labels, all_boxes])
                    train_action_counts[actidx] += 1 #len(labels)
                else: # if test video and has micro-tubes with GT
                    testlist.append([vid, sq_frames, labels, all_boxes])
                    test_action_counts[actidx] += 1 #len(labels)
            elif fulltest:
                if istrain: # if test video with no ground truth and fulltest is trues
                    trainlist.append([vid, sq_frames, [9999], [np.zeros((seq_len, 4))]])
                else:
                    testlist.append([vid, sq_frames, [9999], [np.zeros((seq_len, 4))]])

    print_str = ''
    for actidx, act_count in enumerate(train_action_counts):  # just to see the distribution of train and test sets
        tmp_str = 'train {:05d} test {:05d} action {:02d} {:s}'.format(act_count, test_action_counts[actidx],
                                                                       int(actidx), CLASSES['ucf24'][actidx])
        # print(tmp_str)
        print_str += tmp_str + '\n'

    tmp_str = 'Trainlistlen ' + str(len(trainlist)) + ' testlist ' + str(len(testlist))
    # print(tmp_str)
    print_str += tmp_str + '\n'

    newratios = train_action_counts / 4000
    line = '['
    for r in newratios:
        line += '{:0.2f}, '.format(r)
    print(line + ']')


    return trainlist, testlist, video_list, numf_list, print_str


def make_lists_daly(rootpath, bg_step=40, use_bg=False, fulltest=False, seq_gap=5):

    if fulltest:
        offset = seq_gap
    else:
        offset = 5
    print('root::{} bg {} fulltest{}'.format(rootpath,use_bg,fulltest))
    with open(rootpath + 'splitfiles/finalAnnots.json','r') as f:
        finalAnnot = json.load(f)
    db = finalAnnot['annots']
    testvideos = finalAnnot['testvideos']
    vids = finalAnnot['vidList']
    # pdb.set_trace()
    train_action_counts = np.zeros(len(CLASSES['daly']), dtype=np.int32)
    test_action_counts = np.zeros(len(CLASSES['daly']), dtype=np.int32)


    video_list = []
    numf_list = []
    trainlist = []
    testlist = []
    count = 0
    for vid, videoname in enumerate(vids):
        istrain = videoname not in testvideos
        vid_info = db[videoname]
        numf = vid_info['numf']
        numf_list.append(numf)
        video_list.append(videoname)
        tubes = vid_info['annotations']
        keyframes = dict()
        frame_labels = np.zeros(numf, dtype=np.int8)  # check for each tube if present in

        step = bg_step
        if not istrain:
            step = bg_step*2
        if fulltest:
            step = offset + 1

        keyframe_pairs = dict()

        for tid, tube in enumerate(tubes):
            frame_labels[max(0, tube['sf'] - offset): min(numf, tube['ef'] + offset)] = 1
            if len(tube['frames']) == 1:
                fn = tube['frames'][0]
                fn2 = fn+1
                if fn2>=numf:
                    fn2 = fn-1
                    print('this is ugly check annot for ', fn, numf, videoname, 'class ', CLASSES['daly'][tube['class']])
                if not str(fn) in keyframe_pairs.keys():
                    keyframe_pairs[str(fn)] =  {'boxes':[np.asarray([tube['bboxes'][0],tube['bboxes'][0]])],
                                                'labels':[tube['class']],
                                                'frns':[fn, fn2]}

                else:
                    keyframe_pairs[str(fn)]['boxes'].append(np.asarray([tube['bboxes'][0],tube['bboxes'][0]]))
                    keyframe_pairs[str(fn)]['labels'].append(tube['class'])

                assert keyframe_pairs[str(fn)]['frns'][1]<numf, ' {} {}'.format(fn, numf)
                    # keyframe_pairs[str(fn)]['frns'].append([fn+1, fn+1])
            else:
               for fid in range(len(tube['frames'])-1):
                   fn1 = tube['frames'][fid]
                   fn2 = tube['frames'][fid+1]
                   if not str(fn1) in keyframe_pairs.keys():
                       keyframe_pairs[str(fn1)] = {'boxes': [np.asarray([tube['bboxes'][fid], tube['bboxes'][fid+1]])],
                                                  'labels': [tube['class']],
                                                  'frns': [fn1, fn2]}
                   else:
                        keyframe_pairs[str(fn1)]['boxes'].append(np.asarray([tube['bboxes'][fid], tube['bboxes'][fid+1]]))
                        keyframe_pairs[str(fn1)]['labels'].append(tube['class'])
                        pfn2 = keyframe_pairs[str(fn1)]['frns'][1]

                        assert abs(pfn2 - fn2-1)<=2, 'second frame has a problem'
                   assert keyframe_pairs[str(fn1)]['frns'][1] < numf

        possible_frames = [fn for fn in range(0, numf-step-1, step)]
        # print(possible_frames[-1])

        if fulltest:
            if (numf - step - 1) not in possible_frames:
                possible_frames.append(numf - step - 1)
        # print(possible_frames[-5:])

        if not fulltest:
            for fn in keyframe_pairs.keys():
                if int(fn) not in possible_frames:
                    possible_frames.append(int(fn))

        for fn in possible_frames:
            gt_frame = False
            if str(fn) in keyframe_pairs.keys() and not fulltest:
                labels = keyframe_pairs[str(fn)]['labels']
                boxes = keyframe_pairs[str(fn)]['boxes']
                sq_frames = np.asarray(keyframe_pairs[str(fn)]['frns'])
                gt_frame = True
            else:
                boxes = [np.zeros((2, 4))]
                sq_frames = np.asarray([fn, min(numf-1, fn + step)])
                labels = [9999]

            assert sq_frames[0] != sq_frames[1] and sq_frames[1]<numf, 'sq_frames {}, {}'.format(sq_frames[0], sq_frames[1])

            if gt_frame:
                if istrain:
                    trainlist.append([vid, sq_frames, labels, boxes])
                    for label in labels:
                        train_action_counts[label] += 1  # len(labels)
                else:
                    testlist.append([vid, sq_frames, labels, boxes])
                    for label in labels:
                        test_action_counts[label] += 1  # len(labels)
            elif fulltest or (use_bg and frame_labels[fn] == 0):
                if istrain:
                    trainlist.append([vid, sq_frames, labels, boxes])
                else:
                    testlist.append([vid, sq_frames, labels, boxes])

    print_str = ''
    for actidx, act_count in enumerate(train_action_counts): # just to see the distribution of train and test sets
        tmp_str = 'train {:05d} test {:05d} action {:02d} {:s}'.format(act_count, test_action_counts[actidx] , int(actidx), CLASSES['daly'][actidx])
        print(tmp_str)
        print_str += tmp_str + '\n'

    tmp_str = 'Trainlistlen {} train count {} testlist {} test count {} \n total keyframes with labels {}'.format(len(trainlist),
                np.sum(train_action_counts), len(testlist), np.sum(test_action_counts), count)

    print(tmp_str)
    print_str += tmp_str + '\n'

    return trainlist, testlist, video_list, numf_list, print_str

def make_lists(dataset, rootpath, imgtype, seq_len=2, seq_gap=0, split=1, fulltest=False):
    if dataset.find('ucf24')>-1:
        return make_lists_ucf(rootpath, imgtype, seq_len=seq_len, seq_gap=seq_gap, split=split, fulltest=fulltest)
    elif dataset == 'daly':
        return make_lists_daly(rootpath, fulltest=fulltest, seq_gap=seq_gap)
    else:
        return make_lists_jhmdb(rootpath, imgtype, seq_len=seq_len, seq_gap=seq_gap, split=split, fulltest=fulltest)

class ActionDetection(data.Dataset):

    def __init__(self, args, image_set, transform=None, target_transform=None, split=1, full_test=False):

        self.seq_len = args.seq_len
        self.seq_gap = args.seq_gap
        self.dataset =args.dataset
        if full_test:
            seq_gap = args.eval_gap
        else:
            seq_gap = args.seq_gap
        self.input_frames = args.input_frames
        
        self.input_type = args.input_type+'-images'
        self.root = args.data_root
        self.CLASSES = CLASSES[args.dataset]
        self.num_classes = len(CLASSES)
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.name = args.dataset
        self._annopath = os.path.join(self.root, 'labels/', '%s.txt')
        self._imgpath = os.path.join(self.root, self.input_type)
        self.ids = list()

        trainlist, testlist, video_list, numf_list, self.print_str = make_lists(args.dataset, self.root, self.input_type, seq_len=self.seq_len,
                                                     seq_gap=seq_gap, split=split, fulltest=full_test)
        self.video_list = video_list
        self.numf_list = numf_list
        if self.image_set == 'train':
            self.ids = trainlist
        elif self.image_set == 'test':
            self.ids = testlist
        else:
            print('spacify correct subset ')

    def __getitem__(self, index):
        im, gt, nmt, img_index = self.pull_item(index)

        return im, gt, nmt, img_index

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        # index = index + 1600
        annot_info = self.ids[index]
        actVidName = self.video_list[annot_info[0]]
        frm_nos = annot_info[1] + 1
        # print(frm_nos,self.seq_len)
        assert (len(frm_nos) == self.seq_len)
        labels = annot_info[2]
        gtbxs = annot_info[3]  # boxes are in xmin ymin width and height format
        # print(gtbxs)
        num_mt = len(labels)
        numf = self.numf_list[annot_info[0]]
        num_input_frames = self.input_frames

        all_frames_ids =[]
        first_index = []
        count = 0

        # print(frm_nos, actVidName)
        for fn in frm_nos:
            if numf+1 <= fn + num_input_frames//2 + 1:
                ef = min(numf+1, fn + num_input_frames//2 + 1)
                sf = ef-num_input_frames
            else:
                sf = max(fn - num_input_frames//2, 1)
                ef = sf+num_input_frames
            frames_ids = np.arange(sf,ef)
            c = 0
            for f in frames_ids:
                if f not in all_frames_ids:
                    all_frames_ids.append(f)
                    c+=1
            if count == 0:
                first_index.append(0)
            else:
                first_index.append(c)
                first_index[count] += first_index[count-1]
            count += 1

        # print(all_frames_ids, len(all_frames_ids))
        # if len(all_frames_ids) == 1 and self.dataset == 'daly':
        #      all_frames_ids.append(all_frames_ids[0])
        #print('num to load ',len(all_frames_ids))

        img_path = os.path.join(self._imgpath, actVidName)
        # print(all_frames_ids)
        imgs = []

        for fn in all_frames_ids:
            img_file = '{:s}/{:05d}.jpg'.format(img_path, int(fn))
            #print(img_file)
            img = cv2.imread(img_file)
            height, width, _ = img.shape
            imgs.append(img)

        imgs = np.asarray(imgs)
        #print('imgs shape ', imgs.shape)
        if self.dataset in ['ucf24','jhmdb21'] or self.dataset.find('ucf24')>-1:
            boxes_norm = self.target_transform(gtbxs, width, height, labels, num_mt, self.seq_len)
        else:
            boxes_norm = self.target_transform(gtbxs, 1.0, 1.0, labels, num_mt, self.seq_len)
                                           # normaized gt boxes --->      [xmin ymin xmax ymax label]
        boxes_norm = np.array(boxes_norm)  # converting from list numpy array
        # pdb.set_trace()
        # print(boxes_norm)
        if self.image_set == 'train':
            aug_imgs, aug_bxs, labels = self.transform(imgs, boxes_norm[:, :4], boxes_norm[:, 4], self.seq_len,
                                                           num_mt)  # calling SSDAugmentation
        else:
            aug_imgs, aug_bxs, labels = self.transform(imgs, boxes_norm[:, :4], boxes_norm[:, -1])  # calling BaseTransform

        num_bxs = aug_bxs.shape[0]
        # number of micro tubes after augmentation -- recall after augmentation some micro tubes may be discarded
        # so don't confuse with num_mta and num_mt they are different
        num_mtaa = int(num_bxs / self.seq_len)  # num_mtaa - num micro tube after augmentation


        # aug_imgs is in [seq_len x H x W x C] (0,1,2,3) ---> so converting from RGB (0,1,2) to BGR along 4-th dim
        aug_imgs = aug_imgs[:, :, :, (2, 1, 0)]
        # print('NUm of frame loaded and and required ', aug_imgs.shape[0], ' ', num_input_frames)
        if num_input_frames > 1:
            images = []
            for s in range(self.seq_len):
                sf = first_index[s]
                #print(sf)
                img_stack = aug_imgs[sf:sf+num_input_frames,:,:,:]
                img_stack = torch.from_numpy(img_stack).permute(0, 3, 1, 2).contiguous()
                images.append(img_stack.view(-1, img_stack.size(2), img_stack.size(3)))
                #print(images[s].size())
            images = torch.stack(images, 0)
        else:
            images = torch.from_numpy(aug_imgs).permute(0, 3, 1, 2)
        # stacking the boxes and the labels
        aug_bxsl = np.hstack((aug_bxs, np.expand_dims(labels, axis=1)))
        # converting from [seq_len x H x W x C] (0,1,2,3) -->  [seq_len x C x H x W] (0,3,1,2)
        # pdb.set_trace()
        return images, aug_bxsl, num_mtaa, index


def detection_collate(batch):
    targets = []
    imgs = []
    num_mt = []
    image_ids = []
    # fno = []

    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        num_mt.append(sample[2])
        image_ids.append(sample[3])

    return torch.stack(imgs, 0), targets, num_mt, image_ids
