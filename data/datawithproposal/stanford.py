
import os.path as osp
import xml.etree.ElementTree as ET
import os.path as osp

import mmcv

from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from data.datawithproposal.transforms import (ImageTransform, BboxTransform, MaskTransform,
                         Numpy2Tensor)
from data.datasets.utils import to_tensor, random_scale
from data.datasets.extra_aug import ExtraAugmentation

import mmcv
import numpy as np
from torch.utils.data import DataLoader,Dataset
CLASSES = ('applauding','blowing_bubbles','brushing_teeth','cleaning_the_floor',
               'climbing','cooking','cutting_trees','cutting_vegetables','drinking',
               'feeding_a_horse','fishing','fixing_a_bike','fixing_a_car','gardening',
               'holding_an_umbrella','jumping','looking_through_a_microscope','looking_through_a_telescope',
               'playing_guitar','playing_violin','pouring_liquid','pushing_a_cart','reading',
               'phoning','riding_a_bike','riding_a_horse','rowing_a_boat','running',
               'shooting_an_arrow','smoking','taking_photos','texting_message',
               'throwing_frisby','using_a_computer','walking_the_dog','washing_dishes',
               'watching_TV','waving_hands','writing_on_a_board','writing_on_a_book')
class sDataset(Dataset):
    def __init__(self,ann_file,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 size_divisor=None,
                 proposal_file=None,
                 test_mode=False,):# 只有区分训练和test ，验证 和train一样的dataset ， 需要提前划分好val的数据集地址
        super(sDataset, self).__init__()
        self.cat2label = {cat: i + 1 for i, cat in enumerate(CLASSES)}
        # prefix of images path
        self.img_prefix = img_prefix
        # load annotations (and proposals)加载标注信息 gtbox actionlabel等
        self.img_infos = self.load_annotations(ann_file)  # N个samples 的list
        # 加载已有的proposal 文件

        self.proposals = self.load_proposals(proposal_file)

        # (long_edge, short_edge)
        self.img_scales = img_scale
        # normalization configs
        self.img_norm_cfg = img_norm_cfg  # for examples :img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        # flip ratio
        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor
        # in test mode or not
        self.test_mode = test_mode
        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor,
            **self.img_norm_cfg)  # 初始化trainsform  return img, img_shape, pad_shape, scale_factor
        self.bbox_transform = BboxTransform()
        self.numpy2tensor = Numpy2Tensor()

    def __getitem__(self, idx):

        return self.prepare_img(idx)

    def prepare_img(self,idx):
        img_info = self.img_infos[idx]
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        # load proposals if necessary
        proposals = self.proposals[idx]  # idx 个 proposal box and label
        if len(proposals) == 0:
            return None
        if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                'but found {}'.format(proposals.shape))
        assert proposals.shape[1] == 5
        proposals_label = proposals[:, 4, None]
        proposals_bboxes = proposals[:, :4]
        # 这里是提出的标注信息， 也就是gtbox 和label
        ann = self.get_ann_info(idx)  # bboxes:  actionlabels:
        human_bboxes = ann['bboxes']
        action_labels = ann['actionlabels']
        # apply transforms
        img_scale = (self.img_scales)
        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, keep_ratio=False)
        img = img.copy()

        proposals_bboxes = self.bbox_transform(proposals_bboxes, img_shape, scale_factor,
                                               False)
        proposals = np.hstack(
            [proposals_bboxes, proposals_label])

        human_bboxes = self.bbox_transform(human_bboxes, img_shape, scale_factor,
                                           False)
        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(human_bboxes)))

        data['proposals'] = DC(to_tensor(proposals))
        data['gt_labels'] = DC(to_tensor(action_labels),stack=True)
        return data


    def __len__(self):
        len(self.img_infos)


    def load_annotations(self, ann_file):
        img_infos = []
        img_ids = mmcv.list_from_file(ann_file)# ann file 是划分数据集的标注文件。  读取txt
        for img_id in img_ids:
            img_id = img_id.split('.')[0]# Stanford的划分文件有问题，imgid都带了 。jpg 我们要去掉后缀
            #TODO: 增加对 png 等图片格式的支持
            filename = 'JPEGImages/{}.jpg'.format(img_id) # 文件名才是有后缀的 .jpg
            xml_path = osp.join(self.img_prefix, 'XMLAnnotations',
                                '{}.xml'.format(img_id))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            img_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))#四个基本信息
        return img_infos

    def  load_proposals(self, proposal_file):
        self.proposals = mmcv.load(proposal_file)
        proposal_bboxes = []
        proposal_labels = []
        for i, item in enumerate(self.proposals):
            # 处理一个图片
            # 处理一个图片
            pros = []
            lab = []
            for j, cla in enumerate(item):
                if cla.any():
                    for kk, loc in enumerate(cla):
                        pros.append(loc)
                        lab.append(j)
            if not pros:
                pros = np.zeros((0, 4))
                lab = np.zeros((0,))
            else:
                pros = np.array(pros).astype(np.float32)
                pros = pros[:, :4]
                lab = np.array(lab).astype(np.int64)
            proposal = np.concatenate((pros, lab[:, None]), axis=1)
            assert proposal.shape[1] == 5

            #proposal_labels.append(lab)
            proposal_bboxes.append(proposal)  # bbox and labels 都是有 总共的N个samples的 list
        # dict中两个obj，都是list， 每个list包含N个samples的 nd array对象 ，box或者label
        self.proposals = proposal_bboxes#dict(proposal_bboxes=proposal_bboxes, proposal_labels=proposal_labels)
        return self.proposals

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id'].split('.')[0]
        xml_path = osp.join(self.img_prefix, 'XMLAnnotations',
                            '{}.xml'.format(img_id))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        #bbox 存了person 的gt box
        bboxes = []
        finalbboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            #print(obj)
            action = obj.find('action').text
            #print(action)
            label = self.cat2label[action]
            #difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('xmin').text),
                int(bnd_box.find('ymin').text),
                int(bnd_box.find('xmax').text),
                int(bnd_box.find('ymax').text)
            ]

            bboxes.append(bbox)
            labels.append(label)


        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2).astype(np.float32) - 1
            labels = np.array(labels)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            actionlabels=labels.astype(np.int64))
        return ann

    def filter(self,human_bboxes, proposals_bboxes):
        new_proposal = []
        for ii,pro in proposals_bboxes:
            for jj,person in human_bboxes:
                iou = self.compute_iou(pro,person)
                if iou>0.2:
                    new_proposal.append(pro)
        new_proposal = np.array(new_proposal).astype(np.float32)
        return new_proposal

    def compute_iou(self, Reframe, GTframe):
        """
        自定义函数，计算两矩形 IOU，传入为均为矩形对角线，（x,y）  坐标。
        """
        x1 = Reframe[0]
        y1 = Reframe[1]
        width1 = Reframe[2] - Reframe[0]
        height1 = Reframe[3] - Reframe[1]

        x2 = GTframe[0]
        y2 = GTframe[1]
        width2 = GTframe[2] - GTframe[0]
        height2 = GTframe[3] - GTframe[1]

        endx = max(x1 + width1, x2 + width2)
        startx = min(x1, x2)
        width = width1 + width2 - (endx - startx)

        endy = max(y1 + height1, y2 + height2)
        starty = min(y1, y2)
        height = height1 + height2 - (endy - starty)

        if width <= 0 or height <= 0:
            ratio = 0  # 重叠率为 0
        else:
            Area = width * height  # 两矩形相交面积
            Area1 = width1 * height1
            Area2 = width2 * height2
            ratio = Area * 1. / Area1#(Area1 + Area2 - Area)
        # return IOU
        return ratio

if __name__ == '__main__':
    from mmcv.parallel import collate
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

    dataset = sDataset(ann_file='/home/share/LabServer/DATASET/stanford40/ImageSplits/train.txt',
                 img_prefix='/home/share/LabServer/DATASET/stanford40/',
                 img_scale=(224,224),
                 img_norm_cfg=img_norm_cfg,
                 size_divisor=32,
                 proposal_file='/home/share/LabServer/GLnet/stanford_train_bbox_new.pkl',
                 test_mode=False,)
    dataloader = DataLoader(dataset, shuffle=False,collate_fn=collate,batch_size=1, num_workers=1, )
    for ii, data in enumerate(dataloader):
        print(data)
        break



