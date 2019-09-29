
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np

from .custom import CustomDataset

iouthr = 0.2
class StanfordDataset(CustomDataset):
    CLASSES = ('applauding','blowing_bubbles','brushing_teeth','cleaning_the_floor',
               'climbing','cooking','cutting_trees','cutting_vegetables','drinking',
               'feeding_a_horse','fishing','fixing_a_bike','fixing_a_car','gardening',
               'holding_an_umbrella','jumping','looking_through_a_microscope','looking_through_a_telescope',
               'playing_guitar','playing_violin','pouring_liquid','pushing_a_cart','reading',
               'phoning','riding_a_bike','riding_a_horse','rowing_a_boat','running',
               'shooting_an_arrow','smoking','taking_photos','texting_message',
               'throwing_frisby','using_a_computer','walking_the_dog','washing_dishes',
               'watching_TV','waving_hands','writing_on_a_board','writing_on_a_book')
    def __init__(self, **kwargs):
        super(StanfordDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        self.label2cat = {i + 1: cat_id for i, cat_id in enumerate(self.CLASSES)}

        #if not self.valmode:
        #    self.bbox_infos = mmcv.load('/home/share/LabServer/GLnet/stanford_train_bbox_new.pkl')
        #else:
        #    self.bbox_infos = mmcv.load('/home/share/LabServer/GLnet/stanford_test_bbox_new.pkl')


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
            pros = []
            lab = []
            for j, cla in enumerate(item):
                if cla.any():
                    for kk, loc in enumerate(cla):
                        pros.append(loc)
                        lab.append(j)
            if  pros:
                pros = np.array(pros).astype(np.float32)
                pros = pros[:,:4]
                lab = np.array(lab).astype(np.int64)
                proposal = np.concatenate((pros,lab[:,None]),axis=1)
                assert proposal.shape[1] == 5
            else:
                proposal =None

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

