
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
               'playing_guitar','playing_violin','pouring_liquid','pushing_a_cart','reading',
               'phoning','riding_a_bike','riding_a_horse','rowing_a_boat','running',
               'shooting_an_arrow','smoking','taking_photos','texting_message',
               'throwing_frisby','using_a_computer','walking_the_dog','washing_dishes',
               'watching_TV','waving_hands','writing_on_a_board','writing_on_a_book')
    def __init__(self, **kwargs):
        super(StanfordDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        self.label2cat = {i + 1: cat_id for i, cat_id in enumerate(self.CLASSES)}

        if not self.test_mode:
            self.bbox_infos = mmcv.load('/home/share/LabServer/GLnet/stanford_train_bbox_new.pkl')
        else:
            self.bbox_infos = mmcv.load('/home/share/LabServer/GLnet/stanford_test_bbox_new.pkl')


    def load_annotations(self, ann_file):
        img_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            img_id = img_id.split('.')[0]
            filename = 'JPEGImages/{}.jpg'.format(img_id)
            xml_path = osp.join(self.img_prefix, 'XMLAnnotations',
                                '{}.xml'.format(img_id))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            img_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))
        return img_infos

    def compute_iou(self,Reframe, GTframe):
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
            ratio = Area * 1. / (Area1 + Area2 - Area)
        # return IOU
        return ratio
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
            bboxes = np.array(finalbboxes, ndmin=2) - 1
            labels = np.array(labels)
        bboxes_allclasses = self.bbox_infos[idx]

        for ii,item in enumerate(bboxes_allclasses):
            #if ii>0 and item.any():#第一类是person，我们用gt box of person
            #每类中的所有box
            for k,eachbox in enumerate(item):
                #与每个person gt box
                for jj,person in enumerate(bboxes):
                    #xy->yx
                    #xmin,ymin,xmax,ymax = person
                    #xmin_,ymin_,xmax_,ymax_ = eachbox[0],eachbox[1],eachbox[2],eachbox[3]
                    #m = (ymin,xmin,ymax,xmax)
                    #n = (ymin_,xmin_,ymin_,ymax_)
                    m = person
                    n = np.array(eachbox)
                    iou = self.compute_iou(m,n)
                    print('iou {} '.format(iou))
                    if iou> 0.0:
                        finalbboxes.append(eachbox)
        #for jj, person in enumerate(bboxes):
        #    finalbboxes.append(np.array(person))


        ann = dict(
            bboxes=finalbboxes.astype(np.float32),
            labels=labels.astype(np.int64))
        return ann

