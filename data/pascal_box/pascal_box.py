from torch.utils.data import Dataset,DataLoader
import torch
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np
from PIL import Image
import cv2
from  torchvision import transforms as T

from config.config import cfg

class PascalBox(Dataset):
    def __init__(self,cfg,train=True,val=False,test=False):
        self.CLASSES = ('jumping','phoning','playinginstrument',
                      'reading','ridingbike','ridinghorse',
                      'running','takingphoto','usingcomputer',
                      'walking','other')
        super(PascalBox, self).__init__()
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        self.img_prefix = '/home/share/DATASET/VOCdevkit/VOC2012'
        if train:
            self.img_infos = self.load_annotations(self.img_prefix+'/ImageSets/Action/train.txt')
        elif val:
            self.img_infos = self.load_annotations(self.img_prefix+'/ImageSets/Action/val.txt')
        elif test:
            self.img_infos = self.load_annotations(self.img_prefix + '/ImageSets/Action/test.txt')
        self.train = train
        self.cfg = cfg
        self.count =0
    def __getitem__(self, item):
        img_id = self.img_infos[item]['id']
        filename = self.img_infos[item]['filename']
        bbox_label = self.get_ann_info(item)
        image = cv2.imread(osp.join(self.img_prefix,filename))
        bboxes = bbox_label['bboxes']
        labeles = bbox_label['labels']
        randidx = np.random.randint(0,len(labeles))
        bbox = bboxes[randidx]
        label = labeles[randidx] - 1
        #assert label!=-1,img_id
        #if label==10:
        #    self.count +=1
        #    print(self.count)
        if self.cfg.DATASET.CROP_BBOX:
            #print("crop person image!!!!!")
            person_img = mmcv.imcrop(image,bbox,scale=1.0)
        else:
            person_img = image
        # cv2 to PIL
        person_img = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
        #transform
    #if self.train:
        trn = T.Compose([T.Resize(self.cfg.DATASET.IMAGE_SIZE),
                         T.RandomResizedCrop(224,(0.5,1.0)),
                         T.RandomHorizontalFlip(0.7),
                         T.RandomRotation((0,0.5)),
                         T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.5,0.5,0.5])])
    #else:
        trn = T.Compose([T.Resize(self.cfg.DATASET.IMAGE_SIZE),
                         T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.5, 0.5, 0.5])])
        person_img = trn(person_img)
        return person_img,label
    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        img_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = 'JPEGImages/{}.jpg'.format(img_id)
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                '{}.xml'.format(img_id))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            img_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations',
                            '{}.xml'.format(img_id))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            assert name =='person'
            actions = obj.find('actions')
            if int(actions.find("jumping").text)==1:
                label = self.cat2label["jumping"]
            elif int(actions.find("phoning").text)==1:
                label = self.cat2label["phoning"]
            elif int(actions.find("playinginstrument").text)==1:
                label = self.cat2label["playinginstrument"]
            elif int(actions.find("reading").text)==1:
                label = self.cat2label["reading"]
            elif int(actions.find("ridingbike").text)==1:
                label = self.cat2label["ridingbike"]
            elif int(actions.find("ridinghorse").text)==1:
                label = self.cat2label["ridinghorse"]
            elif int(actions.find("running").text)==1:
                label = self.cat2label["running"]
            elif int(actions.find("takingphoto").text)==1:
                label = self.cat2label["takingphoto"]
            elif int(actions.find("usingcomputer").text)==1:
                label = self.cat2label["usingcomputer"]
            elif int(actions.find("walking").text)==1:
                label = self.cat2label["walking"]
            else:
                #raise EOFError
                label = self.cat2label["other"]

            difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            bbox = [
                round(float(bnd_box.find('xmin').text)),
                round(float(bnd_box.find('ymin').text)),
                round(float(bnd_box.find('xmax').text)),
                round(float(bnd_box.find('ymax').text))
                ]
            if difficult:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0,))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0,))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann


if __name__ == '__main__':
    dataset = PascalBox(train=True,val=False)
    dataloader = DataLoader(dataset,batch_size=80,shuffle=False,num_workers=1)
    for ii,(img,label) in enumerate(dataloader):
        ##print(img.type())
        #print(label.shape)
        #print(label.type())
        pass
    print(dataset.count)
