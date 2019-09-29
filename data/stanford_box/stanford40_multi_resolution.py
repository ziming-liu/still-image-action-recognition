
from torch.utils.data import Dataset,DataLoader
import torch
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np
from PIL import Image
import cv2
from  torchvision import transforms as T
import scipy.io as scio
import PIL
class Stanford_multi_resolution(Dataset):
    def __init__(self,cfg,train=True,val=False,crop = False):
        self.CLASSES = ('applauding',
                        'blowing_bubbles',
                        'brushing_teeth',
                        'cleaning_the_floor',
                        'climbing',
                        'cooking',
                        'cutting_trees',
                        'cutting_vegetables',####
                        'drinking',
                        'feeding_a_horse',
                        'fishing',
                        'fixing_a_bike',
                        'fixing_a_car',
                        'gardening',
                        'holding_an_umbrella',
                        'jumping',
                        'looking_through_a_microscope',
                        'looking_through_a_telescope',
                        'playing_guitar',
                        'playing_violin',
                        'pouring_liquid',##
                        'pushing_a_cart',
                        'reading',##
                        'phoning',##
                        'riding_a_bike',
                        'riding_a_horse',
                        'rowing_a_boat',
                        'running',
                        'shooting_an_arrow',
                        'smoking',
                        'taking_photos',##
                        'texting_message',##
                        'throwing_frisby',
                        'using_a_computer',
                        'walking_the_dog',
                        'washing_dishes',
                        'watching_TV',
                        'waving_hands',
                        'writing_on_a_board',
                        'writing_on_a_book')
        super(Stanford_multi_resolution, self).__init__()
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        self.img_prefix = '/home/share2/zimi/DATASET/stanford40'
        datafile = '/home/share2/zimi/DATASET/stanford40/selective_search/'
        if train and val:
            self.img_infos =[]
            self.img_infos.extend(self.load_annotations(self.img_prefix+'/ImageSplits/train.txt'))
            self.img_infos.extend(self.load_annotations(self.img_prefix+'/ImageSplits/test.txt'))
            #self.boxes = {}
        elif train:
            self.img_infos = self.load_annotations(self.img_prefix+'/ImageSplits/train.txt')
        elif val:
            self.img_infos = self.load_annotations(self.img_prefix+'/ImageSplits/test.txt')

        self.cfg = cfg
        self.train = train
        self.crop = crop
        self.mesh = []


    def __getitem__(self, item):
        img_id = self.img_infos[item]['id']
        filename = self.img_infos[item]['filename']
        #print(self.boxes.keys())
        #print(img_id)
        #ss_box = self.boxes[img_id][:30,:]
        #ss_box = self.mesh
        #print(ss_box.shape)

        bbox_label = self.get_ann_info(item)
        image = cv2.imread(osp.join(self.img_prefix,filename))
        bboxes = bbox_label['bboxes']#person
        labeles = bbox_label['labels']
        randidx = np.random.randint(0,len(labeles))
        bbox = bboxes[randidx]
        label = labeles[randidx] - 1
        #assert label!=-1,img_id
        #if label==10:
        #    self.count +=1
        #    print(self.count)

        import os
        #path_person = os.path.join(self.img_prefix,'PersonImages')
        #if not os.path.exists(path_person):
        #    os.makedirs(path_person)
        #cv2.imwrite(os.path.join(path_person,img_id+'.jpg'),person_img)
        # cv2 to PIL
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #transform
        new_image = []# T=21
        if self.train:
            #for sizer in range(self.cfg.DATASET.LOW_RESOLUTION,self.cfg.DATASET.HIGH_RESOLUTION+1,self.cfg.DATASET.INTERNAL):
            for ii in range(9):
                trn = T.Compose([
                                 #T.RandomAffine(degrees=80,translate=(0.1,0.1),scale=(0.01,10),
                                 #               shear=(0.1,0.5),resample=Image.BILINEAR,fillcolor=0),
                                 T.RandomResizedCrop(224, (0.1, 1.0)),
                                 #T.RandomHorizontalFlip(0.7),
                                 #T.RandomRotation((-30, 30),expand=True),
                                 T.Resize((224, 224)),
                                 T.ToTensor(),
                                 T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                subimg = trn(image)
                new_image.append(subimg)
            new_image = torch.stack(new_image)
        else:
            """
            for sizer in range(self.cfg.DATASET.LOW_RESOLUTION, self.cfg.DATASET.HIGH_RESOLUTION+1,
                                   self.cfg.DATASET.INTERNAL):
                trn = T.Compose([
                    #T.RandomResizedCrop(224, (0.3, 1.0)),
                    T.Resize((224,224)),
                    T.CenterCrop((sizer,sizer)),
                    T.Resize((224,224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                subimg = trn(image)
                new_image.append(subimg)
            new_image = torch.stack(new_image)

            """
            for ii in range(120):
                trn = T.Compose([
                T.RandomResizedCrop(224, (0.1, 1.0)),
                T.Resize((224, 224)),
                             T.ToTensor(),
                             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                subimg = trn(image)
                new_image.append(subimg)
            new_image = torch.stack(new_image)
            """
            transform = T.Compose([T.Resize(224),
                            T.TenCrop(224),  # this is a list of PIL Images
                       T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops]))])  # returns a 4D tensor
            new_image = transform(image)
            """
        return new_image,label
    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        img_infos = []
        # Stanford datset id is  xxx.jpg
        img_full_names = mmcv.list_from_file(ann_file)
        for img_name in img_full_names:
            img_id = img_name.split('.')[0]
            filename = 'JPEGImages/{}'.format(img_name)
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

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'XMLAnnotations',
                            '{}.xml'.format(img_id))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []

        for obj in root.findall('object'):
            name = obj.find('name').text
            assert name =='person'
            action = obj.find('action').text
            label = self.cat2label[action]
            bnd_box = obj.find('bndbox')
            bbox = [
                round(float(bnd_box.find('xmin').text)),
                round(float(bnd_box.find('ymin').text)),
                round(float(bnd_box.find('xmax').text)),
                round(float(bnd_box.find('ymax').text))
                ]

            bboxes.append(bbox)
            labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0,))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)

        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
           )
        return ann


if __name__ == '__main__':
    from config.config import cfg
    from utils.visualize import  Visualizer

    vis = Visualizer("newvis", port=8097)

    cfg.merge_from_file("/home/share/LabServer/GLnet/config/multi_resolution.yaml")
    dataset = Stanford_multi_resolution(cfg,train=False,val=True)
    dataloader = DataLoader(dataset,batch_size=10,shuffle=False,num_workers=4)
    for ii,(img,label) in enumerate(dataloader):
        #print(img.type())
        print(type(img))
        print(len(img))
        print(img[3].shape)
        print(label)
        #print(label.shape)
        #print(label.type())
    print("over")
   # print(dataset.count)













