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
class StanfordBox(Dataset):
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
        super(StanfordBox, self).__init__()
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        self.img_prefix = '/home/share/DATASET/stanford40'
        datafile = '/home/share/DATASET/stanford40/selective_search/'
        if train and val:
            self.img_infos =[]
            self.img_infos.extend(self.load_annotations(self.img_prefix+'/ImageSplits/train.txt'))
            self.img_infos.extend(self.load_annotations(self.img_prefix+'/ImageSplits/test.txt'))
            #self.boxes = {}
            trainbox = scio.loadmat(datafile+'ss_box_train.mat')
            testbox = scio.loadmat(datafile+'ss_box_test.mat')
            self.boxes = {**trainbox,**testbox}
        elif train:
            self.img_infos = self.load_annotations(self.img_prefix+'/ImageSplits/train.txt')
            self.boxes = scio.loadmat(datafile + 'ss_box_train.mat')
        elif val:
            self.img_infos = self.load_annotations(self.img_prefix+'/ImageSplits/test.txt')
            self.boxes = scio.loadmat(datafile + 'ss_box_test.mat')

        self.cfg = cfg
        self.train = train
        self.crop = crop
        self.mesh = []
        internal = 224//3
        for i in range(3):
            ymin = i*internal
            for j in range(3):
                xmin = internal*j
                loc = np.array((xmin,ymin,xmin+internal,ymin+internal))
                self.mesh.append(loc)
        self.mesh = np.array(self.mesh).astype(np.uint8)


    def __getitem__(self, item):
        img_id = self.img_infos[item]['id']
        filename = self.img_infos[item]['filename']
        #print(self.boxes.keys())
        #print(img_id)
        #ss_box = self.boxes[img_id][:16,:]
        ss_box = self.mesh
        #print(ss_box.shape)

        bbox_label = self.get_ann_info(item)
        image = cv2.imread(osp.join(self.img_prefix,filename))
        bboxes = bbox_label['bboxes']#person
        labeles = bbox_label['labels']
        randidx = np.random.randint(0,len(labeles))
        bbox = bboxes[randidx]
        label = labeles[randidx] - 1
        fullimg = image
        fullimg = Image.fromarray(cv2.cvtColor(fullimg, cv2.COLOR_BGR2RGB))
        trn = T.Compose([T.Resize(self.cfg.DATASET.IMAGE_SIZE),
                         T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.5, 0.5, 0.5])])
        fullimg = trn(fullimg)
        subimgs = []
        for ii in range(self.mesh.shape[0]):
            gezi = self.mesh[ii]
            x1, y1, x2, y2 = gezi[0], gezi[1], gezi[2], gezi[3]
            subimg = image[y1:y2, x1:x2, :]
            #print("subimg {}".format(subimg.shape))
            #subimg = cv2.resize(subimg,(224,224))
            #assert label!=-1,img_id
            #if label==10:
            #    self.count +=1
            #    print(self.count)
            #if self.crop:
                #print("crop person image!!!!!")
            #    person_img = mmcv.imcrop(image,bbox,scale=1.0)
            #else:
            #    person_img = image
            import os
            #path_person = os.path.join(self.img_prefix,'PersonImages')
            #if not os.path.exists(path_person):
            #    os.makedirs(path_person)
            #cv2.imwrite(os.path.join(path_person,img_id+'.jpg'),person_img)
            # cv2 to PIL
            subimg = Image.fromarray(cv2.cvtColor(subimg, cv2.COLOR_BGR2RGB))
            #img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            #transform
            #if self.train:
            #    trn = T.Compose([T.Resize(self.cfg.DATASET.IMAGE_SIZE),
            #                     #T.RandomResizedCrop(224, (0.5, 1.0)),
            #                     T.RandomHorizontalFlip(0.7),
                                 #T.RandomRotation((0, 0.5)),
            #                     T.ToTensor(),
            #                     T.Normalize(mean=[0.485, 0.456, 0.406],
            #                                 std=[0.5, 0.5, 0.5])])
            #else:
            trn = T.Compose([T.Resize((112,112)),
                                 T.ToTensor(),
                                 T.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.5, 0.5, 0.5])])
            subimg = trn(subimg)
            subimgs.append(subimg)

        subimgs = torch.stack(subimgs)
        #img = trn(img)
        return fullimg,subimgs,label
        #return img,person_img,label
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
    dataset = StanfordBox(cfg,train=False,val=True)
    dataloader = DataLoader(dataset,batch_size=80,shuffle=False,num_workers=4)
    for ii,(box,img,label) in enumerate(dataloader):
        #print(img.type())
        #print(box.shape)
        #print(label.shape)
        break
        #print(label.type())
    print("over")
   # print(dataset.count)
