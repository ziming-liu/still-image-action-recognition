from  torch.utils.data import Dataset
from  torch.utils.data import DataLoader
import torch as tf
from util.config import opt
import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from data.util import read_image
import scipy
import xml.dom.minidom as minidom
from data.util import read_image
from torchvision import transforms


class dataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        super(dataset, self).__init__()
        # if split not in ['train', 'trainval', 'val']:
        #     if not (split == 'test' and year == '2007'):
        #         warnings.warn(
        #             'please pick split from \'train\', \'trainval\', \'val\''
        #             'for 2012 dataset. For 2007 dataset, you can pick \'test\''
        #             ' in addition to the above mentioned splits.'
        #         )
        classes = VOC_BBOX_LABEL_NAMES.keys()
        ids = []
        labels = []
        human_rois = []
        for cla in classes:
            id_list_file = os.path.join(
                data_dir, 'ImageSets/Action/{0}_{1}.txt'.format(cla,split))
            for id_ in open(id_list_file):
                    if id_.strip().split()[-1] == '1':
                        id = id_.strip().split(' ')[0]
                        ids.append(id)
                        person_id = int(id_.strip().split()[-2])
                        label = VOC_BBOX_LABEL_NAMES[cla]
                        labels.append(label)

            #id = [id_.strip().split()[0] for id_ in open(id_list_file) if id_.strip().split()[-1]=='1']
            ##ids.extend(id)
            #label = [int(id_.strip().split()[-1]) for id_ in open(id_list_file) if id_.strip().split()[-1]=='1']
            #labels.extend(label)

        self.ids = ids
        self.labels = labels
        self.transform = transform
        self.data_dir = data_dir
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        label = self.labels[idx]

        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = Image.open(img_file)
        if self.transform:
            img = self.transform(img)
        return img,label
    def __len__(self):
        return len(self.ids)


VOC_BBOX_LABEL_NAMES = {"jumping":0,
                        "phoning":1,
                        "playinginstrument":2,
                        "reading":3,
                        "ridingbike":4,
                        "ridinghorse":5,
                        "running":6,
                        "takingphoto":7,
                        "usingcomputer":8,
                        "walking":9

                        }

def dataloader(opt,split='train'):
    simple_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset_ = dataset(opt.voc_data_dir,split,simple_transform)
    dataloader_ = DataLoader(dataset_,batch_size=16,num_workers=4,shuffle=True)
    return dataloader_

if __name__ == '__main__':
    d = dataloader(opt,'train')
    for ii,(img,label) in enumerate(d):
        print(img)
        print(label)
