from torch.utils.data import DataLoader,Dataset
from PIL import Image
import os
import scipy.io as io
#from utils.config import cfg
import xml.etree.ElementTree as ET
import numpy as np
from torchvision import transforms as T


def build_transform():
    """
    Creates a basic transformation that was used to train the models
    """


    # we are loading images with OpenCV, so we don't need to convert them
    # to BGR, they are already! So all we need to do is to normalize
    # by 255 if we want to convert to BGR255 format, or flip the channels
    # if we want it to be in RGB in [0-1] range.
    #if False:
    #    to_bgr_transform = T.Lambda(lambda x: x * 255)
    #else:
    #   to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

    normalize_transform = T.Normalize(
        mean= [0.5,0.5,0.5], std=[0.5,0.5,0.5]
    )

    transform = T.Compose(
        [

            T.Resize((224,224)),
            T.ToTensor(),
            #to_bgr_transform,
            normalize_transform,
        ]
    )
    return transform

simpletransform = T.Compose([
        T.Resize((224,224)),


        T.RandomHorizontalFlip(0.5),
        #T.RandomVerticalFlip(0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.5,0.5,0.5])

    ])
strongtransform = T.Compose([
        T.Resize((224,224)),
        T.RandomResizedCrop(224,(0.5,1.0)),
        T.RandomHorizontalFlip(0.7),
        #T.RandomVerticalFlip(0.1),
        #T.RandomGrayscale(1),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
class Stanford40(Dataset):
    def __init__(self,root,phase='train',saliency_root=None):
        super(Stanford40, self).__init__()
        if phase=='train':
            transform = simpletransform
        else:
            transform =simpletransform

        self.root = root
        split_path = os.path.join(root , 'ImageSplits')
        classes = []
        with open(split_path+'/actions.txt') as f:
            txt = f.readlines()
            for ii,name in enumerate(txt):
                if ii>0:
                    classes.append(name.strip().split()[0])
        #print(classes)
        #print(len(classes))
        self.classes = classes
        self.sample = []
        self.label = []
        with open(os.path.join(split_path,phase+'.txt')) as f:
            txt = f.readlines()
            for ii,item in enumerate(txt):
                self.sample.append(item.strip())
                cla = item.strip().split('.')[0].split('_')[:-1]
                cla = '_'.join(cla)
                #print(cla)
                self.label.append(self.classes.index(cla))
        #print(self.label)
        self.saliency_root = saliency_root#cam 的路径
        self.transform = transform
        import h5py
        with h5py.File(''.join([cfg.DATASET.SCENELABELS + '/scenelabels_dataset_'+phase+'.h5']), 'r') as f:  ## 写方式打开文件
            self.name = f['name'].value
            self.IO = f['IO'].value
            self.category = f['category'].value
    def __getitem__(self, item):
        img_path = os.path.join(self.root , 'JPEGImages' , self.sample[item])
        name = [[self.sample[item]]]
        assert self.name[item][0] == self.sample[item]
        image = Image.open(img_path).convert('RGB')
        w,h = image.size
        iolable = self.IO[item]
        catelabel = self.category[item]

        if self.saliency_root:
            cam_path = os.path.join(self.saliency_root , self.sample[item])
            cam = Image.open(cam_path)
        if self.transform:
            image = self.transform(image)
            if self.saliency_root:
                cam = self.transform(cam)

        label = self.label[item]
        """
        #annotation
        annotation_path = ET.parse(os.path.join(self.root,
                                            'XMLAnnotations',
                                            self.sample[item].split('.')[0]+'.xml'))
        bbox = []
        objs =  annotation_path.findall('object')
        obj = objs[0]

        act = str(obj.find('action').text)
        #print(self.classes.index(act))
        #print(self.label)
        assert self.classes.index(act) == self.label[item],'label is not correct'
        bndbox_anno = obj.find('bndbox')
        # subtract 1 to make pixel indexes 0-based
        
        bbox.append([
            round(float(bndbox_anno.find(tag).text)) - 1
            for tag in ( 'xmin', 'ymin', 'xmax','ymax')])

        loc = bbox

        #转化坐标缩放带来的影响
        x_factor = 224.0/w
        y_factor = 224.0/h
        loc[0][0] =  int(loc[0][0]*x_factor)
        loc[0][1] =  int(loc[0][1]*y_factor)
        loc[0][2] =  int(loc[0][2]*x_factor)
        loc[0][3] =  int(loc[0][3]*y_factor)
        #得到四个偏移位置坐标
        #up
        xmin = loc[0][0]
        ymin = 0
        xmax = loc[0][2]
        ymax = loc[0][3] - (loc[0][1])
        loc.append([xmin + 1, ymin + 1, xmax - 1, ymax - 1])
        # down
        xmin = loc[0][0]
        ymin = loc[0][1] +(223-loc[0][3])
        xmax = loc[0][2]
        ymax = 223
        loc.append([xmin + 1, ymin + 1, xmax - 1, ymax - 1])
        # left
        xmin = 0
        ymin = loc[0][1]
        xmax = loc[0][2]-loc[0][0]
        ymax = loc[0][3]
        loc.append([xmin + 1, ymin + 1, xmax - 1, ymax - 1])
        # right
        xmin = loc[0][0] + (223-loc[0][2])
        ymin = loc[0][1]
        xmax = 223
        ymax = loc[0][3]
        loc.append([xmin+1, ymin+1, xmax-1, ymax-1])

        loc = np.array(loc)
        #index = item * np.ones((5), dtype=np.int32)
        #newloc = np.concatenate([index[:, None], loc],axis=1)
        #assert loc.shape[1] == 4, print(self.sample[item])

        #print(self.sample[item])
        #print(image.shape)
        #print(label)
        #print(loc.shape)
        """
        if self.saliency_root:
            return image,cam,label
        else:
            return image,label
    def __len__(self):
        return len(self.sample)
import torch as t
if __name__ == '__main__':
    from torchvision import transforms as T
    simpletransform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    ])
    strongtransform = T.Compose([
        T.RandomCrop((224, 224)),
        T.RandomResizedCrop((224,224),(0.5,1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    data = Stanford40(cfg.DATASET.STANFORD40,phase='test'
                                                   '',)
    dataloader = DataLoader(data,num_workers=4,batch_size=128,shuffle=False)
    for ii,(data) in enumerate(dataloader):
        #print(ii*128)
        #print(data[1].shape)
        c = data[3].view(data[3].size(0), -1)
        col = c[:,1].int()
        print(col)
        print(col.shape)




    print('ok')







