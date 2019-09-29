#coding:utf-8


from torchvision.datasets import ImageFolder
import torch as t
from torch.utils.data import DataLoader
import torchvision.transforms as T
from utils.config import cfg
from data.pascalVOC.newImageFolder import ImageFolder as newIF


def get_basicDataloader():
    simple_transform = T.Compose([
     #   T.RandomHorizontalFlip(),
     #   T.RandomRotation(0.2),
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = ImageFolder(cfg.DATASET.train_img_folder,simple_transform)
    val_data = ImageFolder(cfg.DATASET.val_img_folder,simple_transform)

    train_dataloader = DataLoader(train_data,cfg.DATASET.BATCH_SIZE,shuffle=True,num_workers=4)
    val_dataloader = DataLoader(val_data,cfg.DATASET.BATCH_SIZE,shuffle=False,num_workers=4)

    return train_dataloader,val_dataloader

def camDataloader():
    simple_transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if opt.which =="training":
        data = newIF(opt.train_img_folder,simple_transform)
    elif opt.which =="validation":
        data = newIF(opt.val_img_folder,simple_transform)
    elif opt.which =="testing":
        data = newIF(opt.test_img_folder,simple_transform)
    dataloader = DataLoader(data,opt.batch_size,shuffle=True,num_workers=opt.num_worker)
    #val_dataloader = DataLoader(data,opt.batch_size,shuffle=False,num_workers=opt.num_worker)

    return dataloader,data.classes#包括img label 和name.jpg
