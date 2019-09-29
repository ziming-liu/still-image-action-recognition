# coding:utf-8
import os
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable as V

from torchnet import meter
from config.config import cfg
from utils.visualize import Visualizer
from utils.show_masked_image import show_masked_image
from mmcv.runner import save_checkpoint,load_checkpoint
import cv2
from utils.show_masked_image import tensor_to_np
import numpy as np
#cfg.merge_from_file("config/un_att_pascal_0001.yaml")
vis = Visualizer("newvis", port=8097)

os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2"

def visualize_func(result):
    pass

def inverse_normalize(img):
    #if opt.caffe_pretrain:
    #    img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
    #    return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def show_keypoint(initimg,mask):
    #print(mask.shape)
    A = mask.cpu().numpy()
    #mask = mask.repeat(3,1,1)
    initimg = initimg.transpose(1,2,0)
    initimg_ = np.uint8(initimg)
    #mask_ = tensor_to_np(mask)
   # map = cv2.resize(mask_, (224, 224))
    #map = np.uint8(map)
    #heatmap = cv2.applyColorMap(map, cv2.COLORMAP_HSV)
    #result = heatmap * 0.4+ initimg_* 0.7
    #result = result.transpose(2,0,1)
    #heatmap = heatmap.transpose(2,0,1)
    initimg_ = initimg_.transpose(2,0,1)
    #print(A.shape)
    vis.heatmap(
        X=A,
        opts=dict(
            columnnames=['f'+str(i+1) for i in range(0,9,1)],
            rownames=['f'+str(i) for i in range(9,0,-1)],
           colormap='Electric',),
        win = "sdfas",
    )
    #vis.image(result, win='sdj',opts={'title':'keypointmap'})
    vis.image(initimg_, win='sd12j', opts={'title': 'keypoint'})

def toshowimg(initimg,subimg,title):
    #initimg_ = tensor_to_np(initimg)
    #subimg_ = tensor_to_np(subimg)
    subimg_ = subimg.transpose(1,2,0)
    subimg_= cv2.resize(subimg_,(224,224))
    #initimg_ = np.uint8(initimg_).transpose(2,0,1)
    subimg_ = np.uint8(subimg_).transpose(2,0,1)
    #vis_env.image(initimg_, win='jhkjk', opts={'title': 'initimg'})
    vis.image(subimg_, win=title, opts={'title': title})

#------------------------------------------
# from lib.model.resnet import resnet50
from lib.graphbased.attentionGCN import attentionGCN
from lib.channelmax2attention.backbone_model import backbone_model
#from torchvision.models import resnet50,resnet18,resnet34,resnet101,vgg19_bn
#model = vgg19_bn(pretrained=True)
from global_branch_cnn import resnet50 as globalcnn
#from context_cnn import resnet50 as contextcnn
from context_cnn_simple import resnet50 as contextcnnsimple
from contextcnn_input import resnet50 as contextcnninput
from torchvision.models import  resnet50 as stdres
from meshcnn import rgb_resnet50
import argparse
#from stanford_mesh_dataset import StanfordBox
from data.stanford_box import StanfordBox,Stanford_multi_resolution
from trainepoch import train_epoch
from valepoch import val_epoch
from warmup_scheduler import  GradualWarmupScheduler
from multi_resolution_cnn import multi_resolution_cnn
import adabound
from data.pascal_box.pascal_muilti_resolution import Pascal_multi_resolution
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--cfg_file",default=None, type=str)
    parser.add_argument("--result",default=None,type=str)
    parser.add_argument("--resume",action='store_true',
        help='Save data (.pth) of previous training')
    parser.set_defaults(resume=False)
    parser.add_argument("--dist",action='store_true', help='If true, dist is performed.')
    parser.set_defaults(dist=False)
    args = parser.parse_args()
    if args.dist:
        torch.cuda.set_device(args.local_rank)

        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
    if args.cfg_file != None:
        cfg.merge_from_file(args.cfg_file)
    trainlogwindow = vis.text("this is training log:\n",win="train")
    vis.text('configs:\n {}'.format(cfg.clone()),win=trainlogwindow,append=True)
    vallogwindow   = vis.text("this is validation log:\n",win="val")


    from torchvision.models import  resnet50,resnet101,densenet121,inception_v3
    from fusion_model import fusion_model
    from RFBA_multi_net import RFBA
    from inception_cnn import inception_cnn
    from densenet_cnn import densenet_cnn
    from RFBA import RFBA
    #model = RFBA()
    model = RFBA(depth=101,numclass=40)
    #for param in model.backbone.parameters():
    #    param.requires_grad = False
    #for param in model.atn_s4.parameters():
    #    param.requires_grad = False
    #model = resnet50(True)
    #num = model.fc.in_features
    #model.fc = torch.nn.Linear(num,40)
    #model = densenet121(pretrained=True)
    #num = model.classifier.in_features
    #model.classifier = torch.nn.Linear(num,40)
    #model = inception_v3(pretrained=True)
    #num = model.fc.in_features
    #model.fc = torch.nn.Linear(num,40)
    model = model.cuda()
    if args.dist:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                      output_device=args.local_rank)

    optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=1e-3)
    #scheduler_plateau = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,5], gamma=0.1, last_epoch=-1)
    scheduler_cos  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=200,eta_min=1e-7,last_epoch=-1)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1000, total_epoch=3,
                                              after_scheduler=scheduler_cos)

    if cfg.MODEL.TRAIN:
        if args.dist:
            train_dataset = Stanford_multi_resolution(cfg.clone(), train=True, val=False, crop=False)
            trainsampler = torch.utils.data.DistributedSampler(train_dataset)

            training_data = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,  num_workers=4,sampler=trainsampler )
        else:
            train_dataset = Stanford_multi_resolution(cfg.clone(), train=True, val=False, crop=False)
            training_data = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4 )

    if cfg.MODEL.VAL:
        if args.dist:
            val_dataset = Stanford_multi_resolution(cfg.clone(), train=False, val=True, crop=False)
            valsampler = torch.utils.data.DistributedSampler(val_dataset)
            validation_data = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, num_workers=4,sampler=valsampler)
        else:
            val_dataset = Stanford_multi_resolution(cfg.clone(), train=False, val=True, crop=False)
            validation_data = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, num_workers=4)

    if args.resume:
        load_checkpoint(model,cfg.MODEL.RESUME)
        #load_checkpoint(model, '/home/share2/zimi/GLnet/MODELZOO/local_multi/save_170.pth')
    criterion = torch.nn.CrossEntropyLoss()
    valid_accus = []
    for epoch in range(cfg.TRAIN.BEGIN_EPOCH,cfg.TRAIN.BEGIN_EPOCH+600):
        scheduler_cos.step()
        #scheduler_warmup.step()
        #scheduler_cos.step()
        print('[ Epoch', epoch, ']')
        if cfg.MODEL.TRAIN:
            start = time.time()

            train_loss, train_accu = train_epoch(epoch, training_data, model, criterion, optimizer, cfg,vis,trainlogwindow)
        else:
            train_accu=0
            train_loss=0
        if cfg.MODEL.VAL:
            start = time.time()
            # if epoch_i %3 ==0:
            valid_loss, valid_accu = val_epoch(epoch, validation_data, model, criterion, cfg, vis,vallogwindow)
            #scheduler_warmup.step()  # 更新学习率
        else:
            valid_loss=0
            valid_accu=0

        vis.plot_many_stack({'val accuracy': valid_accu, 'train accuracy': train_accu})
        vis.plot_many_stack({'val loss': valid_loss, 'train loss': train_loss})

        #if torch.cuda.current_device() == 0:
        #    valid_accus += [valid_accu]  # 存储了所有epoch的准确率
        #    if valid_accu >= max(valid_accus):
        #        save_checkpoint(model, cfg.MODEL.SAVE_IN + cfg.MODEL.NAME + '.pth')


    #train(model,baseline, training_data, validation_data, optimizer, scheduler, cfg.clone())
    #test(model,model2,validation_data)



if __name__ == '__main__':
    # import fire
    # fire.Fire()
    main()


