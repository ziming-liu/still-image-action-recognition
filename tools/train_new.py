#coding:utf-8
import os
import math
import time
import numpy as np
from  torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable as V

from torchnet import meter

from util.config import cfg
from util.visualize import Visualizer

from lib.module.FindProposal import FindProposal
import logging
import os
from argparse import ArgumentParser
from collections import OrderedDict
from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import Runner, DistSamplerSeedHook
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from mmcv.parallel import collate,DataContainer,MMDataParallel,MMDistributedDataParallel,scatter
#import resnet_cifar
from mmdet.datasets import StanfordDataset
import sys
sys.path.append("..")
#from data.datasets import StanfordDataset
from lib.model.proposalModel import proposalModel,proposalModel_scene

#create visulized env
vis = Visualizer(cfg.SYSTEM.NAME, port=8097)
#measures created
AP = meter.APMeter()
mAP = meter.mAPMeter()
Loss_meter = meter.AverageValueMeter()
#set cuda env
os.environ["CUDA_VISIBLE_DEVICES"] = "2"



def train_epoch(model, training_data, optimizer):
    ''' Epoch operation in training phase'''
    AP.reset()
    mAP.reset()
    Loss_meter.reset()
    model.train()


    for batch_idx, (data) in enumerate(training_data):
        img = data['img'].data[0]
        label = data['gt_labels'].data[0]
        proposals = data['proposals'].data[0]
        imgmeta = data['img_meta'].data[0]
        gtbox = data['gt_bboxes'].data[0]
        label = label.cuda().squeeze(1)
        for ii in range(len(proposals)):
            proposals[ii] = proposals[ii].cuda()
        for ii in range(len(gtbox)):
            gtbox[ii] = gtbox[ii].cuda()
        # forward
        optimizer.zero_grad()
        pred = model(img.cuda(), gtbox, proposals, label)

        pred = F.softmax(pred, 1)
        #backward
        loss = F.cross_entropy(pred,label)

        Loss_meter.add(loss.detach().item())#转化为了numpy类型

        loss.backward()
        #optimize
        optimizer.step()

        #calculate acc
        one_hot = torch.zeros_like(pred).cuda().scatter(1, label.view(-1, 1), 1)

        AP.add(pred.detach(), one_hot)
        mAP.add(pred.detach(), one_hot)

    return Loss_meter.value()[0], mAP.value()

def eval_epoch( model, validation_data):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    AP.reset()
    mAP.reset()
    Loss_meter.reset()

    for batch_idx, (data) in enumerate(validation_data):
        img = data['img'].data[0]
        label = data['gt_labels'].data[0]
        proposals = data['proposals'].data[0]
        imgmeta = data['img_meta'].data[0]
        gtbox = data['gt_bboxes'].data[0]
        label = label.cuda().squeeze(1)
        for ii in range(len(proposals)):
            proposals[ii] = proposals[ii].cuda()
        for ii in range(len(gtbox)):
            gtbox[ii] = gtbox[ii].cuda()
        # forward
        pred = model(img.cuda(), gtbox, proposals, label)

        pred = F.softmax(pred, 1)
        #calculate  loss
        loss = F.cross_entropy(pred,label)
        Loss_meter.add(loss.detach().item())  # 转化为了numpy类型
        # calculate acc
        #one_hot = torch.zeros_like(pred).cuda().scatter(1, target.view(-1, 1), 1)

        one_hot = torch.zeros_like(pred).cuda().scatter(1, label.view(-1, 1), 1)

        AP.add(pred.detach(), one_hot)
        mAP.add(pred.detach(), one_hot)

    return Loss_meter.value()[0], mAP.value()

def train(model, training_data, validation_data, optimizer,scheduler, cfg):
    ''' Start training '''

    valid_accus = []
    for epoch_i in range(cfg.TRAIN.EPOCHES):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, optimizer,)
        print('  - (Training) ,loss: {loss:3.3f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(loss = train_loss,
             accu=100*train_accu,elapse=(time.time()-start)/60))
        vis.plot_many({'train loss':train_loss})
        vis.plot_many({'train accuracy': train_accu})
        vis.log(
            "Phase:{phase},Epoch:{epoch},AP:{AP},mAP:{mAP},train_loss:{loss}".format(
                phase="train", epoch=epoch_i, AP=AP.value(), mAP=mAP.value(), loss=Loss_meter.value()[0],
            ))


        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data,)
        print('  - (Validation)  ,loss: {loss:3.3f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(loss = valid_loss,
                     accu=100*valid_accu,elapse=(time.time()-start)/60))
        #scheduler.step(valid_loss)#更新学习率
        """
        if epoch_i==4:
            for param_group in (optimizer.param_groups):
                param_group['lr'] = 1e-4
        if epoch_i==8:
            for param_group in (optimizer.param_groups):
                param_group['lr'] = 1e-3

        if epoch_i==10:
            for param_group in (optimizer.param_groups):
                param_group['lr'] = 1e-5
        """
        vis.plot_many({'val loss': valid_loss})
        vis.plot_many({'val accuracy': valid_accu})
        vis.log(
            "Phase:{phase},Epoch:{epoch}, AP:{AP},mAP:{mAP},val_loss:{loss}".format(
                phase="validation", epoch=epoch_i, AP=AP.value(), mAP=mAP.value(), loss=Loss_meter.value()[0],
            ))


        valid_accus += [valid_accu]#存储了所有epoch的准确率

        model_state_dict = {'model':model.state_dict(),}
                            #'submodel':submodel.state_dict()}
        #if validattion accuracy is higher ,we need to save this result
        savingpath = os.path.join(cfg.MODEL.SAVE_IN,cfg.SYSTEM.NAME)
        if os.path.exists(savingpath) is False:
            os.mkdir(savingpath)
        if cfg.MODEL.SAVE_IN:
            if cfg.MODEL.SAVE_MODE == 'all':
                model_name = savingpath +'/' + cfg.SYSTEM.NAME+ 'epoch_{}_accu_{accu:3.3f}.pth'.format(epoch_i,accu=100*valid_accu)
                torch.save(model_state_dict, model_name)
            elif cfg.MODEL.SAVE_MODE == 'best':
                model_name = savingpath +'/'+cfg.SYSTEM.NAME + '_best.pth'
                if valid_accu >= max(valid_accus):
                    torch.save(model_state_dict, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

from lib.model.wideresnet import resnet50

def main():
    ''' Main function '''
    if cfg.SYSTEM.UPDATE_CFG:
        cfg.merge_from_file(cfg.SYSTEM.CFG_FILE)
    cfg.freeze()#冻结参数
    print(cfg)

    vis.log('configs:\n {}'.format(cfg.clone()))
    #LOAD DATA
    ##training_data, validation_data = prepare_stanford40(cfg.clone())
    #training_data, validation_data = prepare_pascal(cfg.clone())
    #normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

    val_dataset = StanfordDataset(ann_file='/home/share/LabServer/DATASET/stanford40/ImageSplits/test.txt',
                                  img_prefix='/home/share/LabServer/DATASET/stanford40/',
                                  img_scale=(224, 224),
                                  img_norm_cfg=img_norm_cfg,
                                  size_divisor=32,
                                  proposal_file='/home/share/LabServer/GLnet/stanford_test_bbox_new.pkl',
                                  test_mode=False, )
    train_dataset = StanfordDataset(ann_file='/home/share/LabServer/DATASET/stanford40/ImageSplits/train.txt',
                                    img_prefix='/home/share/LabServer/DATASET/stanford40/',
                                    img_scale=(224, 224),
                                    img_norm_cfg=img_norm_cfg,
                                    size_divisor=32,
                                    proposal_file='/home/share/LabServer/GLnet/stanford_train_bbox_new.pkl',
                                    test_mode=False, )
    train_loader = DataLoader(
        train_dataset,
        batch_size=10,
        shuffle=True,
        sampler=None,
        collate_fn=partial(collate, samples_per_gpu=2),
        num_workers=4)
    val_loader = DataLoader(
        val_dataset,
        batch_size=10,
        shuffle=False,
        sampler=None,
        collate_fn=partial(collate, samples_per_gpu=2),
        num_workers=4)
    #LOAD MODEL
    model = proposalModel()
    model = model.to("cuda:0")
    #submodel = resnet50(True)
    #submodel = submodel.to("cuda:1")

    #-------------   optim  -----------------#
    optimizer = optim.Adam(model.parameters(), lr=cfg.OPTIM.LR,)
    scheduler = ReduceLROnPlateau(optimizer,mode='min', factor=cfg.OPTIM.LR_DECAY, patience=1,
                 verbose=True, threshold=1e-4, threshold_mode='rel',
                                  cooldown=1, min_lr=0.00005, eps=1e-8)

    #-------------   train  ----------------#
    train(model, train_loader, val_loader,optimizer,scheduler,cfg.clone())

from data.stanford40.Stanford40_new import Stanford40
def prepare_stanford40(cfg):
    # ========= Preparing DataLoader =========#
    data = Stanford40(cfg.DATASET.STANFORD40, 'train')
    trn_dataloader = DataLoader(data, shuffle=True,
                                batch_size=cfg.DATASET.BATCH_SIZE,
                                num_workers=cfg.DATASET.NUM_WORKERS)

    data = Stanford40(cfg.DATASET.STANFORD40, 'test')
    val_dataloader = DataLoader(data, shuffle=True,
                                batch_size=cfg.DATASET.BATCH_SIZE,
                                num_workers=cfg.DATASET.NUM_WORKERS)
    return trn_dataloader, val_dataloader

from data.pascalVOC.basicDataloader import get_basicDataloader
def prepare_pascal(cfg):
    trn_dataloader, val_dataloader = get_basicDataloader()
    return  trn_dataloader, val_dataloader

#load model statedict
def load(model,path):
    ckp_path = path
    ckp = torch.load(ckp_path)

    pretrained_dict = ckp
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

if __name__ == '__main__':
    main()



