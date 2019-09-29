#coding:utf-8

import os
import sys
import numpy as np
import time
import torch as t
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable as V
from torchnet import meter
sys.path.append('../')
from util.config import cfg
from util.visualize import Visualizer as vis
from data.prepareData import stanford40
from lib import model
#net = getattr(model,cfg.MODEL.NAME)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#create visulized env
vis_env = vis(cfg.SYSTEM.NAME, port=8097)
#measures created
AP = meter.APMeter()
mAP = meter.mAPMeter()
Loss_meter = meter.AverageValueMeter()
from lib.core.visImage import tensor_to_PIL,imshow,tensor_to_np,show_from_cv
import cv2
def show_keypoint(initimg,mask):
    #print(mask.shape)
    mask = mask.repeat(3,1,1)
    initimg_ = tensor_to_np(initimg)
    initimg_ = np.uint8(initimg_)
    mask_ = tensor_to_np(mask)
    map = cv2.resize(mask_, (224, 224))
    map = np.uint8(map)
    heatmap = cv2.applyColorMap(map, cv2.COLORMAP_HSV)
    #heatmap = heatmap.transpose(2,0,1)
    result = heatmap * 0.4+ initimg_* 0.7
    result = result.transpose(2,0,1)
    heatmap = heatmap.transpose(2,0,1)
    vis_env.image(result, win='sdj',opts={'title':'keypointmap'})
    vis_env.image(heatmap, win='sd12j', opts={'title': 'keypoint'})

def toshowimg(initimg,subimg,title):
    initimg_ = tensor_to_np(initimg)
    subimg_ = tensor_to_np(subimg)
    subimg_= cv2.resize(subimg_,(224,224))
    initimg_ = np.uint8(initimg_).transpose(2,0,1)
    subimg_ = np.uint8(subimg_).transpose(2,0,1)
    vis_env.image(initimg_, win='jhkjk', opts={'title': 'initimg'})
    vis_env.image(subimg_, win=title, opts={'title': title})

def train_epoch(net,traindata,optimizer,lossfunc):
    net.train()
    #indicator
    AP.reset()
    mAP.reset()
    Loss_meter.reset()

    keypoint = model.onekey_()
    keypoint.eval()
    keypoint.load('/home/share/LabServer/GLnet/ckp/keypoint_bestversion/keypoint_bestversion_200.pth')
    keypoint = keypoint.cuda()
    for batch,(data,label) in enumerate(traindata):
        initimg =data
        data = V(data.cuda())
        label = V(label.cuda())
        optimizer.zero_grad()
        _, _, mask, _ = keypoint(data)
        pred,toshow ,mask= net(data,mask)
        if batch %5==0:
            for ii,item in enumerate(toshow):
                toshowimg(data[0].detach().cpu(),data[0][:,item[0]:item[2],item[1]:item[3]].detach().cpu(),'sub'+str(ii))
            show_keypoint(initimg[0].detach().cpu(),mask[0].detach().cpu())
        #maybe we need a softmax
        pred = F.softmax(pred,1)
        loss = lossfunc(pred,label)
        Loss_meter.add(loss.detach().item())
        loss.backward()
        optimizer.step()

        one_hot = t.zeros_like(pred).cuda().scatter(1,label.view(-1, 1), 1)
        AP.add(pred.detach(), one_hot)
        mAP.add(pred.detach(), one_hot)

    return Loss_meter.value()[0], mAP.value()

def validate_epoch(net,valdata,lossfunc):
    net.eval()
    AP.reset()
    mAP.reset()
    Loss_meter.reset()
    keypoint = model.onekey_()
    keypoint.eval()
    keypoint.load('/home/share/LabServer/GLnet/ckp/keypoint_bestversion/keypoint_bestversion_200.pth')
    keypoint = keypoint.cuda()
    for batch,(data,label) in enumerate(valdata):
        data = V(data.cuda())
        label = V(label.cuda())
        _, _, mask, _ = keypoint(data)
        pred,_,_ = net(data,mask)
        pred = F.softmax(pred, 1)
        # maybe we need a softmax , too
        loss = lossfunc(pred, label)
        Loss_meter.add(loss.detach().item())

        one_hot = t.zeros_like(pred).cuda().scatter(1, label.view(-1, 1), 1)
        AP.add(pred.detach(), one_hot)
        mAP.add(pred.detach(), one_hot)

    return Loss_meter.value()[0], mAP.value()

def printresult(loss,accu,time,term='train'):
    print(  '-',term,'-loss: {}, accu: {} % , elapse: {} min'.format(loss,accu*100,time))
def train(net,traindata,valdata,optimizer=None,lossfunc=None,lr_strategy=None):
    valid_accus = []
    for epoch in range(cfg.TRAIN.EPOCHES):
        print('[epoch ',epoch,']')
        start = time.time()
        trainLoss, trainAccu = train_epoch(net,traindata,optimizer,lossfunc)
        timePoint = time.time()
        printresult(trainLoss,trainAccu,(timePoint-start)/60)

        valLoss, valAccu = validate_epoch(net,valdata,lossfunc)
        timePoint_ = time.time()
        printresult(valLoss,valAccu,(timePoint_-timePoint)/60,'validate')

        vis_env.plot_many({'train loss': trainLoss,'val_loss':valLoss})
        vis_env.plot_many({'train accuracy': trainAccu,'val_accuracy':valAccu})
        lr_strategy.step(epoch)
        #save
        valid_accus.extend( [valAccu])
        if valAccu>=max(valid_accus):
        #if epoch%2 ==0:
            model_name = cfg.SYSTEM.NAME + '_best.pth'#.format(epoch)
            print("saving ")
            net.save(model_name)
def main():
    if cfg.SYSTEM.UPDATE_CFG:
        cfg.merge_from_file(cfg.SYSTEM.CFG_FILE)
    cfg.freeze()#冻结参数
    vis_env.log('configs:\n {}'.format(cfg.clone()))
    #load data
    traindata,valdata = stanford40(cfg)
    #load model
    net = model.multi_two(True)
    net= net.cuda()
    optimizer = t.optim.Adagrad(net.parameters(), lr=cfg.OPTIM.LR)
    lr_strategy = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300, )
    lossfunc = t.nn.CrossEntropyLoss()

    train(net,traindata,valdata,optimizer,lossfunc,lr_strategy)


if __name__ == '__main__':
    #import fire
    #fire.Fire()
    main()
