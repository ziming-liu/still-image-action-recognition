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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#create visulized env
vis_env = vis(cfg.SYSTEM.NAME, port=8097)
#measures created
AP = meter.APMeter()
mAP = meter.mAPMeter()
Loss_meter = meter.AverageValueMeter()
from lib.core.visImage import tensor_to_PIL,imshow,tensor_to_np,show_from_cv
import cv2

def inverse_normalize(img):
    #if opt.caffe_pretrain:
    #    img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
    #    return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255



def show_keypoint(initimg,mask):
    #print(mask.shape)
    mask = mask.repeat(3,1,1)
    #initimg_ = tensor_to_np(initimg)
    initimg = initimg.transpose(1,2,0)
    initimg_ = np.uint8(initimg)
    mask_ = tensor_to_np(mask)
    map = cv2.resize(mask_, (224, 224))
    map = np.uint8(map)
    heatmap = cv2.applyColorMap(map, cv2.COLORMAP_HSV)
    #heatmap = heatmap.transpose(2,0,1)
    result = heatmap * 0.4+ initimg_* 0.7
    result = result.transpose(2,0,1)
    heatmap = heatmap.transpose(2,0,1)
    initimg_ = initimg.transpose(2,0,1)
    vis_env.image(result, win='sdj',opts={'title':'keypointmap'})
    vis_env.image(initimg_, win='sd12j', opts={'title': 'keypoint'})

def toshowimg(initimg,subimg,title):
    #initimg_ = tensor_to_np(initimg)
    #subimg_ = tensor_to_np(subimg)
    subimg_ = subimg.transpose(1,2,0)
    subimg_= cv2.resize(subimg_,(224,224))
    #initimg_ = np.uint8(initimg_).transpose(2,0,1)
    subimg_ = np.uint8(subimg_).transpose(2,0,1)
    #vis_env.image(initimg_, win='jhkjk', opts={'title': 'initimg'})
    vis_env.image(subimg_, win=title, opts={'title': title})

def train_epoch(net,traindata,optimizer):
    net.train()
    #indicator
    AP.reset()
    mAP.reset()
    Loss_meter.reset()

    keypoint = model.onekey_()
    keypoint.eval()
    keypoint.load('../ckp/keypoint_bestversion/keypoint_bestversion_200.pth')
    #keypoint.load('../ckp/keypointT2/keypointT2_best.pth')
    keypoint = keypoint.cuda()
    for batch,(data,label) in enumerate(traindata):
        initimg =data
        data = data.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        _, _, mask, _ = keypoint(data)
        data = V(data)
        label = V(label)

        pred,toshow ,mask= net(data,mask.detach())
        org_img = inverse_normalize(data[0].detach().cpu().numpy())
        if batch %5==0:
            for ii,item in enumerate(toshow):
                toshowimg(org_img,org_img[:,item[0].item():item[2].item(),item[1].item():item[3].item()],'sub'+str(ii))
            show_keypoint(org_img,mask[0].detach().cpu())
        #maybe we need a softmax
        pred = F.softmax(pred,1)
        loss = F.cross_entropy(pred,label)
        Loss_meter.add(loss.detach().item())
        loss.backward()
        optimizer.step()

        one_hot = t.zeros_like(pred).cuda().scatter(1,label.view(-1, 1), 1)
        AP.add(pred.detach(), one_hot)
        #print('pred {}'.format(pred.detach()))
        #print('one hot {}'.format(one_hot))
        mAP.add(pred.detach(), one_hot)

    return Loss_meter.value()[0], mAP.value()

def validate_epoch(net,valdata):
    net.eval()
    AP.reset()
    mAP.reset()
    Loss_meter.reset()
    keypoint = model.onekey_()
    keypoint.eval()
    #keypoint.load('../ckp/keypointT2/keypointT2_best.pth')
    keypoint.load('../ckp/keypoint_bestversion/keypoint_bestversion_200.pth')
    keypoint = keypoint.cuda()
    for batch,(data,label) in enumerate(valdata):
        data = data.cuda()
        label = label.cuda()
        _, _, mask, _ = keypoint(data)
        data = V(data)
        label = V(label)
        pred,_,_ = net(data,mask.detach())
        pred = F.softmax(pred, 1)
        # maybe we need a softmax , too
        loss = F.cross_entropy(pred, label)
        Loss_meter.add(loss.detach().item())

        one_hot = t.zeros_like(pred).cuda().scatter(1, label.view(-1, 1), 1)
        AP.add(pred.detach(), one_hot)
        mAP.add(pred.detach(), one_hot)

    return Loss_meter.value()[0], mAP.value()

def printresult(loss,accu,time,term='train'):
    print(  '-',term,'-loss: {}, accu: {} % , elapse: {} min'.format(loss,accu*100,time))
def train(net,traindata,valdata,optimizer=None,lr_strategy=None):
    valid_accus = []
    for epoch in range(cfg.TRAIN.EPOCHES):
        print('[epoch ',epoch,']')
        start = time.time()
        trainLoss, trainAccu = train_epoch(net,traindata,optimizer)
        timePoint = time.time()
        printresult(trainLoss,trainAccu,(timePoint-start)/60)
        vis_env.log(
            "Phase:{phase},Epoch:{epoch},AP:{AP},mAP:{mAP},train_loss:{loss}".format(
                phase="train", epoch=epoch, AP=AP.value(), mAP=mAP.value(), loss=Loss_meter.value()[0],
            ))

        valLoss, valAccu = validate_epoch(net,valdata)
        timePoint_ = time.time()
        printresult(valLoss,valAccu,(timePoint_-timePoint)/60,'validate')

        vis_env.plot_many({'train loss': trainLoss,'val_loss':valLoss})
        vis_env.plot_many({'train accuracy': trainAccu,'val_accuracy':valAccu})
        vis_env.log(
            "Phase:{phase},Epoch:{epoch}, AP:{AP},mAP:{mAP},val_loss:{loss}".format(
                phase="validation", epoch=epoch, AP=AP.value(), mAP=mAP.value(), loss=Loss_meter.value()[0]))

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
    #traindata,valdata = stanford40(cfg)
    from data.prepareData import BU101Dataset
    traindata,valdata = BU101Dataset()
    print(len(traindata))
    print(len(valdata))
    #load model
    net = model.multi_two(pretrained=True)
    net= net.cuda()
    optimizer = t.optim.Adagrad(net.parameters(), lr=cfg.OPTIM.LR)
    lr_strategy = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300, )
    #lossfunc = t.nn.CrossEntropyLoss()

    train(net,traindata,valdata,optimizer,lr_strategy)


if __name__ == '__main__':
    #import fire
    #fire.Fire()
    main()
