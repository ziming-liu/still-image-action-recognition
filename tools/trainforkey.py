#coding:utf-8
################################
# used to train key point net  #
# get the key region for action#
################################
import os
import numpy as np
import time
import torch as t
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable as V
from torchvision import transforms as T
from torchnet import meter

from util.config import cfg
from util.visualize import Visualizer as vis
from data import stanford40,pascal2012
from lib import model# model file

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#create visulized env
vis_env = vis(cfg.SYSTEM.NAME, port=8097)
#measures created
Loss_meter = meter.AverageValueMeter()
AP = meter.APMeter()
mAP = meter.mAPMeter()

from lib.core.visImage import tensor_to_PIL,imshow,tensor_to_np,show_from_cv
import cv2
import matplotlib.pyplot as plt
def show_keypoint(initimg,mask):
    mask = mask.repeat(3,1,1)
    initimg_ = tensor_to_np(initimg)
    initimg_ = np.uint8(initimg_)
    mask_ = tensor_to_np(mask)
    mask_init = mask_
    mask_init = np.uint8(mask_init)
    mask_init = cv2.resize(mask_init,(224,224))
    map = cv2.resize(mask_, (224, 224))
    map = np.uint8(map)
    heatmap = cv2.applyColorMap(map, cv2.COLORMAP_HSV)
    #heatmap = heatmap.transpose(2,0,1)

    result = heatmap * 0.4+ initimg_* 0.7
    result = result.transpose(2,0,1)
    heatmap = heatmap.transpose(2,0,1)
    mask_init = mask_init.transpose(2,0,1)
    vis_env.image(result, win='sdj',opts={'title':'ssdf'})
    vis_env.image(heatmap, win='sd12j', opts={'title': 'sssdf'})
    vis_env.image(mask_init, win='mask',opts={'title':'mask'})



def train_epoch(net,traindata,optimizer,lossfunc):
    net.train()
    #indicator
    AP.reset()
    mAP.reset()
    Loss_meter.reset()

    for batch,(data,label) in enumerate(traindata):
        data = V(data.cuda())
        label = V(label.cuda())
        optimizer.zero_grad()

        pred,pred2, mask, logit = net(data)
        logit = F.softmax(logit,dim=1)
        loss_label = F.cross_entropy(logit,label)

        show_keypoint(data[0].detach().cpu(),mask[0].detach().cpu())
        #maybe we need a softmax
        target = t.ones(1)*10
        target = target.repeat(pred.size(0)).cuda()
        target2 = t.zeros(1)
        target2 = target2.repeat(pred2.size(0)).cuda()
        #print('pred {}'.format(pred))
        #print(pred.shape)
        #print(target.shape)
        loss1 = lossfunc(pred, target)
        loss2 = lossfunc(pred2,target2)
        #loss3 = lossfunc(pred3,target2)
        loss = loss1 + loss2
        loss_sum = loss*0.1 + loss_label

        print('loss123 {}-{}  losslabel {}'.format(loss1,loss2,loss_label))

        Loss_meter.add(loss_sum.detach().item())
        loss_sum.backward()
        optimizer.step()

        one_hot = t.zeros_like(logit).cuda().scatter(1,label.view(-1, 1), 1)
        AP.add(logit.detach(), one_hot)
        mAP.add(logit.detach(), one_hot)


    return Loss_meter.value()[0],mAP.value()

def validate_epoch(net,valdata,lossfunc):
    net.eval()
    AP.reset()
    mAP.reset()
    Loss_meter.reset()
    for batch,(data,label) in enumerate(valdata):
        data = V(data.cuda())
        label = V(label.cuda())

        pred,pred2, mask, logit = net(data)
        logit = F.softmax(logit, dim=1)
        loss_label = F.cross_entropy(logit, label)
        if batch % 5 == 2:
            #print('that is validation !!!')
            show_keypoint(data[0].detach().cpu(), mask[0].detach().cpu())
        # maybe we need a softmax
        target = t.ones(1)*6
        target = target.repeat(pred.size(0)).cuda()
        # print(pred.shape)
        # print(target.shape)
        loss = lossfunc(pred, target)

        loss_sum =loss_label
        Loss_meter.add(loss_sum.detach().item())

        one_hot = t.zeros_like(logit).cuda().scatter(1, label.view(-1, 1), 1)
        AP.add(logit.detach(), one_hot)
        mAP.add(logit.detach(), one_hot)
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
        vis_env.log(
            "Phase:{phase},Epoch:{epoch},AP:{AP},mAP:{mAP},train_loss:{loss}".format(
                phase="train", epoch=epoch, AP=AP.value(), mAP=mAP.value(), loss=Loss_meter.value()[0],
            ))
        valLoss, valAccu= validate_epoch(net,valdata,lossfunc)
        timePoint_ = time.time()
        printresult(valLoss,valAccu,(timePoint_-timePoint)/60,'valid')

        vis_env.plot_many({'train loss': trainLoss,'val_loss':valLoss})
        vis_env.plot_many({'train accuracy': trainAccu,'val_accuracy':valAccu})
        vis_env.log(
            "Phase:{phase},Epoch:{epoch}, AP:{AP},mAP:{mAP},val_loss:{loss}".format(
                phase="validation", epoch=epoch, AP=AP.value(), mAP=mAP.value(), loss=Loss_meter.value()[0]))

        #lr_strategy.step(epoch)
        #save
        valid_accus.extend( [valAccu])
        if valAccu>=max(valid_accus):
        #if epoch%2 ==0:
            model_name = cfg.SYSTEM.NAME + '_{}.pth'.format('best')
            print("saving ")
            net.save(model_name)
        #elif cfg.MODEL.SAVE_MODE == 'best':
        #    model_name = cfg.SYSTEM.NAME + '_best.pth'
        #    if valAccu >= max(valid_accus):
        #        net.save(model_name)

def main():
    if cfg.SYSTEM.UPDATE_CFG:
        cfg.merge_from_file(cfg.SYSTEM.CFG_FILE)
    cfg.freeze()#冻结参数
    vis_env.log('configs:\n {}'.format(cfg.clone()))
    #load data
    traindata,valdata = stanford40(cfg)
    #load model
    net = model.twokey_copy(True)#getattr(model,cfg.MODEL.NAME)
    #for param in net.layer1.parameters():
    #    param.requires_grad = False

    net = net.cuda()
    optimizer = t.optim.Adagrad(filter(lambda p: p.requires_grad, net.parameters()),lr=cfg.OPTIM.LR)
    lr_strategy = t.optim.lr_scheduler.CosineAnnealingLR(optimizer,50,)
    lossfunc = t.nn.SmoothL1Loss()

    train(net,traindata,valdata,optimizer,lossfunc,lr_strategy)


if __name__ == '__main__':
    #import fire
    #fire.Fire()
    main()

















