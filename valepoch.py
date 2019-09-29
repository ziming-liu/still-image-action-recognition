import torch
from torch.autograd import Variable
import time
import sys

from util import AverageMeter, calculate_accuracy
from torchnet import meter
#from utils.visualize import Visualizer
import numpy as np

#vis = Visualizer("video", port=8097)

num = 9
import cv2
def inverse_normalize(img):
    #if opt.caffe_pretrain:
    #    img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
    #    return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255
def show_keypoint(initimg,mask,vis,title=None):
    #print(mask.shape)
    A = mask.cpu().numpy()
    #mask = mask.repeat(3,1,1)
    initimg = initimg.transpose(1,2,0)
    initimg_ = np.uint8(initimg)
    initimg_ = cv2.resize(initimg_,(224,160))

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
            columnnames=['f'+str(i+1) for i in range(0,num,1)],
            rownames=['f'+str(i) for i in range(num,0,-1)],
           colormap='Electric',),
        win = "sdfas",
    )
    #vis.image(result, win='sdj',opts={'title':'keypointmap'})
    vis.image(initimg_, win='sd12j'+title, opts={'title': 'keypoint'+title})


def val_epoch(epoch, data_loader, model, criterion, opt, vis,vallogwindow):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    mmap = meter.mAPMeter()
    AP = meter.APMeter()
    top = meter.ClassErrorMeter(topk=[1, 3, 5], accuracy=True)
    mmap.reset()
    AP.reset()
    top.reset()
    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        if type(inputs) is list:
            inputs = [Variable(inputs[ii].cuda()) for ii in range(len(inputs))]
        else:
            inputs = Variable(inputs.cuda())
        targets = targets.cuda()
        with torch.no_grad():
            #inputs = Variable(inputs)
            targets = Variable(targets)
            outputs ,context= model(inputs)
            #if i %5==0:
            #for jj in range(num):
            #    org_img = inverse_normalize(inputs[0,jj,:,:,:].detach().cpu().numpy())
            #    show_keypoint(org_img, context[0].detach().cpu(),vis=vis,title = str(jj+1))

            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.data.item(), targets.detach().size(0))
            accuracies.update(acc, targets.detach().size(0))
            one_hot = torch.zeros_like(outputs).cuda().scatter_(1, targets.view(-1, 1), 1)
            mmap.add(outputs.detach(), one_hot.detach())
            top.add(outputs.detach(), targets.detach())
            AP.add(outputs.detach(), one_hot.detach())
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
              'mmap {mmap}\t'
              'top1 3 5: {top}\t'.format(
            epoch,
            i + 1,
            len(data_loader),
            batch_time=batch_time,
            data_time=data_time,
            loss=losses,
            acc=accuracies,
            mmap=mmap.value(),
            top=top.value() ))

    vis.text("gpu:{}, epoch: {},loss: {},accu:{},mAP:{}, top135 {}\nAP:{}".format(torch.cuda.current_device(),epoch,losses.avg,accuracies.avg,mmap.value(),top.value(),AP.value())
    ,win=vallogwindow,append=True)
    #exit()
    #if epoch==10:
    #    exit()
    return losses.avg, mmap.value()



