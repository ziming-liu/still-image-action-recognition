import torch
from torch.autograd import Variable
import time
import os
import sys

from util import AverageMeter, calculate_accuracy
from torchnet import meter
from utils.tricks import *
def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,vis,trainlogwindow):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    mmap = meter.mAPMeter()
    top = meter.ClassErrorMeter(topk=[1, 3, 5], accuracy=True)
    mmap.reset()
    top.reset()
    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        targets = targets.cuda()
        if type(inputs) is list:
            inputs = [Variable(inputs[ii]).cuda() for ii in range(len(inputs)) ]
        else:
            inputs = inputs.cuda()
            #inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, opt.DATASET.ALPHA, True)
            #inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
            inputs = Variable(inputs)
        #print(targets)
        targets = Variable(targets)

        outputs,context = model(inputs)
        #loss_func = mixup_criterion(targets_a, targets_b, lam)
        #loss = loss_func(criterion, outputs)
        loss = criterion(outputs, targets)
        #print(outputs.shape)
        #print(targets)
        acc = calculate_accuracy(outputs, targets)
        one_hot = torch.zeros_like(outputs).cuda().scatter_(1, targets.view(-1, 1), 1)
        mmap.add(outputs.detach(), one_hot.detach())
        top.add(outputs.detach(), targets.detach())
        losses.update(loss.data.item(), targets.detach().size(0))
        accuracies.update(acc, targets.detach().size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        vis.text(
            "gpu{}, epoch: {},batch:{},iter: {},loss: {},acc:{},lr: {}\n".format(torch.cuda.current_device(),epoch, i + 1,(epoch - 1) * len(data_loader) + (i + 1),losses.val, \
                                                  accuracies.val,optimizer.param_groups[0]['lr'])
                                                    ,win=trainlogwindow,append=True)

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
                    top=top.value()))
    vis.text("total:\n gpu:{} epoch: {},loss: {},lr: {}, accu:{},mAP:{}, top135 {}\n".
            format(torch.cuda.current_device(),epoch,losses.avg,optimizer.param_groups[0]['lr'],accuracies.avg,mmap.value(),top.value()),
             win = trainlogwindow, append = True)
    if torch.cuda.current_device()== 0:
        print("saveing ckp ########################################")
        if epoch % opt.MODEL.CKP_DURING == 0:
            save_file_path = os.path.join(opt.MODEL.RESULT,opt.MODEL.NAME,
                                          'save_{}.pth'.format(epoch))
            if not os.path.exists(os.path.join(opt.MODEL.RESULT,opt.MODEL.NAME)):
                os.makedirs(os.path.join(opt.MODEL.RESULT,opt.MODEL.NAME))
            states = {
                'epoch': epoch + 1,
                'arch': opt.MODEL.NAME,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_file_path)
    return losses.avg, mmap.value()
