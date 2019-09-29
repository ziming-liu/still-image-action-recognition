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
from util.visualize import Visualizer
from util.show_masked_image import show_masked_image
from mmcv.runner import save_checkpoint,load_checkpoint

#cfg.merge_from_file("config/un_att_pascal_0001.yaml")
cfg.freeze()  # 冻结参数
vis = Visualizer(cfg.MODEL.NAME, port=8097)
AP = meter.APMeter()
mAP = meter.mAPMeter()
Loss_meter = meter.AverageValueMeter()
os.environ["CUDA_VISIBLE_DEVICES"] ="0"

def visualize_func(result):
    pass

def train_epoch(model, training_data, optimizer):
    ''' Epoch operation in training phase'''
    AP.reset()
    mAP.reset()
    Loss_meter.reset()
    model.train()

    for batch_idx, (data, target) in enumerate(training_data):
        data = V(data.cuda())
        target = V(target.cuda())
        # forward
        optimizer.zero_grad()
        result = model(data)
        visualize_func(result)
        #loss = model.compute_loss(result,target)
        loss = F.cross_entropy(result,target)
        loss.backward()
        optimizer.step()
        # record
        pred = result
        Loss_meter.add(loss.detach().item())

        one_hot = torch.zeros_like(pred).cuda().scatter_(1, target.view(-1, 1), 1)
        AP.add(pred.detach(), one_hot)
        mAP.add(pred.detach(), one_hot)

    return Loss_meter.value()[0], mAP.value()


def eval_epoch(model, validation_data):
    model.eval()
    AP.reset()
    mAP.reset()
    Loss_meter.reset()

    for batch_idx, (data, target) in enumerate(validation_data):
        data = V(data.cuda())
        target = V(target.cuda())
        result = model(data)
        visualize_func(result)
        loss = model.compute_loss(result, target)
        # record
        pred = result
        Loss_meter.add(loss.detach().item())
        one_hot = torch.zeros_like(pred).cuda().scatter(1, target.view(-1, 1), 1)
        AP.add(pred.detach(), one_hot)
        mAP.add(pred.detach(), one_hot)

    return Loss_meter.value()[0], mAP.value()


def train(model, training_data, validation_data, optimizer, scheduler, cfg):
    valid_accus = []
    for epoch_i in range(cfg.TRAIN.EPOCHES):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, optimizer, )
        print('  - (Training) ,loss: {loss:3.3f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(loss=train_loss,
                                                 accu=100 * train_accu,
                                                 elapse=(time.time() - start) / 60))
        vis.log(
        "Phase:{phase},Epoch:{epoch},AP:{AP},mAP:{mAP},train_loss:{loss}".format(
        phase="train", epoch=epoch_i, AP=AP.value(), mAP=mAP.value(), loss=Loss_meter.value()[0],))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, )
        print('  - (Validation)  ,loss: {loss:3.3f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(loss=valid_loss,
                                                 accu=100 * valid_accu,
                                                 elapse=(time.time() - start) / 60))
        scheduler.step(valid_loss)  # 更新学习率

        vis.plot_many_stack({'val loss': valid_loss,'train loss': train_loss})
        vis.plot_many_stack({'val accuracy': valid_accu,'train accuracy': train_accu})
        vis.log(
            "Phase:{phase},Epoch:{epoch}, AP:{AP},mAP:{mAP},val_loss:{loss}".format(
                phase="validation", epoch=epoch_i, AP=AP.value(), mAP=mAP.value(), loss=Loss_meter.value()[0],
            ))

        valid_accus += [valid_accu]  # 存储了所有epoch的准确率
        if valid_accu >= max(valid_accus):
            save_checkpoint(model, cfg.MODEL.SAVE_IN+cfg.MODEL.NAME+'.pth')

def pretrain(model, training_data, optimizer, scheduler, cfg):
    valid_accus = []
    for epoch_i in range(cfg.TRAIN.EPOCHES):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, optimizer, )
        print('  - (Training) ,loss: {loss:3.3f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(loss=train_loss,
                                                 accu=100 * train_accu,
                                                 elapse=(time.time() - start) / 60))
        vis.log(
            "Phase:{phase},Epoch:{epoch},AP:{AP},mAP:{mAP},train_loss:{loss}".format(
                phase="train", epoch=epoch_i, AP=AP.value(), mAP=mAP.value(), loss=Loss_meter.value()[0], ))

        scheduler.step(train_loss)  # 更新学习率

        vis.plot_many({ 'train loss': train_loss})
        vis.plot_many({ 'train accuracy': train_accu})

        valid_accus += [train_accu]  # 存储了所有epoch的准确率
        if train_accu >= max(valid_accus):
            save_checkpoint(model, cfg.MODEL.SAVE_IN+cfg.MODEL.NAME+'.pth')
def main():

    print(cfg)
    vis.log('configs:\n {}'.format(cfg.clone()))
    from data import StanfordBox,PascalBox,BU101Dataset
    # dataset  = PascalBox()
    train_dataset = PascalBox(cfg, train=True, val=False)
    val_dataset = PascalBox(cfg, train=False, val=True)
    training_data = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_data = DataLoader(val_dataset, batch_size=10)
    BUdataloader = BU101Dataset()

    #------------------------------------------
    # from lib.model.resnet import resnet50
    from lib.graphbased.attentionGCN import attentionGCN
    from lib.channelmax2attention.backbone_model import backbone_model
    from torchvision.models import resnet50
    model = resnet50(pretrained=True)
    num = model.fc.in_features
    model.fc = torch.nn.Linear(num,101)
    #fileter grad
    #for params in model.backbone.parameters():
    #    params.requires_grad = False
    model = model.cuda()
    #load_checkpoint(model, "MODELZOO/resnet50.pth")
    #-------------------------------------------
    # optimizer = optim.Adagrad([
    #       {'params': model.backbone.parameters(),'lr':1e-5},
    #       {'params': model.attention.parameters(), 'lr': 1e-4},
    #       {'params': model.GCN.parameters(),'lr':1e-4}
    #       ], lr=1e-4,)
    optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=cfg.OPTIM.LR_DECAY, patience=1,
                                  verbose=True, threshold=1e-4, threshold_mode='rel',
                                  cooldown=1, min_lr=0.00005, eps=1e-8)

    #train(model, training_data, validation_data, optimizer, scheduler, cfg.clone())
    pretrain(model,BUdataloader,optimizer,scheduler,cfg.clone())

if __name__ == '__main__':
    # import fire
    # fire.Fire()
    main()


