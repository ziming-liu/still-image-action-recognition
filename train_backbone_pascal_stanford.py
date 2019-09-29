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
import cv2
from util.show_masked_image import tensor_to_np
import numpy as np
#cfg.merge_from_file("config/un_att_pascal_0001.yaml")
cfg.freeze()  # 冻结参数
vis = Visualizer("newvis", port=8097)
AP = meter.APMeter()
mAP = meter.mAPMeter()
top3 = meter.ClassErrorMeter(topk=[1,3,5],accuracy=True)

Loss_meter = meter.AverageValueMeter()
os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2"
num = 30
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
            columnnames=['f'+str(i+1) for i in range(0,num,1)],
            rownames=['f'+str(i) for i in range(num,0,-1)],
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

def train_epoch(model,baseline, training_data, optimizer):
    ''' Epoch operation in training phase'''
    AP.reset()
    mAP.reset()
    Loss_meter.reset()
    top3.reset()
    model.train()

    for batch_idx, (boxes,data, target) in enumerate(training_data):
        data = V(data.cuda())
        target = V(target.cuda())
        allrois = []
        for ii in range(boxes.shape[0]):
            box = boxes[ii]
            l,f = box.shape
            idx = ii* torch.ones(l,1)
            rois = torch.cat((idx,box),1)
            assert rois.shape[1] == 5
            allrois.append(rois)
        allrois = torch.cat(allrois,0).cuda()
        #print(allrois.shape)
        # forward
        optimizer.zero_grad()
        result,context= model(data,allrois)
        #if batch_idx %5==0:
        org_img = inverse_normalize(data[0].detach().cpu().numpy())
        show_keypoint(org_img, context[0].detach().cpu())


        #result2 = baseline(data)
        #result = result
        #result = model(data)
        visualize_func(result)
        #print(target)
        #print(result.shape)
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

        top3.add(pred.detach(), target)
    print("top3 : {}".format(top3.value()))
    return Loss_meter.value()[0], mAP.value()


def eval_epoch(model, baseline,validation_data):
    model.eval()
    AP.reset()
    mAP.reset()
    top3.reset()
    Loss_meter.reset()

    for batch_idx, (boxes,data, target) in enumerate(validation_data):
        data = V(data.cuda())
        target = V(target.cuda())
        allrois = []
        for ii in range(boxes.shape[0]):
            box = boxes[ii]
            l, f = box.shape
            idx = ii * torch.ones(l, 1)
            rois = torch.cat((idx, box), 1)
            assert rois.shape[1] == 5
            allrois.append(rois)
        allrois = torch.cat(allrois, 0).cuda()
        #print(allrois.shape)
        result,context = model(data,allrois)
        #result2 = baseline(data)
        #result = result + result2
        #result = model(data)

        visualize_func(result)
        loss = F.cross_entropy(result,target)
        # record
        pred = result
        Loss_meter.add(loss.detach().item())
        one_hot = torch.zeros_like(pred).cuda().scatter(1, target.view(-1, 1), 1)
        AP.add(pred.detach(), one_hot)
        mAP.add(pred.detach(), one_hot)
        top3.add(pred.detach(), target)
    print("top3 : {}".format(top3.value()))
    return Loss_meter.value()[0], mAP.value()

def test(model,model2, validation_data):
    model.eval()
    #model2.eval()
    AP.reset()
    mAP.reset()
    top3.reset()
    Loss_meter.reset()
    start = time.time()
    print("testing ...")
    for batch_idx, (boxes,data, target) in enumerate(validation_data):
        data = V(data.cuda())
        #img = img.cuda()
        target = V(target.cuda())
        allrois = []
        for ii in range(boxes.shape[0]):
            box = boxes[ii]
            l, f = box.shape
            idx = ii * torch.ones(l, 1)
            rois = torch.cat((idx, box), 1)
            assert rois.shape[1] == 5
            allrois.append(rois)
        allrois = torch.cat(allrois, 0).cuda()
        #print(allrois.shape)
        result,context = model(data,allrois)
        #result = model(data)
        #result = model2(data)
        #result = result+result2
        visualize_func(result)
        loss = F.cross_entropy(result,target)
        # record
        pred = result
        Loss_meter.add(loss.detach().item())
        one_hot = torch.zeros_like(pred).cuda().scatter(1, target.view(-1, 1), 1)
        AP.add(pred.detach(), one_hot)
        mAP.add(pred.detach(), one_hot)
        top3.add(pred.detach(),target)
    valid_loss,valid_accu = Loss_meter.value()[0], mAP.value()
    print('  - (Validation)  ,loss: {loss:3.3f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(loss=valid_loss,
                                                 accu=100 * valid_accu,
                                                 elapse=(time.time() - start) / 60))
    vis.log(
        "Phase:{phase},Epoch:{epoch}, AP:{AP},mAP:{mAP},val_loss:{loss}".format(
            phase="validation", epoch=0, AP=AP.value(), mAP=mAP.value(), loss=Loss_meter.value()[0],
        ))
    print("top3 {}".format(top3.value()))
    return Loss_meter.value()[0], mAP.value()


def train(model, baseline,training_data, validation_data, optimizer, scheduler, cfg,args):
    valid_accus = []
    for epoch_i in range(cfg.TRAIN.EPOCHES):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, baseline,training_data, optimizer, )
        print('  - (Training) ,loss: {loss:3.3f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(loss=train_loss,
                                                 accu=100 * train_accu,
                                                 elapse=(time.time() - start) / 60))
        vis.log(
        "Phase:{phase},Epoch:{epoch},AP:{AP},mAP:{mAP},train_loss:{loss}".format(
        phase="train", epoch=epoch_i, AP=AP.value(), mAP=mAP.value(), loss=Loss_meter.value()[0],))

        start = time.time()
        #if epoch_i %3 ==0:
        valid_loss, valid_accu = eval_epoch(model,baseline, validation_data, )

        vis.log(
            "Phase:{phase},Epoch:{epoch}, AP:{AP},mAP:{mAP},val_loss:{loss}".format(
                phase="validation", epoch=epoch_i, AP=AP.value(), mAP=mAP.value(), loss=Loss_meter.value()[0],
            ))
        #scheduler.step(valid_loss)  # 更新学习率
        #if epoch_i == 4:
        #    for param_group in optimizer.param_groups:
        #        param_group['lr'] = 1e-4
        print('  - (Validation)  ,loss: {loss:3.3f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(loss=valid_loss,
                                                 accu=100 * valid_accu,
                                                 elapse=(time.time() - start) / 60))

        vis.plot_many_stack({'val accuracy': valid_accu, 'train accuracy': train_accu})
        vis.plot_many_stack({'val loss': valid_loss,'train loss': train_loss})

        if torch.cuda.current_device() == 0:
            valid_accus += [valid_accu]  # 存储了所有epoch的准确率
            if valid_accu >= max(valid_accus):
                save_checkpoint(model, cfg.MODEL.SAVE_IN + cfg.MODEL.NAME +'_gpu0'+ '.pth')
        if torch.cuda.current_device() == 1:
            valid_accus += [valid_accu]  # 存储了所有epoch的准确率
            if valid_accu >= max(valid_accus):
                save_checkpoint(model, cfg.MODEL.SAVE_IN + cfg.MODEL.NAME  +'_gpu1'+ '.pth')
        if torch.cuda.current_device() == 2:
            valid_accus += [valid_accu]  # 存储了所有epoch的准确率
            if valid_accu >= max(valid_accus):
                save_checkpoint(model, cfg.MODEL.SAVE_IN + cfg.MODEL.NAME  +'_gpu2'+ '.pth')


def main():
    print(cfg)
    vis.log('configs:\n {}'.format(cfg.clone()))

    #------------------------------------------
    # from lib.model.resnet import resnet50
    from lib.graphbased.attentionGCN import attentionGCN
    from lib.channelmax2attention.backbone_model import backbone_model
    from torchvision.models import resnet50,resnet18,resnet34,resnet101,vgg19_bn
    #model = vgg19_bn(pretrained=True)
    from global_branch_cnn import resnet50 as globalcnn
    #from context_cnn import resnet50 as contextcnn
    from context_cnn_simple import resnet50 as contextcnnsimple
    from contextcnn_input import resnet50 as contextcnninput
    from torchvision.models import  resnet50 as stdres
    from meshcnn_fromglobal import rgb_resnet50
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument(
        '--resume',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--save',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)

    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')

    model = rgb_resnet50(pretrained=True).cuda()

    #for params in model.layer1.parameters():
    #    params.requires_grad = False
    #for params in model.layer2.parameters():
    #    params.requires_grad = False
    #for params in model.layer3.parameters():
    #    params.requires_grad = False
    #for params in model.layer4.parameters():
    #    params.requires_grad = False
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      output_device=args.local_rank)
    #load_checkpoint(model, "MODELZOO/newdata_mesh9_1block_local+global_gpu0.pth")
    #model = stdres(pretrained=True)
    #model2 = globalcnn(pretrained=True,channel=3,nb_classes=40)
    #model2 = globalcnn()
    #model = vgg19_bn(pretrained=True)
    #num = model.fc.in_features
    #model.fc = torch.nn.Linear(num,40)
    #num = model2.fc.in_features
    #model2.fc = torch.nn.Linear(num, 40)

    #load_checkpoint(model2, "MODELZOO/globalcnn.pth")
    #num = model.classifier[-1].in_features
    #model.classifier[-1] = torch.nn.Linear(num, 11)
    #fileter grad


    model2 = None
    #model2 = model2.cuda()
    from global_branch_cnn import resnet50
    #from mmcv.runner import load_checkpoint
    #baseline = resnet50(pretrained=True, nb_classes=40, channel=3)
    #load_checkpoint(model, "MODELZOO/newdata_mesh9_1block_global_gpu1.pth")

    load_checkpoint(model, "MODELZOO/newdata_mesh9_1block_locao+global_gpu0.pth")
    #baseline = baseline.cuda()
    #baseline.eval()
    baseline =None
    #-------------------------------------------
    # optimizer = optim.Adagrad([
    #       {'params': model.backbone.parameters(),'lr':1e-5},
    #       {'params': model.attention.parameters(), 'lr': 1e-4},
    #       {'params': model.GCN.parameters(),'lr':1e-4}
    #       ], lr=1e-4,)
    optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=1e-3)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=cfg.OPTIM.LR_DECAY, patience=1,
                                  verbose=True, threshold=1e-5, threshold_mode='rel',
                                  cooldown=1, min_lr=0.00005, eps=1e-8)
    from data import StanfordBox, PascalBox, BU101Dataset
    # dataset  = PascalBox()
    train_dataset = StanfordBox(cfg, train=True, val=False, crop=False)
    val_dataset = StanfordBox(cfg, train=False, val=True, crop=False)
    trainsampler = torch.utils.data.DistributedSampler(train_dataset)
    valsampler = torch.utils.data.DistributedSampler(val_dataset)
    training_data = DataLoader(train_dataset, batch_size=20, num_workers=4,sampler= trainsampler )
    validation_data = DataLoader(val_dataset, batch_size=15, num_workers=4, sampler=valsampler)
    #train(model,baseline, training_data, validation_data, optimizer, scheduler, cfg.clone(),args)
    test(model,model2,validation_data)
if __name__ == '__main__':
    # import fire
    # fire.Fire()
    main()


