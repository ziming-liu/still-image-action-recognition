#coding:utf-8
import os
import time
from  torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable as V

from torchnet import meter
from util.config import cfg
from util.visualize import Visualizer

#create visulized env
vis = Visualizer(cfg.SYSTEM.NAME, port=8097)
#measures created
AP = meter.APMeter()
mAP = meter.mAPMeter()
Loss_meter = meter.AverageValueMeter()
#set cuda env
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def train_epoch(model, training_data, optimizer):
    ''' Epoch operation in training phase'''
    AP.reset()
    mAP.reset()
    Loss_meter.reset()
    model.train()

    for batch_idx, (data, target) in enumerate(training_data):

        data = data.cuda()
        target = target.cuda()

        data = V(data)

        target = V(target)
        #forward
        optimizer.zero_grad()
        pred= model(data)

        pred  = F.softmax(pred,1)
        #backward
        loss = F.cross_entropy(pred,target)

        Loss_meter.add(loss.detach().item())#转化为了numpy类型

        loss.backward()
        #optimize
        optimizer.step()

        #calculate acc
        one_hot = torch.zeros_like(pred).cuda().scatter(1, target.view(-1, 1), 1)

        AP.add(pred.detach(), one_hot)
        mAP.add(pred.detach(), one_hot)

    return Loss_meter.value()[0], mAP.value()

def eval_epoch( model, validation_data):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    AP.reset()
    mAP.reset()
    Loss_meter.reset()

    for batch_idx, (data, target) in enumerate(validation_data):
        data = data.cuda()
        target = target.cuda()
        data = V(data)
        target = V(target)

        # forward
        pred = model(data)

        pred = F.softmax(pred, 1)
        # backward
        loss = F.cross_entropy(pred, target)

        #one_hot = torch.zeros_like(pred).cuda().scatter(1, target.view(-1, 1), 1)

        one_hot = torch.zeros_like(pred).cuda().scatter(1, target.view(-1, 1), 1)

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
        scheduler.step(valid_loss)#更新学习率
 
        vis.plot_many({'val loss': valid_loss})
        vis.plot_many({'val accuracy': valid_accu})
        vis.log(
            "Phase:{phase},Epoch:{epoch}, AP:{AP},mAP:{mAP},val_loss:{loss}".format(
                phase="validation", epoch=epoch_i, AP=AP.value(), mAP=mAP.value(), loss=Loss_meter.value()[0],
            ))


        valid_accus += [valid_accu]#存储了所有epoch的准确率
        """
        model_state_dict = model.state_dict()
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
        """


def main():
    ''' Main function '''
    if cfg.SYSTEM.UPDATE_CFG:
        cfg.merge_from_file(cfg.SYSTEM.CFG_FILE)
    cfg.freeze()#冻结参数
    print(cfg)

    vis.log('configs:\n {}'.format(cfg.clone()))
    #LOAD DATA
    training_data, validation_data = prepare_stanford40(cfg.clone())
    #training_data, validation_data = prepare_pascal(cfg.clone())
    from lib.model.resnet import resnet50,resnet18
    from lib import model
    #LOAD MODEL
    model = model.multi_two(pretrained=True)

    model = model.cuda()

    #-------------   optim  -----------------#
    optimizer = optim.Adagrad(model.parameters(), lr=cfg.OPTIM.LR,)
    scheduler = ReduceLROnPlateau(optimizer,mode='min', factor=cfg.OPTIM.LR_DECAY, patience=1,
                 verbose=True, threshold=1e-4, threshold_mode='rel',
                                  cooldown=1, min_lr=0.00005, eps=1e-8)

    #-------------   train  ----------------#
    train(model, training_data, validation_data,optimizer,scheduler,cfg.clone())



from data.stanford40.Stanford40Dataset import Stanford40
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



