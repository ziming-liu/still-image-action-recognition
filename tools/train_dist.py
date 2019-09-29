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

"""
from data.stanford40.Stanford40Dataset import Stanford40
def prepare_stanford40():
    # ========= Preparing DataLoader =========#
    traindata = Stanford40('/home/share/LabServer/DATASET/stanford40', 'train')
    #trn_dataloader = DataLoader(data, shuffle=True,
    #                            batch_size=cfg.DATASET.BATCH_SIZE,
    #                            num_workers=cfg.DATASET.NUM_WORKERS)

    validedata = Stanford40('/home/share/LabServer/DATASET/stanford40', 'test')
    #val_dataloader = DataLoader(data, shuffle=True,
    #                            batch_size=cfg.DATASET.BATCH_SIZE,
    #                            num_workers=cfg.DATASET.NUM_WORKERS)
    return traindata, validedata
"""
#os.environ['RANK'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def batch_processor(model, data, train_mode):
    #img, label = data
    img = data['img'].data[0]
    label = data['gt_labels'].data[0]
    proposals = data['proposals'].data[0]
    imgmeta = data['img_meta'].data[0]
    gtbox = data['gt_bboxes'].data[0]
    label = label.cuda().squeeze(1)

    pred= model(img.cuda(),gtbox,proposals,label)

    pred = F.softmax(pred,1)
    loss = F.cross_entropy(pred, label)
    acc_top1, acc_top5 = accuracy(pred, label, topk=(1, 5))
    log_vars = OrderedDict()
    log_vars['loss'] = loss.item()
    log_vars['acc_top1'] = acc_top1.item()
    log_vars['acc_top5'] = acc_top5.item()
    outputs = dict(loss=loss, log_vars=log_vars, num_samples=img.size(0))
    return outputs


def get_logger(log_level):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
    logger = logging.getLogger()
    return logger


def init_dist(backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    #os.environ['MASTER_ADDR'] = '10.1.114.10'
    #os.environ['MASTER_PORT'] = '29500'
    #os.environ['RANK'] = '0'
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend,**kwargs)


def parse_args():
    parser = ArgumentParser(description='Train stanford40')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    logger = get_logger(cfg.log_level)

    # init distributed environment if necessary
    if args.launcher == 'none':
        dist = False
        logger.info('Disabled distributed training.')
    else:
        dist = True
        init_dist(**cfg.dist_params)
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        if rank != 0:
            logger.setLevel('ERROR')
        logger.info('Enabled distributed training.')

    # build datasets and dataloaders
    normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

    val_dataset =  StanfordDataset(ann_file='/home/share/LabServer/DATASET/stanford40/ImageSplits/test.txt',
                 img_prefix='/home/share/LabServer/DATASET/stanford40/',
                 img_scale=(224,224),
                 img_norm_cfg=img_norm_cfg,
                 size_divisor=32,
                 proposal_file='/home/share/LabServer/GLnet/stanford_test_bbox_new.pkl',
                 test_mode=False,)
    train_dataset = StanfordDataset(ann_file='/home/share/LabServer/DATASET/stanford40/ImageSplits/train.txt',
                 img_prefix='/home/share/LabServer/DATASET/stanford40/',
                 img_scale=(224,224),
                 img_norm_cfg=img_norm_cfg,
                 size_divisor=32,
                 proposal_file='/home/share/LabServer/GLnet/stanford_train_bbox_new.pkl',
                 test_mode=False,)
    #train_dataset,val_dataset = prepare_stanford40()
    print('traindataset {}'.format(len(train_dataset)))
    print('val dataset  {}'.format(len(val_dataset)))
    if dist:
        num_workers = cfg.data_workers
        assert cfg.batch_size % world_size == 0
        batch_size = cfg.batch_size // world_size
        train_sampler = DistributedSampler(train_dataset, world_size, rank)
        val_sampler = DistributedSampler(val_dataset, world_size, rank)
        shuffle = False
    else:
        num_workers = cfg.data_workers * len(cfg.gpus)
        batch_size = cfg.batch_size
        train_sampler = None
        val_sampler = None
        shuffle = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        collate_fn= partial(collate, samples_per_gpu=2),
        num_workers=num_workers)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn= partial(collate, samples_per_gpu=2),
        num_workers=num_workers)

    # build model
    from torchvision import models as M
    #model = M.resnet50(pretrained=True)
    model = proposalModel()
    #model = getattr(resnet_cifar, cfg.model)()
    if dist:
        model = DistributedDataParallel(
            model.cuda(), device_ids=[torch.cuda.current_device()])
    else:
        model = DataParallel(model, device_ids=cfg.gpus).cuda()

    # build runner and register hooks
    runner = Runner(
        model,
        batch_processor,
        cfg.optimizer,
        cfg.work_dir,
        log_level=cfg.log_level)
    runner.register_training_hooks(
        lr_config=cfg.lr_config,
        optimizer_config=cfg.optimizer_config,
        checkpoint_config=cfg.checkpoint_config,
        log_config=cfg.log_config)
    if dist:
        runner.register_hook(DistSamplerSeedHook())

    # load param (if necessary) and run
    if cfg.get('resume_from') is not None:
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from') is not None:
        runner.load_checkpoint(cfg.load_from)
    #dataloader 和 workflow对应
    runner.run([train_loader, val_loader], cfg.workflow, cfg.total_epochs)


if __name__ == '__main__':
    main()
