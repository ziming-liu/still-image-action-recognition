

from mmdet.models import newRPN
import torch
import torch.nn as nn
from .. import builder as builder_# ours
from mmdet.models import builder
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
import mmcv
from mmdet.models import BaseDetector
from mmdet.models import newRPN
from lib.model import SelfAttention

from mmdet.models.backbones import ResNet
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()

        cfg = mmcv.Config.fromfile('/home/liuziming/mmdetection/configs/rpn_r50_fpn_1x.py')
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True
        self.RPN = builder.build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

        self.backbone = ResNet(50,4,frozen_stages=1,)
        self.init_weights(pretrained='modelzoo://resnet50')
        self.relation = SelfAttention(2,256,256,256)
        self.fc = nn.Linear(256*2,40)
        self.avgpool  = nn.AdaptiveAvgPool2d((1,1))

    def init_weights(self,pretrained=None):
        super(Net3, self).init_weights(pretrained)
        load_checkpoint(self.RPN, '/home/share/LabServer/GLnet/MODELZOO/rpn_r50_fpn_2x_20181010-88a4a471.pth')
        self.backbone.init_weights(pretrained)
    def forward(self, x):

        with torch.no_grad():
            #return loss 控制训练/测试
            #参数 传入 basedetector 的forward
            result,roi_feats = self.RPN(return_loss=False, rescale=False, **x)
        roi_feats=self.avgpool(roi_feats)
        roi_feats = torch.mean(roi_feats,dim=1).view(roi_feats.size(0),-1)
        assert roi_feats.size(1) ==256
        global_feat = self.backbone(x)
        global_feat = self.avgpool(global_feat).view(global_feat.size(0),-1)
        assert  global_feat.size(1) ==256
        combine_feat = global_feat + roi_feats
        output = self.fc(combine_feat)
        return output

