import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import (delta2bbox, multiclass_nms, bbox_target,
                        weighted_cross_entropy, weighted_smoothl1, accuracy)


class ActionHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 with_avg_pool=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=40,
                 reg_class_agnostic=False):
        super(ActionHead, self).__init__()

        self.with_avg_pool = with_avg_pool

        self.roi_feat_size = roi_feat_size
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.reg_class_agnostic = reg_class_agnostic

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(roi_feat_size)  # n * 256 * 7 * 7
        else:
            in_channels *= (self.roi_feat_size * self.roi_feat_size)

        self.debug_imgs = None

        self.fc = nn.Linear(in_channels* 2 ,num_classes)

    def init_weights(self):

        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x, globalfeat):#x 是roifeats
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        globalfeat = F.adaptive_avg_pool2d(globalfeat,(1,1)).view(globalfeat.size(0),-1)
        assert globalfeat.size == x.size
        feat = torch.cat((globalfeat,x),1)
        pred = self.fc(feat)
        return pred


    def loss(self, pred, labels, ):
        losses = dict()

        return F.cross_entropy(pred,labels)

