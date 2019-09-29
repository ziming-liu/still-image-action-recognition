import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import (delta2bbox, multiclass_nms, bbox_target,
                        weighted_cross_entropy, weighted_smoothl1, accuracy)


class Head(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 with_avg_pool=True,
                 num_rois = 1000,
                 feat_size=7,
                 in_channels=256,
                 num_classes=40,
               ):
        super(Head, self).__init__()

        self.with_avg_pool = with_avg_pool
        self.feat_size = feat_size
        self.in_channels = in_channels
        self.num_classes = num_classes

        in_channels = self.in_channels
        self.avgpool = nn.AvgPool2d(self.feat_size)
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool1d(num_rois)
        #else:
        #    in_channels *= (self.roi_feat_size * self.roi_feat_size)

        self.fc = nn.Linear(in_channels * 2 , num_classes)

        self.debug_imgs = None

    def init_weights(self):

        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.fc.bias, 0)


    def forward(self, feat, roi_feats):
        if self.with_avg_pool:
            feats = roi_feats.permute(0,2,1)
            feats = self.avg_pool(feats)
            roi_feats = feats.permute(0,2,1)
        roi_feats = roi_feats.view(roi_feats.size(0), -1)
        feat = self.avgpool(feat).view(feat.size()[0],-1)
        assert feat.size()[-1] == 256
        assert roi_feats.size()[-1] ==256
        combine_feat = torch.cat([roi_feats,feat],dim=1)
        output = self.fc(combine_feat)
        return output
