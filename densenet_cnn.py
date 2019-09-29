
from mmcv.cnn import ResNet
from mmcv.cnn.weight_init import kaiming_init,normal_init
import torch.nn as nn
import torch
from torch.nn import  functional as F
import math
import torch.utils.model_zoo as model_zoo
from spatial_multi_head_attention import MultiHeadAttention,MultiHeadAttention_simple,MultiHeadAttention_nolocalversion



from densenet import densenet161

class densenet_cnn(nn.Module):
    def __init__(self,depth=50,numclass=40):
        super(densenet_cnn, self).__init__()

        self.backbone = densenet161(pretrained=True,)
        #self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        #self.classifier = nn.Sequential(
        #    nn.Linear(2048, 512),
        #    nn.ReLU(True),
        #    nn.Dropout(),
        #    nn.Linear(512,numclass)
       # )
        self.classifier = nn.Linear(2048,numclass)
        #self.atn_s3 = MultiHeadAttention(4,2048,512,512)
        self.atn_s4 = MultiHeadAttention(1,2048,512,512)
       # self.avg3dpool = nn.AdaptiveAvgPool3d((2048,1,1))
    def forward(self, x):
        """

        :param x: list[batch,channel,h,w]
        :return:
        """
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feat = self.backbone(x)
        all_feat = feat
        #all_feat = self.avgpool(feat)
        #all_feat = [self.avgpool(self.backbone(x[ii])) for ii in range(len(x))]
        # for ii,content in enumerate(x):
        #    feat = self.backbone(content)
        #    feat_vct = self.avgpool(feat)
        #    all_feat.append(feat_vct)
        # t, b,c,1,1
        #all_feat = torch.stack(all_feat)
        _, c_, h_, w_ = all_feat.shape
        feats = all_feat.contiguous().view(b, t, c_, h_, w_)
        #t, b, c, h, w = all_feat.shape
        #feats = all_feat.contiguous().permute(1, 0, 2, 3, 4)
        attention_feat_vct,context = self.atn_s4(feats,feats,feats)
        #attention_feat_vct, context = self.atn_s4(attention_feat_vct, attention_feat_vct, attention_feat_vct)
        #attention_feat_vct, context = self.atn_s4(attention_feat_vct, attention_feat_vct, attention_feat_vct)
        # print(attention_feat_vct.shape)
        # feats = self.avg3dpool(attention_feat_vct)
        # print(feats.shape)
        #attention_feat_vct, context = self.atn_s4(feats, feats, feats)
        # print(attention_feat_vct.shape)
        attention_feat_vct = attention_feat_vct.contiguous().view(b, t, -1).mean(1).squeeze(1)
        attention_feat_vct = attention_feat_vct.view(b, -1)
        outputs = self.classifier(attention_feat_vct)
        output = F.softmax(outputs, 1)
        # outputs = outputs.view(b,t,-1)
        # output = outputs
        return output,context






















