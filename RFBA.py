
from mmcv.cnn import ResNet
from mmcv.cnn.weight_init import kaiming_init,normal_init
import torch.nn as nn
import torch
from torch.nn import  functional as F
import math
import torch.utils.model_zoo as model_zoo
from spatial_multi_head_attention import MultiHeadAttention,MultiHeadAttention_simple,MultiHeadAttention_nolocalversion
from multi_resolution_cnn import multi_resolution_cnn
import numpy as np
from spatial_multi_head_attention import MultiHeadAttention,MultiHeadAttention_simple
from lib.core.scaledDotProductAttention import ScaledDotProductAttention

num = 40
class RFBA(multi_resolution_cnn):

    def __init__(self,depth=50,numclass=40):
        super(RFBA, self).__init__(depth=depth,numclass=numclass)
        #self.AM = MultiHeadAttention(1,2048,512,512)
        #self.atn_s0 = MultiHeadAttention(224*224,3,3,3)
        self.attention = ScaledDotProductAttention(temperature=np.power(40, 0.5))

    def forward(self, x):
        """
        :param x: list[batch,channel,h,w]
        :return:
        """
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feat = self.backbone(x)
        feat = self.avgpool(feat)
        _, c, h, w = feat.shape
        feats = feat.contiguous().view(b, t, c, h, w)

        X = []
        for i in range(num):
            sample = feats[:,3*i:3*(i+1),:,:,:]
            attention_feat_vct, context = self.atn_s4(sample, sample, sample)
            #attention_feat_vct = attention_feat_vct.contiguous().view(b, 6, c,h,w).mean(1).unsqueeze(1)
            attention_feat_vct = attention_feat_vct.contiguous().view(b,3, -1).mean(1)
            attention_feat_vct = attention_feat_vct.view(b, -1)
            outputs = self.classifier(attention_feat_vct)
            output = F.softmax(outputs, 1)
            X.append(output)
        #print(X[0].shape)
        x = torch.stack(X,1)
        #new_x,context2 = self.atn_s4(new_x,new_x,new_x)
        #X[0],_ = self.AM(X[0],X[0],X[0])
        #X[1],_ = self.AM(X[0],X[0],X[1])
        #X[2],_ = self.AM(X[1],X[1],X[2])
        #X[3], _ = self.AM(X[2], X[2], X[3])
        #X[4], _ = self.AM(X[3], X[3], X[4])
        #X[5], _ = self.AM(X[4], X[5], X[5])
        #X[6], _ = self.AM(X[5], X[5], X[6])
        #X[7], _ = self.AM(X[6], X[6], X[7])
        #X[8], _ = self.AM(X[7], X[7], X[8])
        #X[9], _ = self.AM(X[8], X[8], X[9])

        #X[3],_ = self.AM(X[2],X[2],X[3])
        #Y = X[1]
        # print(attention_feat_vct.shape)
        #attention_feat_vct = new_x.contiguous().view(b,num, -1).mean(1).squeeze(1)
        #attention_feat_vct = attention_feat_vct.view(b, -1)
        #outputs = self.classifier(attention_feat_vct)
        #output = F.softmax(outputs, 1)

        # out = out.contiguous().view(b, self.during, -1)
        # out = out.sum(1).squeeze(1)
        x, _ = self.attention(x, x, x)
        #x, _ = self.atn_s4(x, x, x)
        #x, _ = self.atn_s4(x, x, x)
        x = x.mean(1)

        return x,_









