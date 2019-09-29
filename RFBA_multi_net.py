
from mmcv.cnn import ResNet
from mmcv.cnn.weight_init import kaiming_init,normal_init
import torch.nn as nn
import torch
from torch.nn import  functional as F
import math
import torch.utils.model_zoo as model_zoo
from spatial_multi_head_attention import MultiHeadAttention,MultiHeadAttention_simple,MultiHeadAttention_nolocalversion
from multi_resolution_cnn import multi_resolution_cnn
from inception_cnn import inception_cnn
from inception import inception_v3

from mmcv.runner import load_checkpoint
num = 2
class RFBA(nn.Module):

    def __init__(self,depth=50,numclass=40):
        super(RFBA, self).__init__()
        self.backbone1 = multi_resolution_cnn()
        load_checkpoint(self.backbone1,'/home/share2/zimi/GLnet/MODELZOO/local_multi/save_170.pth')
        self.backbone2 = inception_cnn()
        load_checkpoint(self.backbone2,'/home/share2/zimi/GLnet/MODELZOO/local_multi_inception2/save_63.pth')

        #self.atn_s0 = MultiHeadAttention(224*224,3,3,3)
    def forward(self, input):
        """
        :param x: list[batch,channel,h,w]
        :return:
        """
        x = input[:,0:9,:,:,:].contiguous()
        X = []
        r1,_ = self.backbone1(x)

        x = input[:,9:18,:,:,:].contiguous()
        r2,_ = self.backbone2(x)

        output = r1 + r2

        return output,output









