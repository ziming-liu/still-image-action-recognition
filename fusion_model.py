
from mmcv.cnn import ResNet
from mmcv.cnn.weight_init import kaiming_init,normal_init
import torch.nn as nn
import torch
from torch.nn import  functional as F
import math
import torch.utils.model_zoo as model_zoo
from spatial_multi_head_attention import MultiHeadAttention,MultiHeadAttention_simple,MultiHeadAttention_nolocalversion
from multi_resolution_cnn import multi_resolution_cnn
from torchvision.models import resnet50,densenet121
from mmcv.runner import load_checkpoint


class fusion_model(nn.Module):
    def __init__(self):
        super(fusion_model, self).__init__()
        self.m1 = multi_resolution_cnn(50,40)
        #self.m2 = densenet121(False)
        #num = self.m2.classifier.in_features
        #self.m2.classifier = torch.nn.Linear(num, 40)

        self.m3 = resnet50(False)
        num = self.m3.fc.in_features
        self.m3.fc = torch.nn.Linear(num,40)
        self.init_weight(self.m1,self.m3)
    def forward(self, input):
        newin = input[-1]
        assert  newin.shape[-1] ==224
        #print(newin.shape)
        #out2 = self.m2(newin)
        out1 = self.m1(input[:-1])
        out3 = self.m3(newin)
        result = out3+out1
        return result

    def init_weight(self,model1,model3):
        print("m1.....")
        load_checkpoint(model1, '/home/share2/zimi/GLnet/MODELZOO/local_multi_fineturn/save_178.pth')
        #print("m2.....")
        #load_checkpoint(model2,'/home/share2/zimi/GLnet/MODELZOO/densnet/save_6.pth')
        print("m3.....")
        load_checkpoint(model3, '/home/share2/zimi/GLnet/MODELZOO/std_resnet50_norm2/save_16.pth')
        print("loadd ckp success")

