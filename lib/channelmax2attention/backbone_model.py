from mmcv.cnn import ResNet
import torch.nn as nn
from torch.nn import functional as F
class backbone_model(nn.Module):
    def __init__(self):
        super(backbone_model, self).__init__()
        self.backbone = ResNet(50,4,out_indices=(3,))
        self.fc = nn.Linear(2048,11)
        self.global_pool = nn.AvgPool2d(kernel_size=(7,7),stride=1)

    def forward(self, input):
        stage4_feat  = self.backbone(input)
        stage4_feat = self.global_pool(stage4_feat)
        feat = stage4_feat.contiguous().view(stage4_feat.size(0),-1)
        output = self.fc(feat)
        return output

    def compute_loss(self,result,target):
        crossloss = F.cross_entropy(result,target)

        return crossloss

