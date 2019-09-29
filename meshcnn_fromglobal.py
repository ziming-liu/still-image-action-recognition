import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'rgb_resnet18', 'rgb_resnet34', 'rgb_resnet50', 'rgb_resnet50_aux', 'rgb_resnet101',
           'rgb_resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
from torch.nn import functional as F
#from spatial_multi_head_attention import MultiHeadAttention
from lib.module.attention_layer import  MultiHeadAttention
from mmdet.models.roi_extractors import SingleRoIExtractor

num = 30

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=40):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.conv1g = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1g = nn.BatchNorm2d(64)
        self.relug = nn.ReLU(inplace=True)
        self.maxpoolg = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1g = self.layer1
        self.layer2g = self.layer2
        self.layer3g = self.layer3
        self.layer4g = self.layer4
        self.avgpooll = nn.AvgPool2d(2)
        self.avgpoolg = nn.AvgPool2d(7)
        #self.fcadd = nn.Linear(2048,2048)
        #self.atn_s1 = MultiHeadAttention(56*56,256,64,64)
        #self.atn_s2 = MultiHeadAttention(28*28,512,128,128)
        self.atn_s4 = MultiHeadAttention(1,2048*2*2,512,512)
        #self.atn_s3 = MultiHeadAttention(14*14,1024,256,256)
        # self.fc_aux = nn.Linear(512 * block.expansion, 101)
        self.dp = nn.Dropout(p=0.8)
        self.fc_g = nn.Linear(512 * block.expansion, num_classes)
        self.fc_l = nn.Linear(512 * block.expansion, num_classes)
        # self.bn_final = nn.BatchNorm1d(num_classes)
        # self.fc2 = nn.Linear(num_classes, num_classes)
        # self.fc_final = nn.Linear(num_classes, 101)
        self.extractor = SingleRoIExtractor(dict(type='RoIAlign', out_size=2, sample_num=1),out_channels=2048,
                                           featmap_strides=[32, ]
                                           )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x,rois):
        g = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)##

        x = self.layer4(x)##

        #print("x shape {}".format(x.shape))
        #multi sample
        feature = x.unsqueeze(0)
        roi_feats = self.extractor(feature, rois)
        b2,c2,h2,w2 = roi_feats.shape
        roi_feats = roi_feats.contiguous().view(b2//num,num,c2,h2,w2).view(b2//num,num,-1)
        roi_feats_withatn,context = self.atn_s4(roi_feats,roi_feats,roi_feats)
        #print("context {}".format(context.shape))
        roi_feats_withatn = roi_feats_withatn.contiguous().view(b2//num,num,c2,h2,w2).contiguous().mean(1).squeeze(1)
        l = roi_feats_withatn


        l = self.avgpooll(l)
        l = l.view(l.size(0),-1)
        l = self.dp(l)
        outl = self.fc_l(l)

        g = self.conv1g(g)
        g = self.bn1g(g)
        g = self.relug(g)
        g = self.maxpoolg(g)

        g = self.layer1g(g)

        g = self.layer2g(g)

        g = self.layer3g(g)  ##

        g = self.layer4g(g)  ##
        g = self.avgpoolg(g)
        g = g.view(g.size(0), -1)
        g = self.dp(g)
        outg= self.fc_g(g)

        out = 2*outg + outl
        out = F.softmax(out,1)
        return out,context


def rgb_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def rgb_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

from mmcv.runner import load_checkpoint
def rgb_resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        #load_checkpoint(model,model_urls['resnet50'])
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        #pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        pretrained_dict = torch.load("MODELZOO/globalcnn_best.pth")['state_dict']
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k.split('.',1)[0]+'g.'+k.split('.',1)[-1] : v for k, v in pretrained_dict.items() if k.split('.')[0] +'g.'+k.split('.',1)[-1]  in model_dict}
        print(pretrained_dict)
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model

def rgb_resnet50_aux(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])

        model_dict = model.state_dict()
        fc_origin_weight = pretrained_dict["fc.weight"].data.numpy()
        fc_origin_bias = pretrained_dict["fc.bias"].data.numpy()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # print(model_dict)
        fc_new_weight = model_dict["fc_aux.weight"].numpy()
        fc_new_bias = model_dict["fc_aux.bias"].numpy()

        fc_new_weight[:1000, :] = fc_origin_weight
        fc_new_bias[:1000] = fc_origin_bias

        model_dict["fc_aux.weight"] = torch.from_numpy(fc_new_weight)
        model_dict["fc_aux.bias"] = torch.from_numpy(fc_new_bias)

        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model

def rgb_resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def rgb_resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
        pretrained_dict = model_zoo.load_url(model_urls['resnet152'])
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model
