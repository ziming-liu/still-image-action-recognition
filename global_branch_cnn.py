import torch.nn as nn
import math
import numpy as np
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Variable
from torch.nn  import functional as F
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

from torchvision import transforms
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


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


class ResNet(nn.Module):

    def __init__(self, block=Bottleneck,  layers=[3, 4, 6, 3], nb_classes=40, channel=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1_custom = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.pool14 = nn.MaxPool2d(14)
        #self.fc_custom = nn.Linear(512 * block.expansion, nb_classes)
        #self.fc_at1 = nn.Linear(1024,nb_classes)
        #self.fc_at2 = nn.Linear(2048,nb_classes)
        self.conv_lower = nn.Sequential(OrderedDict([
            ('con1',nn.Conv2d(1024,521,kernel_size=3,stride=2,padding=1)),
            ('bn1',nn.BatchNorm2d(521)),
            ('relu1',nn.ReLU()),
                                        ('conv2',(nn.Conv2d(521,256,kernel_size=3,stride=1,padding=1))),
            ('bn2',nn.BatchNorm2d(256)),
            ('relu2',nn.ReLU()),
                                        ('conv3',(nn.Conv2d(256,10,kernel_size=1,stride=1,padding=0))),
            ('bn3',nn.BatchNorm2d(10)),
            ('relu3',nn.ReLU()),
                                        ('conv4',(nn.Conv2d(10,1,kernel_size=1,stride=1))),
            ('relu4',nn.ReLU()),
                                        ]))
        self.conv_atten = self.layer4
        #self.layer3_main = self.layer3
        #self.layer4_main = self.layer4
        self.fc_custom = nn.Linear(2048,nb_classes)
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
    def forward(self, x):
        x = self.conv1_custom(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        #atn = self.conv_lower(x3) #self.conv_lower(x3).repeat(1,1,1,1)
        #print(atn.shape)
        feat = x4#.repeat(1,1,1,1) #* atn.repeat(1,2048,1,1)
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0),-1)
        out = F.softmax(self.fc_custom(feat),1)
        return  out





"""
    def forward(self, x):
        x = self.conv1_custom(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        go = x
        x = self.layer3(x)
        x_at1 = x
        # generate at1
        b, c, h, w = x_at1.shape
        global_attent_waiter = x_at1.contiguous().view(b, c, h * w).detach()
        x_at1 = self.pool14(x_at1).view(x_at1.size(0), -1)
        out_at1 = F.softmax(self.fc_at1(x_at1),dim=1)
        index = torch.argsort(out_at1, dim=1, descending=True)
        maxscore_index = index[:, 0]#max class
        params_w = list(self.fc_at1.parameters())[0].detach()
        max_params_w = [params_w[maxscore_index[i]] for i in range(len(maxscore_index))]#b*channel
        global_attent = [sum(max_params_w[i].view(c, 1).repeat(1, h * w) * (global_attent_waiter[i]),0) for i in
                         range(len(max_params_w))]
        global_attent_1 = torch.stack(global_attent).view(b, 1, h, w).repeat(1,c,1,1)  # b,c,7,7
        global_attent_1.requires_grad = False

        x = self.layer4(x)
        x_at2  = x
        b,c,h,w = x_at2.shape
        global_attent_waiter = x_at2.contiguous().view(b,c,h*w).detach()
        x_at2 = self.avgpool(x_at2)
        #generate at2
        x_at2 = x_at2.view(x_at2.size(0), -1)
        out_at2 = F.softmax(self.fc_at2(x_at2),dim=1)
        index = torch.argsort(out_at2,dim=1,descending=True)
        #print(index.shape)
        maxscore_index = index[:,0]
        params_w = list(self.fc_at2.parameters())[0].detach()
        max_params_w = [params_w[maxscore_index[i]] for i in range(len(maxscore_index))]

        global_attent = [sum(max_params_w[i].view(c, 1).repeat(1, h * w) * (global_attent_waiter[i]), 0) for i in
                         range(len(max_params_w))]
        global_attent_2 = torch.stack(global_attent).view(b, 1, h, w).repeat(1, c, 1 ,1 )  # b,c,7,7
        global_attent_2.requires_grad = False
        #main branch
        feat = self.layer3_main(go)
        feat = feat * global_attent_1
        feat = self.layer4_main(feat)
        feat = feat * global_attent_2
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0),-1)
        out_main = F.softmax(self.fc_main(feat),1)
        #print(len(list(self.fc_custom.parameters())[0]))
        #out = F.softmax(out,dim=1)
        return 0.75*out_at2 + out_at1+out_main
"""

def resnet18(pretrained=False, channel= 20, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], nb_classes=101, channel=channel, **kwargs)
    if pretrained:
       pretrain_dict = model_zoo.load_url(model_urls['resnet18'])                  # modify pretrain code
       model_dict = model.state_dict()
       model_dict=weight_transform(model_dict, pretrain_dict, channel)
       model.load_state_dict(model_dict)
    return model


def resnet34(pretrained=False, channel= 20, **kwargs):

    model = ResNet(BasicBlock, [3, 4, 6, 3], nb_classes=101, channel=channel, **kwargs)
    if pretrained:
       pretrain_dict = model_zoo.load_url(model_urls['resnet34'])                  # modify pretrain code
       model_dict = model.state_dict()
       model_dict=weight_transform(model_dict, pretrain_dict, channel)
       model.load_state_dict(model_dict)
    return model


def resnet50(pretrained=False, channel= 20,nb_classes=40, **kwargs):

    model = ResNet(Bottleneck, [3, 4, 6, 3], nb_classes=nb_classes, channel=channel, **kwargs)
    if pretrained:
       pretrain_dict = model_zoo.load_url(model_urls['resnet50'])                  # modify pretrain code
       model_dict = model.state_dict()
       model_dict=weight_transform(model_dict, pretrain_dict, channel)
       model.load_state_dict(model_dict)
    return model


def resnet101(pretrained=False, channel= 20, **kwargs):

    model = ResNet(Bottleneck, [3, 4, 23, 3],nb_classes=101, channel=channel, **kwargs)
    if pretrained:
       pretrain_dict = model_zoo.load_url(model_urls['resnet101'])                  # modify pretrain code
       model_dict = model.state_dict()
       model_dict=weight_transform(model_dict, pretrain_dict, channel)
       model.load_state_dict(model_dict)

    return model


def resnet152(pretrained=False, **kwargs):

    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

def cross_modality_pretrain(conv1_weight, channel):
    # transform the original 3 channel weight to "channel" channel
    S=0
    for i in range(3):
        S += conv1_weight[:,i,:,:]
    avg = S/3.
    new_conv1_weight = torch.FloatTensor(64,channel,7,7)
    #print type(avg),type(new_conv1_weight)
    for i in range(channel):
        new_conv1_weight[:,i,:,:] = avg.data
    return new_conv1_weight

def weight_transform(model_dict, pretrain_dict, channel):
    weight_dict  = {k:v for k, v in pretrain_dict.items() if k  in model_dict }
    for k, v in pretrain_dict.items():
        if k+'_main' in model_dict:
            weight_dict[k+'_main'] = v
    #print pretrain_dict.keys()
    w3 = pretrain_dict['conv1.weight']
    #print type(w3)
    if channel == 3:
        wt = w3
    else:
        wt = cross_modality_pretrain(w3,channel)

    weight_dict['conv1_custom.weight'] = wt
    model_dict.update(weight_dict)
    return model_dict

#Test network
if __name__ == '__main__':
    model = resnet34(pretrained= True, channel=10)
    print(model)





