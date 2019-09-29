from torch import nn
import torch.utils.model_zoo as model_zoo
from lib.module.BasicModule import BasicModule
from lib.core.maxchannelpool import ChannelPool
import torch
from util.config import cfg
import math

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
########################################################
########################################################
class KeyPoint(BasicModule):

    def __init__(self, block, layers, num_classes=40,num_layers=6):
        self.inplanes = 64
        super(KeyPoint, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # previous stride is 2
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_ = self._make_layer(block, num_layers, layers[3], stride=2)
        #self.convnext = nn.Conv2d(512*block.expansion,num_layers,kernel_size=1,stride=1,padding=0,bias=False)
        self.avgpool = nn.AvgPool2d(14)
        self.maxchannelpool = ChannelPool(num_layers)

        #self.dropout  = nn.Dropout(0.9)
        #self.fc_ = nn.Linear(256 * block.expansion, num_classes,bias=True)
        self.fc2 = nn.Linear(7*7,num_classes)
        #module needed
        #self.op1 = MultiHeadAttention(n_head=1, d_model=28*28, d_k=512, d_v=512, dropout=0.2)

        #50 以上的模型 expansion 为4  其他为1

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                #m.weight.data.fill_(1)
                #m.bias.data.zero_()
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)

        x1 = self.layer1(x0)

        x2 = self.layer2(x1)

        x3 = self.layer3(x2)

        x4 = self.layer4_(x3)
        #x44 = self.avgpool(x3)
        #x_linear = x44.view(x44.size(0), -1)
        #x_out = self.fc_(x_linear)
        ###################################
        feat_next = x4
        _,c,_,_ = x4.shape
        pooled = self.maxchannelpool(feat_next)
        #print(pooled.shape)
        pooled_ = pooled.view(pooled.size(0),-1)
        re, idx = pooled_.sort(1,True)
        x_out = self.fc2(pooled_)

        max = re[:,:6].sum(1)
        min = re[:,6:].sum(1)
        return x_out,max,min,pooled
##########################################################
##########################################################

def key18(pretrained=False, **kwargs):
   model = KeyPoint(BasicBlock, [2,2,2,2], **kwargs)#expand 4
   if pretrained:
       #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
       load(model,cfg.MODEL.BASELINE)
       #ckp = torch.load(cfg.MODEL.PRETRAINED)
       #model.load_state_dict(ckp)
   #num = model.fc_.in_features
   #model.fc_ = torch.nn.Linear(num, cfg.MODEL.NUM_CLASS)
   # change the classes' number
   #print(model)
   return model


def key34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = KeyPoint(BasicBlock, [3, 4, 6, 3], **kwargs)#expand 1
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    num = model.fc.in_features
    model.fc = torch.nn.Linear(num, cfg.MODEL.NUM_CLASS)
    return model


def key50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = KeyPoint(Bottleneck, [3, 4, 6, 3], **kwargs)#expand 4

    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        load(model,cfg.MODEL.BASELINE)
        #ckp = torch.load(cfg.MODEL.PRETRAINED)
        #model.load_state_dict(ckp)
    num = model.fc_.in_features
    model.fc_ = torch.nn.Linear(num, cfg.MODEL.NUM_CLASS)
    # change the classes' number
    #print(model)
    return model


def key101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = KeyPoint(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = KeyPoint(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


#load model statedict
def load(model,path):
    ckp_path = path
    ckp = torch.load(ckp_path)

    pretrained_dict = ckp
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

