import torch.nn as nn
import math
import torch as t
import torch.utils.model_zoo as model_zoo
#from lib.model import generateProposal

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
from lib.module.AttentionFuntions import blockAttention
from lib.module.attention_layer import MultiHeadAttention
from lib.module.BasicModule import BasicModule
class ResNet(BasicModule):

    def __init__(self, block, layers, num_classes=40,num_regions=5,size_region=(112//2)):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) # previous stride is 2
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(14)
        self.fc_ = nn.Linear(2048*num_regions, num_classes)
        self.adaptpool = nn.AdaptiveAvgPool2d((5,5))
        self.pool3d = nn.AdaptiveAvgPool3d((2048,1,1))
        self.num_regions = num_regions
        self.size_region = size_region
        self.attention = MultiHeadAttention(2,2048,2048,2048)
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

    def forward(self, x,mask):

        x = self.conv1(x)
        x = self.bn1(x)
        input = self.relu(x)
        B,_,m,_ = input.shape
        location = generateProposal(mask,self.num_regions,self.size_region,ratio=112//14)
        imgs =[]
        toshow = []
        for jj in range(B):
            objects = []
            for ii in range(self.num_regions):
                ymin,xmin,ymax,xmax = location[jj,ii],location[jj,ii+self.num_regions],\
                                      location[jj,ii+2*self.num_regions],location[jj,ii+3*self.num_regions]
                if jj==0:
                    toshow.append((2*ymin.cpu().int(),2*xmin.cpu().int(),2*ymax.cpu().int(),2*xmax.cpu().int()))
                #x = self.maxpool(x)
                # x [b,64,112,112]  proposal [1,64,h,w]
                proposal = input[jj,:,ymin:ymax,xmin:xmax].unsqueeze(0)
                #print(proposal.shape)
                x = self.layer1(proposal)
                x = self.layer2(x)
                x = self.layer3(x)
                #proposal ([1, 2048, 7, 7])
                proposal = self.layer4(x)
                #([1, 2048, 5, 5])
                proposal_ = self.adaptpool(proposal)
                #proposal_ = proposal_.contiguous().view(1,3,)
                objects.append(proposal_)
            imgs.append(t.cat(objects,dim=0))
        #[10, 3, 2048, 5, 5])
        images = t.stack(imgs)

        assert images.size(0) ==B
        #attention
        images = self.pool3d(images)

        images = blockAttention(images, self.attention)

        #print('images {}'.format(images.shape))
        #images = t.mean(images,dim=1)
        #x = self.avgpool(x)
        x = images
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc_(x)

        return x, toshow, mask


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        path = ('/home/share/LabServer/GLnet/ckp/resnet50_baseline/resnet50_baseline.pth')
        load(model, path)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        path = ('/home/share/LabServer/GLnet/ckp/resnet50_baseline/resnet50_baseline.pth')
        load(model,path)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
#load model statedict
def load(model,path):
    ckp_path = path
    ckp = t.load(ckp_path)

    pretrained_dict = ckp
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model
if __name__ == '__main__':
    print(help(nn.AdaptiveAvgPool2d))
