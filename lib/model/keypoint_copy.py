import torch as t
from torch import nn
import torch.utils.model_zoo as model_zoo
from lib.module.BasicModule import BasicModule
from lib.module.resnet import BasicBlock,Bottleneck,conv1x1
from lib.core.maxchannelpool import ChannelPool


class KeyPoint(BasicModule):
    def __init__(self, block, layers, k=10, zero_init_residual=True):
        super(KeyPoint, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_ = self._make_layer(block, k, layers[2], stride=2)
        # self.layer4_=self._make_layer(block,k,layers[3],stride=2)
        self.maxchannelpool = ChannelPool(k*block.expansion)
        self.fc_ = nn.Linear(100352, 40)
        self.adappool = nn.AdaptiveAvgPool2d((14,14))
        self.adappool2 = nn.AdaptiveAvgPool2d((1,1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        midout = x
        midout = self.adappool(midout)

        x3 = self.layer3_(x)

        x = self.maxchannelpool(x3)
        pooled = x

        feat_class = pooled.repeat(1,midout.size(1),1,1) * midout
        #feat_class = self.adappool2(feat_class)# global pooling
        logit = self.fc_(feat_class.view(feat_class.size(0), -1))

        x_ = x.view(x.size(0), -1)
        re , index = x_.sort(1,True)
        max = re[:,:10].sum(1)

        min = re[:,10:].sum(1)


        return max,min, x, logit


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def onekey_copy(pretrained=False,**kwargs):
    model = KeyPoint(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        ckp = model_zoo.load_url(model_urls['resnet18'])
        pretrained_dict = ckp
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model
def twokey_copy(pretrained=False,**kwargs):
    model = KeyPoint(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        #ckp = model_zoo.load_url(model_urls['resnet50'])
        ckp = t.load('/home/share/LabServer/GLnet/ckp/resnet50_baseline/resnet50_baseline.pth')
        pretrained_dict = ckp
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


