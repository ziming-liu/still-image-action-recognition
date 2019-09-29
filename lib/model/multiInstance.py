import torch as t
from torch import nn
import math
import torch.utils.model_zoo as model_zoo
from lib.module.resnet import BasicBlock,Bottleneck
from util.config import cfg
from lib.module.AttentionFuntions import blockAttention
from lib.module.attention_layer import MultiHeadAttention
from lib.module.BasicModule import BasicModule
class multiInstance(BasicModule):
    def __init__(self, block, layers, num_classes=1000,num_region = 5):
        super(multiInstance, self).__init__()
        self.inplanes = 64


        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # previous stride is 2
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d((4,4))
        # self.dropout  = nn.Dropout(0.9)
        self.fc1 = nn.Linear( num_region*512 * block.expansion, cfg.MODEL.NUM_CLASS, bias=False)
        #self.fc2 = nn.Linear(512 * block.expansion,40,bias=True)
        self.num_region = num_region
        self.attention = MultiHeadAttention(2,2048,2048,2048)
        # 50 以上的模型 expansion 为4  其他为1

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()
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

    def forward(self, input,mask):

        B,C,H,W = mask.shape
        #print('mask {}'.format(mask.shape))
        mask_out = mask
        mask = mask.view(mask.size(0),-1)
        pixel , index = mask.sort(1,True)
        toshow = []
        step = int(224 // 2)
        store = []
        re_loc = []
        for ii in range(self.num_region):
            y_,x_ = index[:, ii] / 14, index[:, ii]%14 #- index[:, ii] / 14 * 14 - 1
            re_loc.append(t.cat((y_.view(-1,1),x_.view(-1,1)),dim=1))
        #print(re_loc[0].shape
        re_loc = t.stack(re_loc)
        #[n,5,2]
        re_loc = re_loc.permute(1,0,2)


        #print(re_loc.shape)

        location,idx = re_loc.sort(1,False)
        #print(location[0])
        #print('location {}'.format(location.shape))
        for ii in range(self.num_region):
            locx, locy = location[:,ii,1],location[:,ii,0]
            #print('locx {}'.format(locx.shape))
            #loclist.append([y1,x1,y2,x2])
            subregion = []
            for jj in range(B):
                xmin = locx[jj] * (224 / 14) - (step // 2)
                if xmin < t.tensor(0).cuda():
                    xmin = t.tensor(0).cuda()
                    xmax = t.tensor(step).cuda()
                else:
                    xmax = locx[jj] * (224 / 14) + (step // 2)
                    if xmax > t.tensor(223).cuda():
                        xmax = t.tensor(223).cuda()
                        xmin = t.tensor(223 - step).cuda()
                ymin = locy[jj] * (224 / 14) - (step // 2)
                if ymin < t.tensor(0).cuda():
                    ymin = t.tensor(0).cuda()
                    ymax = t.tensor(step).cuda()
                else:
                    ymax = locy[jj] * (224 / 14) + (step // 2)
                    if ymax > t.tensor(223).cuda():
                        ymax = t.tensor(223).cuda()
                        ymin = t.tensor(223 - step).cuda()
                if jj==0:
                    toshow.append((ymin.cpu().int(),xmin.cpu().int(),ymax.cpu().int(),xmax.cpu().int()))
                tem = input[jj,:,ymin.cpu().int():ymax.cpu().int(),xmin.cpu().int():xmax.cpu().int()]

                #print('tem {}'.format(tem.shape))
                subregion.append(tem)
            #print(len(subregion))
            #([50, 3, 112, 112])
            subregion_ = t.stack(subregion)
            #print('subregin {}'.format(subregion_.shape))
           # subregion_ = subregion_.squeeze()
            #print(subregion_.shape)
            x = self.conv1(subregion_)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)#28*28
            x = self.layer3(x)
            x = self.layer4(x)
            #[50, 2048, 1, 1])
            x = self.avgpool(x)

            store.append(x)
        #[50, 10240, 1, 1])
        #store_ = t.cat(store,1)
        #stack 是 50,5,2048,1,1
        store_ = t.stack(store,dim=1)
        store_ = blockAttention(store_,self.attention)
        #print(store_.shape)
        store_ = store_.view(store_.size(0), -1)
        #store = t.stack(store)
        #print(store.shape)
        #store = store.sum(0).squeeze()
        #print(store.shape)
        #print('stroe shape {}'.format(store_))
        #x = self.layer3_(store_)
        #x = self.layer4_(x)
        #x = F.relu(F.dropout(self.fc_(store)))
        #x = F.relu(F.dropout(self.fc2_(x)))
        #x = F.relu(F.dropout(self.fc3_(x)))
        x = self.fc1(store_)

        return x,toshow,mask_out


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def multi_one(pretrained=False,**kwargs):
    model = multiInstance(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        ckp = model_zoo.load_url(model_urls['resnet34'])
        pretrained_dict = ckp
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model
def multi_two(pretrained=False,**kwargs):
    model = multiInstance(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        #ckp = model_zoo.load_url(model_urls['resnet50'])
        path = ('/home/share/LabServer/GLnet/ckp/resnet50_baseline/resnet50_baseline.pth')
        ckp = t.load(path)
        pretrained_dict = ckp
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # print(pretrained_dict)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    #num = model.fc.in_features
    #model.fc = t.nn.Linear(num, int(num /9) )
    return model

def multi_three(pretrained=False,**kwargs):
    model = multiInstance(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        ckp = model_zoo.load_url(model_urls['resnet101'])

        pretrained_dict = ckp
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #print(pretrained_dict)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model



def multi_four(pretrained=False,**kwargs):
    model = multiInstance(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        ckp = model_zoo.load_url(model_urls['resnet152'])

        pretrained_dict = ckp
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        #print(pretrained_dict)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model
