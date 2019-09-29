import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo

from util.config import cfg

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
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # previous stride is 2
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        #self.dropout  = nn.Dropout(0.9)
        self.fc = nn.Linear(512 * block.expansion, num_classes,bias=True)
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

        x4 = self.layer4(x3)

        x_gp = self.avgpool(x4)

        x_linear = x_gp.view(x_gp.size(0), -1)
        # concate the vector
        #combined_p = torch.cat((p1, p3), 1)
        x_out = self.fc(x_linear)

        return x_out
##########################################################
##########################################################

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    num = model.fc.in_features
    model.fc = torch.nn.Linear(num, cfg.MODEL.NUM_CLASS)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)#expand 1
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    num = model.fc.in_features
    model.fc = torch.nn.Linear(num, cfg.MODEL.NUM_CLASS)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)#expand 4

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        #load(model,cfg.MODEL.PRETRAINED)
        #ckp = torch.load(cfg.MODEL.PRETRAINED)
        #model.load_state_dict(ckp)
    num = model.fc.in_features
    model.fc = torch.nn.Linear(num, cfg.MODEL.NUM_CLASS)
    # change the classes' number
    #print(model)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
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
    ckp = torch.load(ckp_path)

    pretrained_dict = ckp
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
    for k,v in pretrained_dict:
        if k+'_' in model_dict:
            pretrained_dict[k+'_'] = v

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model
if __name__ == '__main__':
    m = resnet50(True)
    m.eval()
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from torchvision import transforms
    from PIL import Image
    from PIL import Image
    from lib.module.FindProposal import FindProposal

    loader = transforms.Compose([
        transforms.ToTensor()])
    unloader = transforms.ToPILImage()
    # 输入tensor变量
    # 输出PIL格式图片
    def tensor_to_PIL(tensor):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
        return image

    def imshow(tensor, title=None):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        image = unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    image = Image.open('./writing_on_a_book_093.jpg')

    from torchvision import transforms as T

    trn = T.Compose([T.Resize((224, 224)),
                     T.ToTensor(),
                     ])
    new_img = trn(image)
    #imshow(new_img,'new')
    pltimg = new_img
    norm = T.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
    new_img = norm(new_img)
    new_img = torch.autograd.Variable(new_img.unsqueeze(0))

    logit = m(new_img)
    logit = F.softmax(logit,1).squeeze()
    prob , index = logit.sort(0,True)
    print("right label is P{}".format(index[0]))

    box = FindProposal(m,new_img.data.squeeze(),5)
    y1, x1, y2, x2 = box
    image = tensor_to_PIL(pltimg)
    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image

    ax.imshow(image)

    # Create a Rectangle patch
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.show()
    plt.pause(0.001)
    print("result box is {} ".format(box))






