import torch
from torch.nn import functional as F
import torch.nn as nn
from mmcv.cnn.weight_init import kaiming_init, xavier_init, normal_init, constant_init
from mmcv.runner import load_checkpoint
from mmcv.cnn import ResNet
from lib.core.maxchannelpool import ChannelPool


class MaxScore(nn.Module):
    def __init__(self, K=12):
        super(MaxScore, self).__init__()
        self.K = K
        self.backbone = ResNet(50, num_stages=4,
                               strides=(1, 2, 2, 1),
                               out_indices=(2, 3))
        self.backbone.init_weights(pretrained='https://download.pytorch.org/models/resnet50-19c8e357.pth')
        self.attention = Attention(1024, K)
        self.attention.weight_init(None)
        # self.lin = nn.Linear(2048,64)
        self.fc2 = nn.Linear(512, 40)
        self.fc1 = nn.Linear(2048,512)
        self.fc = nn.Linear(80,40)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        # self.conv2 = nn.Conv2d(1024,self.K,1,1)
        # self.channelpool = ChannelPool(self.K)

    def forward(self, input,target):
        feature1, feature2 = self.backbone(input)
        primal_feat = F.avg_pool2d(feature2,(14,14)).view(feature2.size(0),feature2.size(1))
        primal_x = self.fc1(primal_feat)
        primal_x = self.fc2(primal_x)

        local_map, global_map = self.attention(feature1)
        # local_map = self.conv2(feature1)
        # n,c,h,w = local_map.shape
        # local_map = F.softmax(local_map,dim=1)
        # global_map = self.channelpool(local_map)
        # n,c,h,w = global_map.shape

        b, c, h, w = feature2.shape
        feat = feature2.view(b, 1, c, h, w)
        feat = feat.repeat(1, self.K , 1, 1, 1)
        map = local_map
        assert map.size()[1] == self.K
        b, c, h, w = map.shape
        map = map.view(b, c, 1, h, w)
        map = map.repeat(1, 1, 2048, 1, 1)
        feat_att = feat * map
        feat_att = F.avg_pool3d(feat_att, (1, 14, 14)).squeeze(-1).squeeze(-1)
        feat_att = feat_att.contiguous().view(b*c,2048)
        secondary_x = self.fc1(feat_att)
        secondary_x = self.fc2(secondary_x)
        secondary_x = secondary_x.view(b,c,-1)
        secondary_att = []
        for i in range(b):
            sec = secondary_x[i]
            index = torch.argmax(sec[:,target[i]].unsqueeze(1),dim=1)
            choose = sec[index[0]]
            secondary_att.append(choose)
        secondary_att = torch.stack(secondary_att)
        #output = torch.cat((primal_x,secondary_att),1)
        output = primal_x #+ secondary_att
        #output = self.fc(output)
        return output, global_map, local_map

    def weight_init(self, pretrained):
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False, )
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')


class Attention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Attention, self).__init__()
        self.conv1_ = nn.Conv2d(in_channel, 512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv_mid = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.conv2_ = nn.Conv2d(128, out_channel, kernel_size=3, stride=1, padding=1)

        self.channelpool = ChannelPool(out_channel)
        # self.weight_init(None)

    def forward(self, input):
        out1 = F.relu((self.conv1_(input)))
        out1_2 = F.relu(self.bn1(self.conv_mid(out1)))
        out2 = (self.conv2_(out1_2))
        n, c, h, w = out2.shape
        out2 = out2.contiguous().view(n, c, -1)
        out2 = F.softmax(out2, dim=2)
        out2 = out2.contiguous().view(n, c, h, w)
        global_map = self.channelpool(out2)  # [n,1,h,w]
        # 抑制边界集聚
        mask = torch.zeros_like(global_map)
        mask[:, :, 1:h - 1, 1:w - 1] = torch.ones(1)
        mask = mask.cuda()
        # print(mask)
        global_map = global_map * mask

        """
        #print(out2.shape)
        localmap = out2.clone()
        localmap = localmap.sum(3)
        #print(localmap.shape)
        maxindex = torch.argmax(localmap, dim=2)
        #print(maxindex.shape)
        #print(maxindex[0])
        index = torch.argsort(maxindex, dim=1)
        #print(index.shape)
        #print(index[0])
        newlocalmap = out2.clone()
        for i in range(n):
            newlocalmap[i] = out2[i, index[i], :, :]
        #print(newlocalmap.shape)
        """
        return out2, global_map

    def weight_init(self, pretrained):
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False, )
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')


from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feature, in_feature)
        self.conv2 = GCNConv(in_feature, out_feature)

    def forward(self, input, edge_index, edge_weight):
        x = self.conv1(input, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.softmax(x, dim=1)
