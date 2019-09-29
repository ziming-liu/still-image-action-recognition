import torch
from torch.nn import functional as F
import torch.nn as nn
from mmcv.cnn.weight_init import kaiming_init,xavier_init,normal_init,constant_init
from mmcv.runner import load_checkpoint
from mmcv.cnn import ResNet
from lib.core.maxchannelpool import ChannelPool
from lib.attention_reid.multi_spatial_attention import SattentionNet
class attentionReid(nn.Module):
    def __init__(self,K=6):
        super(attentionReid, self).__init__()
        self.K = K
        self.backbone = ResNet(50,num_stages=4,
                               strides=(1,2,2,1),
                               out_indices=(2,3))
        self.backbone.init_weights(pretrained='https://download.pytorch.org/models/resnet50-19c8e357.pth')
        self.attention = SattentionNet(num_features=128,seqlen=1,spanum=self.K)
        self.attention.reset_params()
        #self.GCN = GCN(2048,2048//K)
        self.lin = nn.Linear(2048,2048)
        self.fc = nn.Linear(128*self.K,11)
        nn.init.xavier_normal_(self.lin.weight)
        nn.init.xavier_normal_(self.fc.weight)
        self.conv2 = nn.Conv2d(1024,self.K,1,1)
        self.channelpool = ChannelPool(self.K)
    def forward(self, input):
        feature1,feature2 = self.backbone(input)
        #print("feautre1  {}".format(feature1))
        x,reg,attmap = self.attention(feature2)

        #print(feat_att)
        """
        #edge index
        idx = [i for i in range(self.K+1)]
        a = torch.LongTensor(idx).cuda()
        a2 = a.repeat(self.K+1)
        a1 = a.view(a.size(0), 1).repeat(1, self.K+1).view(-1)
        a1 = a1.view(1, a1.size(0))
        a2 = a2.view(1, a2.size(0))
        edge_index = torch.cat((a1, a2), dim=0).cuda()
        graph_feat = []
        for ii in range(feat_att.size(0)):
            graph_x = feat_att[ii]
            #print("befor {}".format(graph_x))
            #print(self.lin(graph_x).shape)
            edge_weight = self.lin(graph_x).mm(self.lin(graph_x).t())

            edge_weight = F.softmax(edge_weight.view(-1).cuda(),dim=0)

            #print(edge_weight.size(0))
            #print(edge_index.size(1))
            graph_x_ = self.GCN(graph_x,edge_index,edge_weight)
            #print("after {}".format(graph_x_))

            graph_feat.append(graph_x_)
        graph_feat_ = torch.stack(graph_feat)
        graph_feat_ = graph_feat_.view(graph_feat_.size(0),-1)

        output = self.fc(graph_feat_)
        #print(output)"""
        feat_att = x.view(x.size(0),-1)
        output = self.fc(feat_att)

        return output,reg,attmap

    def weight_init(self,pretrained):
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
    def __init__(self,in_feature,out_feature):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feature,1024)
        self.conv2 = GCNConv(1024,out_feature)

    def forward(self, input,edge_index,edge_weight):
        x = self.conv1(input,edge_index,edge_weight)
        x =F.relu(x)
        x = F.dropout(x,training=self.training)
        x = self.conv2(x,edge_index,edge_weight)
        return F.softmax(x,dim=1)
