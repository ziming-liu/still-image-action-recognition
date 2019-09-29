import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import ResNet
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler
from mmdet.models.roi_extractors import SingleRoIExtractor
from mmdet.core.bbox.geometry import bbox_overlaps


class proposalModel(nn.Module):
    def __init__(self):
        super(proposalModel, self).__init__()
        self.backbone = ResNet(101,4,out_indices=(1,3,))# int obj不可迭代，加个 ,
        self.roiextract = SingleRoIExtractor(roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
                                            out_channels=512,featmap_strides=[8,])# 没有FPN 所以只有一个步长
        self.fc = nn.Linear(2048+2048,40)
        self.backbone.init_weights(pretrained='/home/share/LabServer/GLnet/MODELZOO/resnet50.pth')
        self.conv1 = nn.Conv2d(512,1024,3,1)
        self.conv2 = nn.Conv2d(1024,2048,3,1)


    def forward(self, img,gtbboxes,proposals,label):
        backbonefeat,backbonefeat_ = self.backbone(img)#  14 *14
        #print(backbonefeat.shape)
        #print(backbonefeat_.shape)
        #print(backbonefeat)
        #print('backbonefeat shape {}'.format(backbonefeat.shape))
        #print('out222  \n{}'.format(backbonefeat.shape))
        #print('out333  \n{}'.format(backbonefeat_.shape))
        globalfeat = F.adaptive_avg_pool2d(backbonefeat_,(1,1))
        globalfeat = globalfeat.view(globalfeat.shape[0],-1)
        # proposal  gtbboxes 都是 list 里面是tensor
        #print('globalfeat \n {}'.format(globalfeat.shape))
        # 筛选proposal

        #print('before {}'.format(proposals[0].shape))
        #print('gtbox {}'.format(gtbboxes[0].shape))
        for ii in range(len(gtbboxes)):
            tem_proposal = []
            iofs = bbox_overlaps(proposals[ii][:,:4].double(),gtbboxes[ii].double(),mode='iof',is_aligned=False)#
            iofs = iofs.sum(1)
            for jj,iof in enumerate(iofs):
                if iof>=0.2:
                    tem_proposal.append(proposals[ii][jj])
            if tem_proposal:
                proposals[ii] = torch.stack(tem_proposal,0)
                #print(proposals[ii].shape)
                assert proposals[ii].dim() ==2
            else:
                proposals[ii] = torch.cat((gtbboxes[ii],torch.zeros((gtbboxes[ii].size(0),1)).cuda()),1)# 如果没有符合要求的proposal，就让person box作为 proposal
            proposals[ii] = proposals[ii].double()
        #print('after{}'.format(proposals[0].shape))
        #list 中的proposal可以是5列或者4列 都可以转换一样的roi
        proposals = bbox2roi(proposals) #(list[Tensor]) --》Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
        proposals = proposals.cuda().float()
        idxs = proposals[:,0].data.cpu().numpy()
        #print('proposal shape {}'.format(proposals.shape))
        bbox_feat = self.roiextract([backbonefeat.float()],proposals)#只有一个尺度无FPN，所以list里面只有一个尺度的feat
        #print('bbox feat shape {}'.format(bbox_feat.shape))
        #print(torch.sum(bbox_feat[0]))
        split_bbox_feat = []
        start = 0
        for ii,index in enumerate(idxs):

            if  ii!=len(idxs)-1 and index==idxs[ii+1]:
                continue
            else:
                end = ii+1
                mean_proposal = bbox_feat[start:end].sum(0)
                split_bbox_feat.append(mean_proposal)
                start =ii+1
                #print(index)
        #print((split_bbox_feat))
        bbox_feat = torch.stack(split_bbox_feat,0) # B , c ,7 ,7
        bbox_feat = self.conv1(bbox_feat)
        bbox_feat = self.conv2(bbox_feat)
        #print(bbox_feat.shape)
        bbox_feat = F.adaptive_avg_pool2d(bbox_feat,(1,1))#B , c ,1 ,1
        #print('bboxfeat  \n{}'.format(bbox_feat.shape))
        #print(bbox_feat.shape)
        assert bbox_feat.shape[1]==2048
        bbox_feat = bbox_feat.view(bbox_feat.size(0),-1)
        end_feat = torch.cat((globalfeat , bbox_feat),1)
        output = self.fc(end_feat)
        return output


if __name__ == '__main__':
    print(help(F.adaptive_avg_pool2d))

