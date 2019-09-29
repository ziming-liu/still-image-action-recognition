import torch as t
from torch.nn import functional as F


def generateProposal(keypoint,num_regions=6,size_region=(112//2),ratio = 112 // 14):
    B,C,H,W = keypoint.shape
    assert H==14
    assert C ==1
    values, index = keypoint.contiguous().view(B,-1).sort(1,True)
    index_wanted = index[:,:num_regions]
    indices = t.cat(((index_wanted / t.tensor(H).cuda()).view(-1,num_regions),\
                     (index_wanted % t.tensor(H).cuda()).view(-1,num_regions)),dim=1)
    assert indices.size(0) == B
    #indices is  [y_i, x_i]
    #compute the four location of regions(proposals)
    step = size_region // 2
    ymin = indices[:,:num_regions] * ratio- t.tensor(1).cuda()*step
    ymax = indices[:,:num_regions] * ratio+ t.tensor(1).cuda()*step
    xmin = indices[:,num_regions:] * ratio- t.tensor(1).cuda()*step
    xmax = indices[:,num_regions:] * ratio+ t.tensor(1).cuda()*step
    location = t.cat((ymin,xmin,ymax,xmax),dim=1)
    assert location.size(0) == B
    assert location.size(1) == num_regions*4
    location = F.relu(location)  # 负数坐标全部归零
    # shape  batchsize * (num regions*4)
    return location










