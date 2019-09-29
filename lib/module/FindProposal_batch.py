#coding:utf-8
import torch as t
import numpy as np
import torch.nn.functional as F
from collections import Counter
import torch


def FindProposal_batch(model,init_img,n_grids=4):
    #################################定位###############################
    B,C,H,W = init_img.shape
    step = H / n_grids
    num_coord = n_grids * n_grids
    coord_grids = np.zeros((num_coord,4))# y x y x
    k = 0
    for yy in range(n_grids):
        for xx in range(n_grids):
            coord_grids[k] = np.array([yy*step,xx*step,(yy+1)*step,(xx+1)*step])
            k += 1
    assert k==num_coord-1,"wei得到所有的网格坐标"
    #抑制之后的图片 tensor
    masked_img = torch.zeros((num_coord,B,C,W,H))
    predict_labels = [] #存储所有的结果
    logits = []
    for i in range(num_coord):
        tmp_img = init_img.detach().clone()
        #一个batch的所有图片  抑制同一个区域

        tmp_img[:,:,coord_grids[0]:coord_grids[2],coord_grids[1]:coord_grids[3]] = t.from_numpy(np.array(0.5))
        masked_img[i] = tmp_img
        # inference
        logit = model(tmp_img)# batch * classes number
        logit = F.softmax(logit,1).detach()
        probs, index = logit.sort(1,True)#降序
        index = index.cpu() # 之后的索引中不能是cuda类型
        predict_labels.append(index[:,0])# shape  batch * 1 对于某一种mask情况
        logits.append(logit)
    predict_labels = t.stack(predict_labels,1)# shape batch * num grids
    logits = t.stack(logits)# num grid * batch * num classes
    logits = logits.permute(1,0,2)# batch * num grid * classes
    new_probs = []
    for batch_i in range(B):
        #投票 选择最可能的类别
        label_counts = Counter(predict_labels[batch_i])
        top = label_counts.most_common(3)
        prob_grids=logits[batch_i,:,top[0][0]].squeeze()# 预测到的 label 对应的 所有格子数的概率
        new_probs.append(prob_grids)
    new_probs_ = t.stack(new_probs)# batch * num grid
    assert new_probs_.shape == B,num_coord
    sorted_probs, index = new_probs_.sort(1,False)#升序
    #batch * 1
    most_imp_grid = index[:,0].squeeze()#target label预测概率最低的 mask情况说明该格子  对系统最重要。
    ##################### 回归########################################
    print(" most imp grid  {} ".format(most_imp_grid))
    #TODO : 得到的结果是batch个 坐标？？
    coord_final_batch = coord_grids[most_imp_grid.cpu().numpy()]


    print('masked image {}'.format(masked_img))



















