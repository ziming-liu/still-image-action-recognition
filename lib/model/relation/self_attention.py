import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import (delta2bbox, multiclass_nms, bbox_target,
                        weighted_cross_entropy, weighted_smoothl1, accuracy)

''' Define the sublayers in encoder/decoder layer '''
import numpy as np
from lib.core.scaledDotProductAttention import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"

class SelfAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v,mode='block', dropout=0.3):
        super(SelfAttention,self).__init__()
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward_unified(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        if mask:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn
    def forward(self, x):

        if self.mode=='block':
            self.blockAttention(x,)
        elif self.mode == 'channel':
            self.channelAttention(x)
        elif self.mode == 'pixel':
            self.pixelAttention(x)
        else:
            raise IOError
    def channelAttention(self,x):
        """
        to learn the attention of one channel and the rest channels
        :param x:input tensor , shape as B,C,H,W
        :return: output tensor after self-attention op. shape as B,C,H,W
        """
        b, c, h, w = x.shape
        pre_x = x.contiguous().view(b, c, h * w)

        att_x, relations = self.forward_unified(pre_x, pre_x, pre_x)
        att_x = att_x.contiguous().view(b, c, h, w)
        return att_x

    def pixelAttention(self,x):
        """
        to learn the attention of one position and the other position
        in spicial scale.
        :param x:
        :param attention_op:
        :return:
        """
        b, c, w, h = x.shape
        pre_x = x.permute(0, 2, 3, 1)
        pre_x = pre_x.contiguous().view(b, h * w, c)
        att_x, relations = self.forward_unified(pre_x, pre_x, pre_x)
        att_x = att_x.contiguous().view(b, h, w, c)
        att_x = att_x.permute(0, 3, 1, 2)
        return att_x

    def blockAttention(self,x):
        """
        to learn the relation of n objects
        :param x:
        :param attention_op:
        :return:
        """
        b, n, c, w, h = x.shape
        pre_x = x.contiguous().view(b, n, -1)
        att_x, relations = self.forward_unified(pre_x, pre_x, pre_x)
        att_x = att_x.contiguous().view(b, n, c, w, h)
        return att_x


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
