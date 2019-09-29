import torch
import torch.nn as nn
import numpy as np

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.contiguous().permute(0,2,1))
        attn = attn / self.temperature
        b,h,w = attn.shape
        #attn = attn.contiguous().view(attn.size()[0],-1)
        attn = self.softmax(attn)
        #attn = attn.contiguous().view(b,h,w)
        #attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn
