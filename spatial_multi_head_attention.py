''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from lib.core.scaledDotProductAttention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.5):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.w_qs = nn.Linear(d_model,  d_k)
        self.w_ks = nn.Linear(d_model,  d_k)
        self.w_vs = nn.Linear(d_model,  d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear( d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        """
        input: batch time channel h w
        output:  batch time channel h w
        :param q:
        :param k:
        :param v:
        :param mask:
        :return:
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        qb,qt,qc,qh,qw = q.shape
        kb,kt,kc,kh,kw = k.shape
        vb,vt,vc,vh,vw = v.shape
        q = q.contiguous().view(qb,qt,qc,-1)
        k = k.contiguous().view(kb,kt,kc,-1)
        v = v.contiguous().view(vb,vt,vc,-1)
        q = q.contiguous().permute(3,0,1,2).contiguous().view(-1,qt,qc)
        k = k.contiguous().permute(3,0,1,2).contiguous().view(-1,kt,kc)
        v = v.contiguous().permute(3,0,1,2).contiguous().view(-1,vt,vc)
        residual = q
        q = (self.w_qs(q))
        k = (self.w_ks(k))
        v = (self.w_vs(v))
        # nhead*batch  t  channel
        output,contet = self.attention(q, k, v)
        output,contet = self.attention(output, output, output)
        output,contet = self.attention(output, output, output)

        assert n_head == qh*qw
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)  # add & norm
        output = output.contiguous().view(n_head,qb, qt, self.d_model)
        output = output.contiguous().permute(1, 2, 3, 0).contiguous().view(vb,vt, vc, vh, vw)

        contet = contet.contiguous().view(n_head,qb,qt,qt)
        contet = contet.contiguous().permute(1,0,2,3).mean(1).squeeze(1)
        #contet = contet.view(contet.size()[0],-1)
        ##contet = F.softmax(contet,1)
        #contet = contet.view(qb,qt,qt)
        return output,contet



class MultiHeadAttention_simple(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.5,norm=True):
        super(MultiHeadAttention_simple,self).__init__()
        """
        q k v都不做线性变换，直接dot 。结果维度没有变，不需要升降维度  
        """
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        #self.w_qs = nn.Linear(d_model,  d_k)
        #self.w_ks = nn.Linear(d_model,  d_k)
        #self.w_vs = nn.Linear(d_model,  d_v)
        #nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        #nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        #nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.norm = norm
        if norm:
            self.layer_norm = nn.LayerNorm(d_model)
        #self.W= nn.Conv1d( d_v, d_model,kernel_size=1,stride=1,padding=0)
        #nn.init.xavier_normal_(self.fc.weight)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        """
        input: batch time channel h w
        output:  batch time channel h w
        :param q:
        :param k: inter dimension
        :param v: ouput dim d_out
        :param mask:
        :return:
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        qb,qt,qc,qh,qw = q.shape
        kb,kt,kc,kh,kw = k.shape
        vb,vt,vc,vh,vw = v.shape
        q = q.contiguous().view(qb,qt,qc,-1)
        k = k.contiguous().view(kb,kt,kc,-1)
        v = v.contiguous().view(vb,vt,vc,-1)
        q = q.contiguous().permute(3,0,1,2).contiguous().view(-1,qt,qc)
        k = k.contiguous().permute(3,0,1,2).contiguous().view(-1,kt,kc)
        v = v.contiguous().permute(3,0,1,2).contiguous().view(-1,vt,vc)
        residual = q
        q = q
        k = q
        v = v
        # nhead*batch  t  channel
        output,contet = self.attention(q, k, v)
        #output,contet = self.attention(output, output, output)
        #output,contet = self.attention(output, output, output)
        #output,contet = self.attention(output, output, output)

        assert n_head == qh*qw
        #output = self.W(output)
        if self.norm:
            output = self.layer_norm(output + residual)  # add & norm
        output = output.contiguous().view(n_head,qb, qt, self.d_model)
        output = output.contiguous().permute(1, 2, 3, 0).contiguous().view(vb,vt, vc, vh, vw)

        contet = contet.contiguous().view(n_head,qb,qt,qt)
        contet = contet.contiguous().permute(1,0,2,3).mean(1).squeeze(1)
        #contet = contet.view(contet.size()[0],-1)
        ##contet = F.softmax(contet,1)
        #contet = contet.view(qb,qt,qt)
        return output,contet


import torch

class MultiHeadAttention_nolocalversion(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(MultiHeadAttention_nolocalversion, self).__init__()
        """
        nonlocal 的dotproduct 版本改编为 multi head 的模型
        """
        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x,k,v):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        x = x.contiguous().permute(0,2,1,3,4)
        b,c,t,h,w =x.shape
        batch_size = x.size(0)


        g_x = self.g(x)
        # nhead,b,t,c
        g_x = g_x.view(batch_size, self.inter_channels, t,h*w).contiguous().permute(3,0,2,1)
        g_x = g_x.contiguous().view(-1,t,self.inter_channels)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, t,h*w).contiguous().permute(3,0,2,1)
        theta_x = theta_x.contiguous().view(-1,t,self.inter_channels)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, t,h*w).contiguous().permute(3,0,1,2)
        phi_x = phi_x.contiguous().view(-1,self.inter_channels,t)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N  # scale

        y = torch.matmul(f_div_C, g_x)
        # bhw , inter, t
        y = y.permute(0, 2, 1).contiguous().view(h*w,b,self.inter_channels,t).contiguous().permute(1,2,3,0)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        z = z.contiguous().permute(0,2,1,3,4)
        return z,f_div_C


