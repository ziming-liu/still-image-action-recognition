import torch.nn as nn
import torch.nn.functional as F

#实例化  chpool = ChannelPool(intput_channel)
#通过一维pooling实现通道pooling
class ChannelPool(nn.MaxPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n,c,w*h).permute(0,2,1)
        pooled =  F.max_pool1d(input, self.kernel_size, self.stride,
                        self.padding, self.dilation, self.ceil_mode,
                        self.return_indices)
        _, _, c = input.size()
        pooled = pooled.view(n,w,h,1).permute(0,3,1,2)
        return pooled
