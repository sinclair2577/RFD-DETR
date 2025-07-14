import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from ..modules.conv import Conv
from ..modules.block import BasicBlock
from .wtconv2d import WTConv2d

__all__=['BasicBlock_WTConv','WTConv2d','WFU','CSP_PAC','AttentionDownsample']

class BasicBlock_WTConv(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2b = WTConv2d(ch_out, ch_out)

class HaarWavelet(nn.Module):
    def __init__(self, in_channels, grad=False):
        super(HaarWavelet, self).__init__()
        self.in_channels = in_channels

        self.haar_weights = torch.ones(4, 1, 2, 2)
        #h
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1
        #v
        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1
        #d
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = grad

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.in_channels) / 4.0
            out = out.reshape([x.shape[0], self.in_channels, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            out = x.reshape([x.shape[0], 4, self.in_channels, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.in_channels)

class WFU(nn.Module):
    def __init__(self, chn):
        super(WFU, self).__init__()
        dim_big, dim_small = chn
        self.dim = dim_big
        self.HaarWavelet = HaarWavelet(dim_big, grad=False)
        self.InverseHaarWavelet = HaarWavelet(dim_big, grad=False)
        self.RB = nn.Sequential(
            # nn.Conv2d(dim_big, dim_big, kernel_size=3, padding=1),
            # nn.ReLU(),
            Conv(dim_big, dim_big, 3),
            nn.Conv2d(dim_big, dim_big, kernel_size=3, padding=1),
        )

        self.channel_tranformation = nn.Sequential(
            # nn.Conv2d(dim_big+dim_small, dim_big+dim_small // 1, kernel_size=1, padding=0),
            # nn.ReLU(),
            Conv(dim_big+dim_small, dim_big+dim_small // 1, 1),
            nn.Conv2d(dim_big+dim_small // 1, dim_big*3, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x_big, x_small = x
        haar = self.HaarWavelet(x_big, rev=False)
        a = haar.narrow(1, 0, self.dim)
        h = haar.narrow(1, self.dim, self.dim)
        v = haar.narrow(1, self.dim*2, self.dim) 
        d = haar.narrow(1, self.dim*3, self.dim)

        hvd = self.RB(h + v + d)
        a_ = self.channel_tranformation(torch.cat([x_small, a], dim=1))
        out = self.InverseHaarWavelet(torch.cat([hvd, a_], dim=1), rev=True)
        return out
    
class ParallelAtrousConv(nn.Module):
    def __init__(self, inc, ratio=[1, 2, 3]) -> None:
        super().__init__()
        
        self.conv1 = Conv(inc, inc, k=3, d=ratio[0])
        self.conv2 = Conv(inc, inc // 2, k=3, d=ratio[1])
        self.conv3 = Conv(inc, inc // 2, k=3, d=ratio[2])
        self.conv4 = Conv(inc * 2, inc, k=1)
    
    def forward(self, x):
        return self.conv4(torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1))

class CSP_PAC(nn.Module):
    """CSP Bottleneck with ParallelAtrousConv."""

    def __init__(self, c1, c2, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = ParallelAtrousConv(c_)

    def forward(self, x):
        """Forward pass through the CSP bottleneck with ParallelAtrousConv."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class AttentionDownsample(nn.Module):
    def __init__(self, inc) -> None:
        super().__init__()
        
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gate = nn.Sequential(
            nn.Conv2d(inc, inc, 1),
            nn.Hardsigmoid()
        )
        
        self.conv = Conv(inc, inc, k=1)
        self.down_branch1 = Conv(inc, inc // 2, 3, 2)
        self.down_branch2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(inc, inc // 2, k=1)
        )
        
    def forward(self, x):
        channel_gate = self.gate(self.globalpool(x))
        x_up = torch.cat([self.down_branch1(x), self.down_branch2(x)], dim=1) * channel_gate
        output = self.conv(x_up)
        return output