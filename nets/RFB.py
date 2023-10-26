import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.Attention import CA_Att,SCse,CBAM,GAM_Att,SE,CriCro,ECA

class BasicConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.oup = oup
        self.conv = nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(oup, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicSepConv(nn.Module):
    def __init__(self, inp, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicSepConv, self).__init__()
        self.oup = inp
        self.conv = nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=inp, bias=bias)
        self.bn = nn.BatchNorm2d(inp, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class RFB(nn.Module):
    def __init__(self, dim_in, dim_out, stride=1, scale=0.1):
        super(RFB, self).__init__()
        self.scale = scale
        self.dim_out = dim_out
        inter_planes = dim_in // 4

        self.branch0 = nn.Sequential(
            BasicConv(dim_in, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1, relu=False)

        )
        self.branch1 = nn.Sequential(
            BasicConv(dim_in, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)

        )
        self.branch2 = nn.Sequential(
            BasicConv(dim_in, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(1,3), stride=stride, padding=(0, 1)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)

        )
        self.branch3 = nn.Sequential(
            BasicConv(dim_in, inter_planes // 2, kernel_size=1, stride=1),
            BasicConv(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0 ,1)),
            BasicConv((inter_planes // 4) * 3, inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)

        )

        self.ConvLinear = BasicConv(4*inter_planes, dim_out, kernel_size=1, stride=1, relu=False)

        if dim_in == dim_out:
            self.identity = True
        else:
            self.identity = False
            self.shortcut = BasicConv(dim_in, dim_out, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)

        if self.identity:
            out = out*self.scale + x
        else:
            short = self.shortcut(x)
            out = out*self.scale +short
        out = self.relu(out)

        return out













