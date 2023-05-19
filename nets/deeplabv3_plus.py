import onnxruntime
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import initialize_weights
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2
from nets.resnet import resnet50,resnet101,resnext50_32x4d
from nets.mobilenetv3 import mobilenetv3_small, mobilenetv3_large
from nets.Attention import CA_Att,SCse,CBAM,GAM_Att,SE,CriCro,ECA
from nets.RFB import RFB

class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet101', pretrained=True):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            initialize_weights(self.layer0)
        else:
            self.layer0 = nn.Sequential(*list(model.children())[:4])

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if output_stride == 16:
            s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8:
            s3, s4, d3, d4 = (1, 1, 2, 4)

        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'downsample.0' in n:
                    m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'downsample.0' in n:
                m.stride = (s4, s4)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        low_level_features = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_features


class MobileNetv3_Large(nn.Module):
    def __init__(self, downsample_factor=16, pretrained=True):
        super(MobileNetv3_Large, self).__init__()
        from functools import partial
        model = mobilenetv3_large(pretrained)
        self.features = model.features[:-1]
        self.total_idx = len(self.features)
        self.down_idx =[2, 4, 7, 13]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x

class MobileNetv3_Small(nn.Module):
    def __init__(self, downsample_factor=16, pretrained=True):
        super(MobileNetv3_Small, self).__init__()
        from functools import partial
        model = mobilenetv3_small(pretrained)
        self.features = model.features[:-1]
        self.total_idx = len(self.features)
        self.down_idx = [1, 2, 4, 9]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=16, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial
        
        model           = mobilenetv2(pretrained)
        self.features   = model.features[:-1]

        self.total_idx  = len(self.features)
        self.down_idx   = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch4 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
		self.branch5_relu = nn.ReLU(inplace=True)

		self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)

	def forward(self, x):
		[b, c, row, col] = x.size()
        #-----------------------------------------#
        #   一共五个分支
        #-----------------------------------------#
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
        #-----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        #-----------------------------------------#
		global_feature = torch.mean(x,2,True)
		global_feature = torch.mean(global_feature,3,True)
		global_feature = self.branch5_conv(global_feature)
		global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        #-----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        #-----------------------------------------#
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result

class DeepLab(nn.Module):
    def __init__(self, num_classes=10, backbone="mobilenetv3_large", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone=="xception":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            #----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone=="mobilenet":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            #----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        elif backbone=="mobilenetv3_small":
            self.backbone = MobileNetv3_Small(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 96
            low_level_channels = 24
        elif backbone=="mobilenetv3_large":
            self.backbone = MobileNetv3_Large(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 160
            low_level_channels = 24
        elif backbone=="resnet101":
            self.backbone = ResNet(downsample_factor ,pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone=="resnet50":
            self.backbone = ResNet(downsample_factor ,pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        #-----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        #-----------------------------------------#
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        # self.rfb = RFB(dim_in=in_channels, dim_out=256)

        #----------------------------------#
        #   浅层特征边
        #----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            # nn.GroupNorm(6, 48),
            nn.ReLU6(inplace=True),
            # CA_Att(48, 48),
            # CriCro(48),
            # SCse(48),
            # CBAM(48, 48),
            # GAM_Att(48, 48),
        )		

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            # nn.GroupNorm(8, 256),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            # nn.GroupNorm(8, 256),
            nn.ReLU6(inplace=True),
            # CA_Att(256, 256),
            # CriCro(256),
            # SCse(256),
            # CBAM(256, 256),
            # GAM_Att(256, 256),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)
        # self.SCse = SCse(num_classes)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        #-----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        #-----------------------------------------#
        low_level_features, x = self.backbone(x)
        # x = self.rfb(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)
        
        #-----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        #-----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        # x = self.SCse(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

# from torchstat import stat
# model = DeepLab()
# stat(model, (3, 512, 512))


'''
if __name__ == '__main__':
    model = DeepLab()

    # 查看网络结构
    print(model)

    # 打印模型参数
    for param in model.parameters():
        print(param)

    # 打印模型名称与shape
    for name,parameters in model.named_parameters():
        print(name,':',parameters.size())

def get_parameter_number(self,model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

a = get_parameter_number(DeepLab)
print(a)
'''
'''
from torchstat import stat
model = DeepLab()
stat(model, (3, 128, 128))
'''