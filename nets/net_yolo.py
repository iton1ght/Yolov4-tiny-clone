import torch
import torch.nn as nn

from nets.CSPdarknet53_tiny import darknet53_tiny

# yolo主体结构包括Backbone主干和其他（头部和上采样等）
# Backbone主干主要由三个Resblock_body组成

# 定义基础卷积块:Conv2d + BatchNorm2d + LeakyReLU
# 在PyTorch中，当你定义一个nn.Module的子类时，你通常实现一个forward方法，该方法定义了模型的前向传播。
# 当你创建这个类的实例并调用它时，PyTorch在背后实际上调用了forward方法。这就是为什么你可以使用类似model(input)的
# 语法来执行模型的前向传播，即使model的类定义中并没有__call__方法。
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

# 定义上采样模块
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
    def forward(self, x,):
        x = self.upsample(x)
        return x

# 定义头部模块
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, filters_list[0], kernel_size=3),
        nn.Conv2d(filters_list[0], filters_list[1], kernel_size=1, stride=1)
    )
    return m

# 定义yolo主体结构
class YoloBody(nn.Module):
    def __init__(self, anchor_mask, num_classes, phi=0, pretrained=False):
        super().__init__()

        self.phi = phi
        self.backbone = darknet53_tiny(pretrained)

        self.conv_for_P5  = BasicConv(512, 256, 1, 1)
        self.yolo_headP5 = yolo_head([512, len(anchor_mask[0])*(5+num_classes)], 256)

        self.upsample     = Upsample(256, 128)
        self.yolo_headP4 = yolo_head([256, len(anchor_mask[1])*(5+num_classes)], 384)

    def forward(self, x):

        feat1, feat2 = self.backbone(x)

        P5   = self.conv_for_P5(feat2)
        out0 = self.yolo_headP5(P5)

        P5_Upsample = self.upsample(P5)
        P4   = torch.cat([P5_Upsample, feat1], 1)
        out1 = self.yolo_headP4(P4)

        return out0, out1



