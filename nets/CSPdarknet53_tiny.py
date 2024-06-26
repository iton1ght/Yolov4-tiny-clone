import math
import torch
import torch.nn as nn

#定义基础卷积块
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        # 卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        # 标准层
        self.bn   = nn.BatchNorm2d(out_channels)
        # 激活层
        self.activation = nn.LeakyReLU(0.1)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
# 定义Resblock_body模块
# 模块的特点为存在一大一小残差边，大残差边为第一次卷积标准化激活的输出，小残差边为第二次卷积标准化激活的输出
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 定义该模块的输入通道数和输出通道数属性，对一个实例来说，输入为64，输出为128
        self.in_channels  = in_channels
        self.out_channels = out_channels
        # 卷积层
        self.conv1 = BasicConv(self.in_channels, self.in_channels, 3)
        self.conv2 = BasicConv(self.in_channels//2, self.in_channels//2, 3)
        self.conv3 = BasicConv(self.in_channels//2, self.in_channels//2, 3)
        self.conv4 = BasicConv(self.out_channels//2, self.out_channels//2, 1)

        # 二维最大池化层
        self.maxpool = nn.MaxPool2d([2,2], [2,2])
    def forward(self, x):
        # 第一个卷积3x3
        x = self.conv1(x)
        # 引出大残差边
        route_1 = x
        # 对特征层的通道进行分割，取第二部分进行主干运算，第二部分通道数为原来的一半
        c = self.in_channels
        x = torch.split(x, c//2, 1)[1] # 沿着第二个维度（通道维度）将张量 x 分割成大小为 c//2 的块，并取这些块中的第二个块作为新的 x。
        # 第二个卷积3x3
        x = self.conv2(x)
        # 引出小残差边
        route_2 = x
        # 第三个卷积3x3
        x = self.conv3(x)
        # 小残差边与主干进行通道维度合并
        x = torch.cat([x, route_2],1)
        # 第四个卷积1x1，卷积核为1时，通常为改变通道数
        x = self.conv4(x)
        # 输出特征层
        feat = x
        # 大残差边与主干进行通道维度合并
        x = torch.cat([x,route_1],1)
        # 进行池化层操作,对高和宽进行压缩
        x = self.maxpool(x)
        return x, feat

#定义CSPdarknet模块
class CSPDarkNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 两个标准化卷积模块
        # 第一个卷积3x3，416x416x3 -> 208x208x32
        self.conv1 = BasicConv(3, 32, 3,2)
        # 第二个卷积3x3, 208x208x32 -> 104x104x64
        self.conv2 = BasicConv(32, 64,3 ,2)
        # 三个resblock_body模块
        # 104x104x64 -> 52x52x128
        self.resblock_body1 = Resblock_body(64, 128)
        # 52x52x128 -> 26x26x256
        self.resblock_body2 = Resblock_body(128, 256)
        # 26x26x256 -> 13x13x512
        self.resblock_body3 = Resblock_body(256, 512)
        # 一个标准化卷积模块
        # 第三个卷积3x3，13x13x512 -> 13x13x512
        self.conv3 = BasicConv(512,512,3,1)

        # 定义初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels 
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x, _ = self.resblock_body1(x)
        x, _ = self.resblock_body2(x)
        x, feat1 = self.resblock_body3(x) #feat1 为26x26x256
        x = self.conv3(x)
        feat2 = x  #feat2 为13x13x512
        return feat1, feat2

def darknet53_tiny(pretrained,**kwargs):
    model = CSPDarkNet()
    # 如果pretrained为TRUE,则加载主干的预训练权重。若已经全模型加载了预训练权重，则pretrained的值无所谓，主干部分权重已经在全模型预训练权重中被加载了
    if pretrained:
        model.load_state_dict(torch.load("model_data/CSPdarknet53_tiny_backbone_weights.pth"))
    return model