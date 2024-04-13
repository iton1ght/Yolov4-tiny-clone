import math
import numpy as np
import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]], label_smoothing = 0):
        super().__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.label_smoothing = label_smoothing

        self.balance   = [0.4, 1.0, 4.0]
        self.box_ratio = 0.05
        self.obj_ratio = 5 * (input_shape[0] * input_shape[1]) / (416**2)
        self.cls_ratio = 1 * (num_classes / 80)

        self.ignore_threshord = 0.5
        self.cuda = cuda
    # 定义张量裁剪函数，可将张量的每个元素裁剪至[t_min,t_max]区间
    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * t + (result > t_max).float() * t_max
        return result
    # 定义MSELoss损失函数，预测值与目标值的差值的平方
    def MSELoss(self, pred, target):
        return torch.pow(pred - target,2)
    # 定义BSELoss损失函数，二元交叉熵
    def BSELoss(self, pred, target):
        espsion = 1e-7
        pred = self.clip_by_tensor(pred, espsion, 1.0 - espsion)
        result = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return result
    # 真实框和预测框交并比计算方法，输入为5维张量，返回值为4维张量
    def box_ciou(self, box_1, box_2):
        """
        函数定义：计算真实框张量和预测框张量的所有CIoU
        输入为：
        ----------
        box_1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        box_2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

        返回为：
        -------
        ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        # 求真实框左上角和右下角坐标
        box_1_xy   = box_1[..., :2]
        box_1_wh   = box_1[..., 2:4]
        box_1_mins = box_1_xy - box_1_wh / 2.
        box_1_maxs = box_1_xy + box_1_wh / 2.

        # 求预测框左上角和右下角坐标
        box_2_xy   = box_2[..., :2]
        box_2_wh   = box_2[..., 2:4]
        box_2_mins = box_2_xy - box_2_wh / 2.
        box_2_maxs = box_2_xy + box_1_wh / 2.

        # 求真实框和预测框‘所有的’交集的左上角和右下角坐标，并求交集面积和并集面积，求得交并比
        intersect_mins = torch.max(box_1_mins, box_2_mins)
        intersect_maxs = torch.min(box_1_maxs, box_2_maxs)
        intersect_wh   = torch.max(intersect_maxs - intersect_mins, torch.zeros_like(intersect_maxs))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_1_area     = box_1_wh[..., 0] * box_1_wh[..., 1]
        box_2_area     = box_2_wh[..., 0] + box_2_wh[..., 1]
        union_area     = box_1_area +box_2_area -intersect_area
        iou            = intersect_area / torch.clamp(union_area,1e-6)

        # 求两个框中心点距离，为了简化运算避免开方，则求取中心点距离的平方
        center_distance = torch.sum(torch.pow(box_1_xy - box_2_xy, 2), dim = -1) #此处指定最后一维为计算对象，center-distance的张量形状与iou一致

        # 计算包裹真实框和预测框的最小包络框,求取包络框的左上角和右下角
        enclose_mins   = torch.min(box_1_mins, box_2_mins)
        enclose_maxs   = torch.max(box_1_maxs, box_2_maxs)
        enclose_wh     = torch.max(enclose_maxs - enclose_mins, torch.zeros_like(enclose_maxs))

        # 求包络框的对角线距离
        enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), dim = -1)

        # 将中心线比例引入交并比iou，center_distance越小,ciou越大
        # 其中利用包络框的对角线距离对center_distance进行归一化操作
        # （注：也可以用其中固定值进行归一化，感觉用enclose_diagonal进行归一化，可以‘减缓’center_distance参数对ciou的影响，因为center_distance和enclose_diagonal是同向变化的）
        ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, 1e-6)

        # 将真实框和预测框的宽长比引入交并比，从而体现两个框的形状重合特性
        box_wh_ratio = (4 / (math.pi ** 2)) * torch.pow((torch.atan(box_1_wh[..., 0] / torch.clamp(box_1_wh[..., 1], 1e-6)) - torch.atan(box_2_wh[..., 0] / torch.clamp(box_2_wh[..., 1], 1e-6))), 2)
        # 系数alpha用于调整宽高比损失项的权重，当iou减小时，alpha增大，代表增大权重，使得ciou快速降低，因为若无重合，即使形状相似也无用
        # alpha是关于box_wh_ratio的增函数，此处为归一化操作
        alpha = box_wh_ratio / torch.clamp((1.0 - iou +box_wh_ratio), 1e-6)
        ciou = ciou - alpha * box_wh_ratio
        return ciou
    def box_iou(self, box_a, box_b):
        """
        函数定义：计算一张图片上的所有真实框和先验框的IoU
        输入为：
         ----------
        box_a: tensor, shape=(gt_num, 4), xywh
        box_b: tensor, shape=(anchor_num, 4), xywh

        返回为：
        -------
        iou: tensor, shape=(gt_num, anchor_num, 1)
        """
    
    def smooth_labels(self, y_ture, label_smoothing, num_classes):

    def forward(self, l, input, targets = None):



    def get_target(self, l, target, anchors, in_h, in_w):

    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):

def weight_init():

def get_lr_scheduler():

def set_optimizer_lr():