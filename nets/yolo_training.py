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
        box_b: tensor, shape=(anchors_num, 4), xywh

        返回为：
        -------
        iou: tensor, shape=(gt_num, anchors_num, 1)
        """
        # 计算真实框的左上角和右下角坐标
        box_a_x1, box_a_x2 = box_a[:, 0] - box_a[:, 2] / 2, box_a[:, 0] + box_a[:, 2] / 2
        box_a_y1, box_a_y2 = box_a[:, 1] - box_a[:, 3] / 2, box_a[:, 1] + box_a[:, 3] / 2

        # 计算先验框的左上角和右下角坐标
        box_b_x1, box_b_x2 = box_b[:, 0] - box_b[:, 2] / 2, box_b[:, 0] + box_b[:, 2] / 2
        box_b_y1, box_b_y2 = box_b[:, 1] - box_b[:, 3] / 2, box_b[:, 1] + box_b[:, 3] / 2

        # 将真实框和先验框张量最后一维转化为左上角和右上角坐标形式，x1，y1, x2, y2
        _box_a = torch.zeros_like(box_a)
        _box_b = torch.zeros_like(box_b)
        _box_a[:, 0], _box_a[:, 1], _box_a[:, 2], _box_a[:, 3] = box_a_x1, box_a_y1, box_a_x2, box_a_y2
        _box_b[:, 0], _box_b[:, 1], _box_b[:, 2], _box_b[:, 3] = box_b_x1, box_b_y1, box_b_x2, box_b_y2

        # 计算真实框和先验框数量
        A = _box_a.size(0)
        B = _box_b.size(0)

        # 计算每个真实框与每个先验框的iou
        # 先将两个张量转化为相同维度，再进行计算
        # 计算所有交框的左上角和右下角坐标，再计算交框的面积
        intersect_mins = torch.max(_box_a[:, :2].unsqueeze(1).expand(A, B, 2), _box_b[:, :2].unsqueeze(0).expand(A, B, 2)) #[A, B, 2]
        intersect_maxs = torch.min(_box_a[:, 2:].unsqueeze(1).expand(A, B, 2), _box_b[:, 2:].unsqueeze(1).expand(A, B, 2)) #[A, B, 2]
        intersect_wh   = torch.max(intersect_maxs - intersect_mins, torch.zeros_like(intersect_maxs)) #[A, B, 2]
        intersect_area = intersect_wh[:, :, 0] * intersect_wh[:, :, 1] #[A, B]

        #计算并框的面积，再计算交并比
        box_a_area = ((_box_a[:, 2] - _box_a[:, 0]) * (_box_a[:, 3] - _box_a[:, 1])).unsqueeze(1).expand(A, B) #[A, B]
        box_b_area = ((_box_b[:, 2] - _box_b[:, 0]) * (_box_b[:, 3] - _box_b[:, 1])).unsqueeze(0).expand(A, B) #[A, B]
        iou = intersect_area / (box_a_area + box_b_area - intersect_area) #[A, B]

        return iou

    def smooth_labels(self, y_ture, label_smoothing, num_classes):
        """
        函数定义：平滑真实的标签值，略小于1，且略大于0
        """
        return y_ture * (1- label_smoothing) + label_smoothing / num_classes
    #定义方法，计算y_true，noobj_mask, box_loss_scale
    def get_target(self, l, target, scale_anchors, in_h, in_w):
        """
        函数定义:
        :param l: l代表特征图序号，即选择第l个特征图
        :param target: 代表真实框数据标签，包括批次，真实框数量，以及真实框坐标和类别信息，shape=[bs, gt_num, 5], 5->xywhc
        :param anchors: 代表先验框列表
        :param in_h: 特征图高度
        :param in_w: 特征图宽度

        :return
        y_true:创建一个与模型输出形状相同的张量，用于存储真实的标签信息。在训练过程中，这个张量将用于计算损失函数，并反向传播到网络中。
        noobj_mask:创建一个张量，用于标记哪些先验框不包含物体（即负样本）。在训练过程中，这些负样本通常对损失函数的贡献较小，有助于平衡正负样本的影响
        box_loss_scale:用于调整不同大小的物体在损失函数中的权重，特别是用于让网络更加关注小目标。小目标在特征图中通常只占据少量像素，因此可能需要额外的权重来确保它们得到足够的关注。
        """
        # 获取批次大小
        bs = target.size(0)
        # 获取该特征图下的先验框数量
        anchors_num = len(self.anchors_mask[l])
        # 创建一个张量，标记哪些先验框不包含物体，初始化时全都不包含，即张量元素均为1
        noobj_mask = torch.ones(bs, anchors_num, in_h, in_w, requires_grad=False)
        # 创建一个张量，用于调整不同大小物体在损失函数中的权重，让网络更加关注小目标
        box_loss_scale = torch.zeros(bs, anchors_num, in_h, in_w, requires_grad=False)
        # 创建一个张量，与模型输出形状相同，存储真实的标签信息
        y_true =torch.zeros(bs, anchors_num, in_h, in_w, self.bbox_attrs, requires_grad=False)

        # 对批次内每张图片的每个真实框与预设所有先验框进行交并比计算，从而选出最大交并比的先验框，作为该真实框的最初尺寸分类
        for b in range(bs):
            if target[b].size(0)==0:
                continue
            batch_target = torch.zeros_like(target[b])
            # 计算出原图中正样本在特征图上的中心点和宽高
            # 这里的target中的xywh可能为归一化坐标，故乘以特征图尺寸还原到特征图的像素坐标,因为后续先验框的尺寸为像素坐标，故两者统一
            batch_target[:, [0, 2]] = target[b][:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = target[b][:, [1, 3]] * in_h
            # 提取真实框的类别序号
            batch_target[:, 4] = target[b][:, 4]

            # 将真实框张量转换形式，中心坐标归0（先验框无中心坐标，所以两者都取0），即移至坐标原点,方便后续计算交并比,代入到box_iou进行计算
            # 真实框张量, shape = [gt_num, 4]
            gt_box = torch.cat((torch.zeros(batch_target.size(0), 2), batch_target[:, 2:4]), 1)
            # 先验框张量，shape = [anchors_num, 4]，这里的先验框数量取的是总数量，并非子列表的先验框数量,按照该类的先验框初始化情况，应为9个
            anchors_box = torch.cat((torch.zeros(len(scale_anchors), 2), torch.tensor(scale_anchors)),1)
            # 计算交并比
            # self.calculate_iou(gt_box, anchors_box) = [num_true_box, 9]每一个真实框和9个先验框的重合情况
            iou = self.box_iou(gt_box, anchors_box) #[gt_num, anchors_num]
            # best_ns:
            # [每个真实框最大的重合度max_iou, 每一个真实框最重合的先验框的序号]
            best_ns = torch.argmax(iou, dim=-1)
            sort_ns = torch.argsort(iou, dim=-1, descending=True)

            # 检查最重合先验框的序号是否在先验框掩码里
            def check_in_anchors_mask(index, anchors_mask):
                for sub_anchors_mask in anchors_mask:
                    if best_ns in sub_anchors_mask:
                        return True
                return False
            for t, best_n in enumerate(best_ns):
                if not check_in_anchors_mask(best_n, self.anchors_mask):
                    for index in sort_ns[t]:
                        if check_in_anchors_mask(index, self.anchors_mask):
                            best_n = index
                            break

                if best_n not in self.anchors_mask:
                    continue
                # 判断当前先验框是当前特征点的哪一个先验框，即确定当前特征点的最重合先验框的序号
                k = self.anchors_mask[l].index(best_n)
                # 获得真实框属于哪个网格点
                i = torch.floor(batch_target[t, 0]).long()
                j = torch.floor(batch_target[t, 1]).long()
                # 取出该真实框的标签种类，c从0开始，0对应第一个类别，依次类推
                c = batch_target[t, 4].long()
                # noobj_mask代表无目标的掩码，初始值均为1，若当前图片、当前先验框、当前特征点有目标，则置为0
                noobj_mask[b, k, j, i] = 0
                # 真实标签的张量y_true, shape=[bs, anchors_num, in_h, in_w, bbox_attrs]
                # 将真实框的中心点、宽高储存到当前特征点上
                y_true[b, k, j, i, 0] = batch_target[t, 0]
                y_true[b, k ,j ,i, 1] = batch_target[t, 1]
                y_true[b, k ,j ,i, 2] = batch_target[t, 2]
                y_true[b, k, j, i, 3] = batch_target[t, 3]
                # 将最后一维第5个元素置为1，即当前特征点一定含有目标，置信度为100%；将第c+5元素置为1，即当前特征点的该类别输出目标置1，其余类别输出目标仍为0
                y_true[b, k, j, i, 4] = 1
                y_true[b, k, j, i, c + 5] = 1

                # 损失函数中的不平衡：在目标检测任务中，尤其是当场景中同时存在大目标和小目标时，直接计算所有目标的损失可能会导致网络更关注大目标。
                # 这是因为大目标通常占据更多的像素，因此在损失计算中贡献更大。这可能导致网络对小目标的检测性能不佳。
                # 为了解决这个问题，可以通过给不同大小的目标分配不同的权重来调整损失函数。
                # 对于小目标，可以给予更高的权重，这样网络在训练时会更加关注小目标的预测误差，并尝试减小这些误差。
                # 此处计算存在归一化操作，归一化的宽在0-1之间，高在0-1之间，乘积也在
                box_loss_scale = 2-batch_target[t, 2] * batch_target[t, 3] / in_h / in_w
        return y_true, noobj_mask, box_loss_scale
    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):
    def forward(self, l, input, targets=None):

def weight_init():

def get_lr_scheduler():

def set_optimizer_lr():