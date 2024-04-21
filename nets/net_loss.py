import functools
import math
import numpy as np
import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]], label_smoothing=0) -> object:
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
        return torch.pow(pred - target, 2)

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
        box_1: tensor, shape=(batch, anchor_num, feat_w, feat_h, 4), xywh
        box_2: tensor, shape=(batch, anchor_num, feat_w, feat_h, 4), xywh

        返回为：
        -------
        ciou: tensor, shape=(batch, anchor_num, feat_w, feat_h, 1)
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
        union_area     = box_1_area + box_2_area - intersect_area
        iou            = intersect_area / torch.clamp(union_area, 1e-6)

        # 求两个框中心点距离，为了简化运算避免开方，则求取中心点距离的平方
        center_distance = torch.sum(torch.pow(box_1_xy - box_2_xy, 2), dim=-1)  # 此处指定最后一维为计算对象，center-distance的张量形状与iou一致

        # 计算包裹真实框和预测框的最小包络框,求取包络框的左上角和右下角
        enclose_mins   = torch.min(box_1_mins, box_2_mins)
        enclose_maxs   = torch.max(box_1_maxs, box_2_maxs)
        enclose_wh     = torch.max(enclose_maxs - enclose_mins, torch.zeros_like(enclose_maxs))

        # 求包络框的对角线距离
        enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), dim=-1)

        # 将中心线比例引入交并比iou，center_distance越小,ciou越大
        # 其中利用包络框的对角线距离对center_distance进行归一化操作
        # （注：也可以用其中固定值进行归一化，感觉用enclose_diagonal进行归一化，可以‘减缓’center_distance参数对ciou的影响，因为center_distance和enclose_diagonal是同向变化的）
        ciou = iou - 1.0 * center_distance / torch.clamp(enclose_diagonal, 1e-6)

        # 将真实框和预测框的宽长比引入交并比，从而体现两个框的形状重合特性
        box_wh_ratio = (4 / (math.pi ** 2)) * torch.pow((torch.atan(box_1_wh[..., 0] / torch.clamp(box_1_wh[..., 1], 1e-6)) - torch.atan(box_2_wh[..., 0] / torch.clamp(box_2_wh[..., 1], 1e-6))), 2)
        # 系数alpha用于调整宽高比损失项的权重，当iou减小时，alpha增大，代表增大权重，使得ciou快速降低，因为若无重合，即使形状相似也无用
        # alpha是关于box_wh_ratio的增函数，此处为归一化操作
        alpha = box_wh_ratio / torch.clamp((1.0 - iou + box_wh_ratio), 1e-6)
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
        intersect_mins = torch.max(_box_a[:, :2].unsqueeze(1).expand(A, B, 2), _box_b[:, :2].unsqueeze(0).expand(A, B, 2))  # [A, B, 2]
        intersect_maxs = torch.min(_box_a[:, 2:].unsqueeze(1).expand(A, B, 2), _box_b[:, 2:].unsqueeze(1).expand(A, B, 2))  # [A, B, 2]
        intersect_wh   = torch.max(intersect_maxs - intersect_mins, torch.zeros_like(intersect_maxs))  # [A, B, 2]
        intersect_area = intersect_wh[:, :, 0] * intersect_wh[:, :, 1]  # [A, B]

        # 计算并框的面积，再计算交并比
        box_a_area = ((_box_a[:, 2] - _box_a[:, 0]) * (_box_a[:, 3] - _box_a[:, 1])).unsqueeze(1).expand(A, B)  # [A, B]
        box_b_area = ((_box_b[:, 2] - _box_b[:, 0]) * (_box_b[:, 3] - _box_b[:, 1])).unsqueeze(0).expand(A, B)  # [A, B]
        iou = intersect_area / (box_a_area + box_b_area - intersect_area)  # [A, B]

        return iou

    def smooth_labels(self, y_ture, label_smoothing, num_classes):
        """
        函数定义：平滑真实的标签值，略小于1，且略大于0
        """
        return y_ture * (1 - label_smoothing) + label_smoothing / num_classes
    # 定义方法，计算y_true，noobj_mask, box_loss_scale

    def get_target(self, l, targets, scale_anchors, in_h, in_w):
        """
        函数定义: 待填充
        :param l: l代表特征图序号，即选择第l个特征图
        :param targets: 代表真实框数据标签，包括批次，真实框数量，以及真实框坐标和类别信息，shape=[bs, gt_num, 5], 5->xywhc
        :param scale_anchors: 在特征图尺度上的先验框列表, 即原anchors进行缩放
        :param in_h: 特征图高度
        :param in_w: 特征图宽度

        :return
        y_true:创建一个与模型输出形状相同的张量，用于存储真实的标签信息。在训练过程中，这个张量将用于计算损失函数，并反向传播到网络中。
        noobj_mask:创建一个张量，用于标记哪些先验框不包含物体（即负样本）。在训练过程中，这些负样本通常对损失函数的贡献较小，有助于平衡正负样本的影响
        box_loss_scale:用于调整不同大小的物体在损失函数中的权重，特别是用于让网络更加关注小目标。小目标在特征图中通常只占据少量像素，因此可能需要额外的权重来确保它们得到足够的关注。
        """
        # 获取批次大小
        bs = targets.size(0)
        # 获取该特征图下的先验框数量
        anchors_num = len(self.anchors_mask[l])
        # 创建一个张量，标记哪些先验框不包含物体，初始化时全都不包含，即张量元素均为1
        noobj_mask = torch.ones(bs, anchors_num, in_h, in_w, requires_grad=False)
        # 创建一个张量，用于调整不同大小物体在损失函数中的权重，让网络更加关注小目标
        box_loss_scale = torch.zeros(bs, anchors_num, in_h, in_w, requires_grad=False)
        # 创建一个张量，与模型输出形状相同，存储真实的标签信息
        y_true = torch.zeros(bs, anchors_num, in_h, in_w, self.bbox_attrs, requires_grad=False)

        # 对批次内每张图片的每个真实框与预设所有先验框进行交并比计算，从而选出最大交并比的先验框，作为该真实框的最初尺寸分类
        for b in range(bs):
            if targets[b].size(0) == 0:
                continue
            batch_target = torch.zeros_like(targets[b])
            # 计算出原图中正样本在特征图上的中心点和宽高
            # 这里的target中的xywh可能为归一化坐标，故乘以特征图尺寸还原到特征图的像素坐标,因为后续先验框的尺寸为像素坐标，故两者统一
            batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
            # 提取真实框的类别序号
            batch_target[:, 4] = targets[b][:, 4]

            # 将真实框张量转换形式，中心坐标归0（先验框无中心坐标，所以两者都取0），即移至坐标原点,方便后续计算交并比,代入到box_iou进行计算
            # 真实框张量, shape = [gt_num, 4]
            gt_box = torch.cat((torch.zeros(batch_target.size(0), 2), batch_target[:, 2:4]), 1)
            # 先验框张量，shape = [anchors_num, 4]，这里的先验框数量取的是总数量，并非子列表的先验框数量,按照该类的先验框初始化情况，应为9个
            anchors_box = torch.cat((torch.zeros(len(scale_anchors), 2), torch.tensor(scale_anchors)), 1)
            # 计算交并比
            # self.calculate_iou(gt_box, anchors_box) = [num_true_box, 9]每一个真实框和9个先验框的重合情况
            iou = self.box_iou(gt_box, anchors_box)  # [gt_num, anchors_num]
            # best_ns:
            # [每个真实框最大的重合度max_iou, 每一个真实框最重合的先验框的序号]
            best_ns = torch.argmax(iou, dim=-1)
            sort_ns = torch.argsort(iou, dim=-1, descending=True)

            # 检查最重合先验框的序号是否在先验框掩码里
            def check_in_anchors_mask(index, anchors_mask):
                for sub_anchors_mask in anchors_mask:
                    if index in sub_anchors_mask:
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
                y_true[b, k, j, i, 1] = batch_target[t, 1]
                y_true[b, k, j, i, 2] = batch_target[t, 2]
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
        """
        函数定义：在当前特征图l下，根据先验框以及模型输出值，求得预测框张量，并将每个预测框与真实框求交并比，将最大交并比的预测框（先验框）设为大概率有目标，故noobj_mask掩码对应位置置0，后续不参与损失计算
        :param l: l代表特征图序号，即选择第l个特征图
        :param x: 经过神经网络计算输出x，x代表先验框的调整量，从而得到预测框x
        :param y: 同x
        :param h: 同x，注意宽高调整量不能直接相加，因为已经经过对数处理，还原的话要在经指数处理
        :param w: 同h
        :param targets: 代表真实框数据标签，包括批次，真实框数量，以及真实框坐标和类别信息，shape=[bs, gt_num, 5], 5->xywhc
        :param scaled_anchors: 在特征图尺度上的先验框列表, 即原anchors进行缩放
        :param in_h: 特征图高度
        :param in_w: 特征图宽度
        :param noobj_mask: 无目标掩码
        :return:
        pred_boxes: 预测框张量，shape=[bs, len(anchors_mask[l]), in_h, in_w, 4]
        noobj_mask: 无目标掩码
        """
        # 获得批次大小
        bs = targets.size(0)
        # 生成网格，网格的左上角坐标即为先验框的中心
        grid_x = torch.linspace(0, in_w-1, in_w).repeat(in_h, 1).repeat(
            int(bs*len(self.anchors_mask[l])), 1, 1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0, in_h-1, in_h).repeat(in_w, 1).t().repeat(
            int(bs*len(self.anchors_mask[l])), 1, 1).view(y.shape).type_as(x)

        # 生成先验框的在特征图尺度下的宽高，并转化为与输出相同的张量形状
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]  # scaled_anchors为元组列表，先转化为二维数组，并取出当前l掩码的宽高，一行代表一组宽高
        anchor_w = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([0])).type_as(x)
        anchor_h = torch.Tensor(scaled_anchors_l).index_select(1, torch.LongTensor([1])).type_as(x)

        # 转换张量形状，用于后续计算
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h, in_w).view(w.shapes)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h, in_w).view(h.shapes)

        # 计算调整之后的先验框的中心和宽高，也就是预测框的中心和宽高
        pred_boxes_x = torch.unsqueeze(grid_x + x, -1)
        pred_boxes_y = torch.unsqueeze(grid_y + y, -1)
        pred_boxes_w = torch.unsqueeze(torch.exp(w) + anchor_w, -1)
        pred_boxes_h = torch.unsqueeze(torch.exp(h) + anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_h, pred_boxes_w], -1)

        # 求得更新后的noobj_mask
        for b in range(bs):
            # 将第b批次的预测框张量转换形式，[len(anchors_mask[l]), in_h, in_w, 4] ->[anchors_sum_num, 4]
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)  # B=len(anchors_mask[l])*in_h*in_w

            # 计算真实框张量，并转换为特征图尺度大小，shape=[gt_num, 4], A=4
            batch_target = torch.zeros_like(targets[b])
            batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
            batch_target[:, :4] = batch_target[:, :4].type_as(x)

            # 计算所有预测框和所有真实框的交并比
            # 对于每一个预测框，求取最大交并比的真实框iou数值
            iou = self.box_iou(batch_target, pred_boxes_for_ignore)  # iou shape=[A, B]
            max_iou, _ = torch.max(iou, dim=0)  # max_iou为一维张量，大小为B
            # 将张量max_iou再转换为与第b批次的pred_boxes相同的形状
            max_iou = max_iou.view(pred_boxes[b].size()[:3])  # B -> [len(anchors_mask[l]), in_h, in_w]

            # 将max_iou的值与预设的ignore_threshord比较，大于预设交并比时为ture，形成一个布尔掩码。
            # 利用布尔掩码对noobj_mask对应位置进行操作，如果为真则置0
            # 该操作将预测框交并比较大的不予进入损失计算
            noobj_mask[b][max_iou > self.ignore_threshord] = 0

        return pred_boxes, noobj_mask
    def forward(self, l, input, targets=None):
        """

        :param l: 第几个特征图序号
        :param input: 特征图输入，yolov4-tiny有两个特征图
                      l=0时，shape = [bs, 3*(5+classes_num), 13, 13]
                      l=1时，shape = [bs, 3*(5+classes_num), 26, 26]
        :param targets: 真实图输入，即真实标签输入情况
                      shape = [bs, gt_num, 5]
        :return:
        loss: 输出损失值
        """
        # 获得批次大小，特征图宽和高的大小
        bs   = input.size(0)
        in_w = input.size(2)
        in_h = input.size(3)

        # 计算步长
        # 每一个特征点对应原来的图片上多少个像素点
        # 如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        # 如果特征层为26x26的话，一个特征点就对应原来的图片上的16个像素点
        # stride_h = stride_w = 32、16
        stride_w = self.input_shape[1] / in_w
        stride_h = self.input_shape[0] / in_h

        # 计算scale_anchors，这个是相对特征图的先验框尺寸
        scale_anchors = [(anchor_w / stride_w, anchor_h / stride_h) for anchor_w, anchor_h in self.anchors]

        # 将输入的input张量进行转换，input张量一共有两个，对应两个特征图。将四维张量转换为五维。
        # shape: [bs, 3*(5+classes_num), 13, 13] -> [bs, 3, 13, 13, 5+classes_num]
        # shape: [bs, 3*(5+classes_num), 26, 26] -> [bs, 3, 26, 26, 5+classes_num]
        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # 先验框的中心调整参数，利用sigmoid函数将其映射至0-1之间，使其不会超出一个网格。后面再加上网格坐标，则得到预测框中心
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])

        # 先验框的宽高调整参数
        w = prediction[..., 2]
        h = prediction[..., 3]

        # 由先验框得到的预测框的位置置信度
        conf = torch.sigmoid(prediction[..., 4])

        # 由先验框得到的预测框的种类置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # 获得网络应该得到的真实的预测结果，即目标值，y_true为真实框张量
        y_true, noobj_mask, box_loss_scale = self.get_target(l, targets, scale_anchors, in_h, in_w)

        # 将预测结果进行解码，获得网络根据先验框和计算结果得到的预测框, pred-boxes为预测框张量
        # 判断预测结果和真实值的重合程度，如果重合程度过大则忽略，因为这些特征点属于预测比较准确的特征点，作为负样本不合适
        pred_boxes, noobj_mask = self.get_ignore(l, x, y, h, w, targets, scale_anchors, in_h, in_w, noobj_mask)

        if self.cuda:
            y_true          = y_true.type_as(x)
            noobj_mask      = noobj_mask.type_as(x)
            box_loss_scale  = box_loss_scale.type_as(x)
        # --------------------------------------------------------------------------#
        #   box_loss_scale是真实框宽高的乘积，宽高均在0-1之间，因此乘积也在0-1之间。
        #   2-宽高的乘积代表真实框越大，比重越小，小框的比重更大。
        #   使用iou损失时，大中小目标的回归损失不存在比例失衡问题，故弃用
        # --------------------------------------------------------------------------#
        box_loss_scale = 2 - box_loss_scale

        # 计算损失
        loss = 0
        # 生成布尔掩码，真实框上为正样本，即相应位置置信度为1，即有目标的位置为True，其余不参与损失计算
        obj_mask = y_true[..., 4] == 1
        n = torch.sum(obj_mask)
        if n != 0:
            # 计算所有真实框和所有预测框的ciou
            ciou = self.box_ciou(y_true[..., :4], pred_boxes).type_as(x)
            # 计算回归损失，即定位回归损失值
            # 通过使用布尔索引，我们只选择那些 obj_mask 为 True 的位置上的 (1 - ciou) 值。这确保了只有正样本的预测框对损失有贡献。
            loss_loc = torch.mean((1 - ciou)[obj_mask])
            # 同理使用布尔索引，利用二元交叉熵损失函数计算分类损失
            loss_cls = torch.mean(self.BSELoss(pred_cls[obj_mask], y_true[..., 5:][obj_mask]))
            # 合并损失，引入权重
            loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio
        # 引入置信度损失
        # conf 与 obj_mask进行交叉熵损失计算，这计算了每个预测框的置信度损失。
        # 注意，这里实际上是将置信度预测与 obj_mask 作为目标值进行比较，这在YOLO中是一个常见的做法，因为YOLO将置信度解释为预测框内存在目标的概率。
        # noobj_mask用于标识哪些预测框没有与任何真实目标匹配（即负样本），由于在get_target和get_ignore函数中，对noobj_mask已更新，对部分位置不参与计算的预测框已经置0
        # noobj_mask为元素为0或1，noobj_mask.bool()将0转换为False，1转换为True
        # [noobj_mask.bool() | obj_mask]，这个逻辑或运算的结果是一个新的布尔数组，其中每个元素都是 noobj_mask 和 obj_mask 对应位置元素逻辑或的结果，形成新的布尔掩码
        # 因此置信度计算既考虑了正样本，也考虑负样本
        loss_conf = torch.mean(self.BSELoss(conf, obj_mask.type_as(conf))[noobj_mask.bool() | obj_mask])
        loss += loss_conf * self.balance[l] * self.obj_ratio
        return loss


# 定义函数weight_init(),初始化神经网络的权重
def weights_init(net, init_type='normal', init_gain=0.02):
    """

    :param net: 需要初始化的神经网络模型
    :param init_type: 初始化方法，默认为normal
    :param init_gain: 初始化增益，默认为0.02
    :return:
    对模型使用初始化方法，完成模型权重的初始化
    """
    def init_func(m):
        classname = m.__class__.__name__
        # 检测该层m是否包含权重属性'weight', 若包含则为TRUE
        # 检测该层m的类名中是否包含'Conv', 即是否为卷积层，若包含则返回索引，非-1，则逻辑判断为TRUE。若不包含则返回-1，逻辑判断为FALSE。
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if   init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, 0.0, 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        # 检查该层是否是批量归一化层
        # 初始化权重正态分布，初始化偏置为常数0
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)
        print('initialize network with %s type' % init_type)
        net.apply(init_func)


def get_lr_scheduler(lr_delay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1, no_aug_iter_ratio=0.05, step_num=10):
    """
    函数定义：函数 get_lr_scheduler，它用于获取学习率调度器。学习率调度器用于在训练神经网络时动态地调整学习率，以提高训练效率和模型性能。
    函数内部定义了两个内部函数 yolox_warm_cos_lr 和 step_lr，分别用于实现余弦退火策略和步长衰减策略。
    根据 lr_decay_type 的值，函数返回相应的学习率调度器函数 func
    :param lr_delay_type: 学习率衰减类型。如果为 "cos"，则使用余弦退火策略；否则，使用步长衰减策略。
    :param lr: 初始学习率。
    :param min_lr: 最小学习率，当使用余弦退火策略时，学习率将在这个值和初始学习率之间变化；当使用步长衰减策略时，学习率将逐渐衰减到这个值。
    :param total_iters: 总迭代次数，即整个训练过程的迭代次数
    :param warmup_iters_ratio: 预热迭代次数的比例，默认为0.05。预热阶段是在训练开始时逐渐增加学习率的过程，有助于模型更好地适应初始学习率。
    :param warmup_lr_ratio: 预热阶段开始时的学习率比例，默认为0.1。
    :param no_aug_iter_ratio: 不使用数据增强的迭代次数比例，默认为0.05。这部分迭代通常用于模型微调，此时可能不再使用数据增强。
    :param step_num: 步长衰减策略的步数，即学习率在每个步长后衰减一次。
    :return:
    func = functools.partial
    注： functools.partial是Python标准库中的一个函数，它的主要作用是部分应用一个函数，也就是固定函数的一部分参数，
    返回一个新的可调用对象。这个新的对象类似于原始函数，但其中的一些参数已经被预先设置，因此在后续调用中可以减少需要传递的参数数量
    """
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr = warmup_lr_start + (lr - warmup_lr_start) * pow(iters / warmup_total_iters, 2)
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi * ((iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))))
        return lr

    def step_lr(lr, delay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        else:
            n = iters // step_size
            lr = lr * delay_rate ** n
        return lr
    # 最后，根据 lr_decay_type 的值，函数返回相应的学习率调度器函数 func。如果 lr_decay_type 为 "cos"，
    # 则返回 yolox_warm_cos_lr 函数的部分应用（使用 functools.partial）；否则，返回 step_lr 函数的部分应用。
    if lr_delay_type == 'cos':
        warmup_toal_iters = min(max(total_iters * warmup_iters_ratio, 1), 3)
        warmup_lr_start = max(lr * warmup_lr_ratio, 1e-6)
        no_aug_iter = min(max(total_iters * no_aug_iter_ratio, 1), 15)
        func = functools.partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_toal_iters,
                                 warmup_lr_start, no_aug_iter)
    else:
        delay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = functools.partial(step_lr, lr, delay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    """
    函数定义定义了一个函数 set_optimizer_lr，它用于设置 PyTorch 优化器的学习率。这个函数接受三个参数：，lr_scheduler_func（学习率调度函数），和 epoch（当前的训练周期数）
    :param optimizer: optimizer（优化器对象）
    :param lr_scheduler_func: lr_scheduler_func（学习率调度函数）
    :param epoch: epoch（当前的训练周期数）
    """
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
