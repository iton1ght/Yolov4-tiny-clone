import torch
from torchvision.ops import nms
import numpy as np


class DecodeBox():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        """
        :param anchors: 先验框数组
        :param num_classes: 类别数量
        :param input_shape: 输出尺寸(416,416)
        :param anchors_mask: 先验框掩码

        """
        super().__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        # -------------------------------------------#
        # 13x13的特征层对应的anchor是[81,82],[135,169],[344,319]
        # 26x26的特征层对应的anchor是[10,14],[23,27],[37,58]
        # -------------------------------------------#
        self.anchors_mask = anchors_mask

    # -------------------------------------------#
    #
    # -------------------------------------------#
    def decode_box(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            # -------------------------------#
            # 输入的input共2个，他们的shape分别是：
            # batch_size, 3x(5+80)=255, 13, 13
            # batch_size, 3x(5+80)=255, 26, 26
            # 具体可以看模型结构net_yolo中的输出端形状
            # -------------------------------#
            # 获取批次大小，和特征图的高和宽
            batch_size = input.size(0)
            input_height = input.size(2)
            input_width = input.size(3)
            # 获取先验框由原图到特征图尺度下的缩放比例
            stride_h = input_height / self.input_shape[0]
            stride_w = input_width / self.input_shape[1]
            # 获取在特征图尺度下的先验框大小
            scaled_anchors = [(anchor_width * stride_w, anchor_height * stride_h)for anchor_width, anchor_height in self.anchors[self.anchors_mask[i]]]

            # -----------------------------------------#
            # 将输入的input张量转换形状
            # 利用view方法，只改变形状，不改变元素的物理顺序，再利用permute调整维度顺序
            # -----------------------------------------#
            prediction = input.view(batch_size, len(self.anchors_mask[i]), self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

            # 先验框的中心位置调整参数
            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])

            # 先验框的宽高调整参数
            w = prediction[..., 2]
            h = prediction[..., 3]

            # 获得位置置信度，是否有物体
            conf = torch.sigmoid(prediction[..., 4])

            # 获得每个种类置信度
            pred_cls = torch.sigmoid(prediction[..., 5:])

            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTentor if x.is_cuda else torch.LongTensor

            # ------------------------------------------#
            # 生成网格，先验框的中心，每个网格的左上角
            # 网格形状，batch_size, 3, 13, 13或batch, 3, 26, 26
            # ------------------------------------------#
            grid_x = torch.linespace(0, input_width-1, input_width).repeat(input_height, 1).repeat(
                     batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linespace(0, input_height-1, input_height).repeat(input_width, 1).t().repeat(
                     batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)

            # ------------------------------------------#
            # 按照网格格式生成先验框的宽高
            # batch_size, 3, 13, 13或batch, 3, 26, 26
            # ------------------------------------------#
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor[0])
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor[1])
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

            # ------------------------------------------#
            # 利用预测结果对先验框进行调整
            # 首先调整先验框的中心，先验框中心向右下角调整
            # 再调整先验框的宽高
            # ------------------------------------------#
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

            # ------------------------------------------#
            # 将输出结果归一化成小数的形式
            # 将预测框的中心和宽高分别除以特征图的宽高，相当于框与图的比例尺，进行归一化
            # output 的shape=[batch_size, 3*13*13, 85]或shape=[batch_size, 3*26*26, 85]
            # ------------------------------------------#
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                               conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
            outputs.append(output.data)

        return outputs

    # 该方法将筛选后的预测框进行处理，转化到原始图尺寸上
    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        # -------------------------------------#
        # 将宽度方向和高度放的坐标值互换，方便后续与图像的高与宽相乘
        # 在yolo中，凡是框都是宽->高的顺序，凡是图片都是高->宽的顺序
        # -------------------------------------#
        box_yx = box_xy[:, :, ::-1]
        box_hw = box_wh[:, :, ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        # ---------------------------------------#
        # 这部分涉及从特征图尺度到input_shape尺度再到new_shape尺度在到image_shape的转化
        # 函数输入的box_yx和box_hw是特征图尺度的归一化参数，由于特征图宽高比与input_shape宽高比相同，则可以直接同步（例如框的中心点坐标相对特征图的位置，转化为input_shape尺度是一致的）
        # 而new_shape和image_shape的宽高比是一致的，因此也可以直接同步。关键就在input_shape到new_shape这一步转换上。
        # 当由input_shape尺度转化为new_shape尺度是需要缩放，因为两个宽高比可能不同
        # 另外offset值可以这么理解，new_shape和input_shape左上角重合,因此相对input_shape的坐标要向左偏移才能转到new_shape上，并且还要进行缩放，转化到相对new_shape尺度的归一化
        # 例如相对input_shape的框的中心点x的值，其归一化值为x/input_shape，要转换到相对new_shape尺度上，则x/input_shape * input_shape/new_shape = x/new_shape
        # 举例，框在特征图的中心，宽高为0.1比例
        # 假设input_shape=(416,416),image_shape=(932,416),box_yx=(0.5,0.5),box_hw=(0.1,0.1)
        # 则new_shape=(416,208)
        # offset=(0,0.25),scale=(1,2)
        # box_yx=(0.5, 0.5),box_hw=(0.1,0.2)
        # ---------------------------------------#

        if letterbox_image:
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2.0 / input_shape
            scale = input_shape / new_shape
            # box_yx -offset的值为相对input_shape尺度的框中心坐标，再乘以scale转化为相对new_shape尺度的框中心坐标，input_shape和new_shape的左上角重合，且为坐标原点
            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.0)
        box_maxs = box_yx + (box_hw / 2.0)
        boxes = np.concatenate([box_mins[..., 0], box_mins[..., 1], box_maxs[..., 0], box_maxs[..., 1]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)

        return boxes

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
        # -----------------------------------------------#
        # 将预测结果转化为左上角右下角形式，并将预测张量的shape转化为[batch_size, num_anchors, 85]，
        # 其中num_anchors代表总的先验框数量，每个像素点3个，对13x13特征图来说，共计507个
        # -----------------------------------------------#
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        # 构建循环，针对每一张图片的预测结果进行处理
        for i, image_pred in enumerate(prediction):
            # --------------------------------------------#
            # 将种类提取出来，80个种类选择种类置信度max和相应的种类（索引）
            # class_conf [num_anchors, 1] 种类中最大的置信度值
            # class_pred [num_anchors, 1] 种类中最大的置信度的索引，即种类
            # --------------------------------------------#
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], dim=1, keepdim=True)

            # --------------------------------------------#
            # 利用置信度进行第一轮筛选
            # 本质上来说，对一张图片的预测结果包括num_anchors预测框，每个预测框均包含85个属性，xywhc+cls
            # 首先要找的每个预测框的最大种类置信度和其索引，即该框内的种类预测
            # 然后再找到所有预测框中满足预设置信度的索引，用布尔索引完成该方法实现
            # 判别式：位置置信度x种类置信度>=预设值
            # --------------------------------------------#
            conf_mask = (image_pred[:, 4] * class_conf >= conf_thres).squeeze()

            # 根据置信度进行预测结果的筛选和保存
            # 利用置信度布尔掩码进行筛选，保留满足要求的
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]

            # 判断该图片是否有预测结果，若无则循环重新开始
            if not image_pred.size(0):
                continue

            # --------------------------------------------#
            # 将经过第一轮筛选的结果合并到一个张量detections中
            # detections shape=[num_anchors, 7]
            # num_anchors为筛选后的数量
            # 7代表x1, y1, x2, y2, obj_conf, class_conf, class_pred
            # --------------------------------------------#
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

            # --------------------------------------------#
            # 将预测结果中的所有预测种类提取出来,重复的种类合并
            # --------------------------------------------#
            unique_labels = detections[:, -1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            # 遍历预测种类，将对应类的全部预测结果提取，然后处理
            # 举例来说，此时c代表狗，则利用布尔索引将所有预测框预测结果中含狗的提取出来，进行非极大值抑制
            # 可能来说，对一个像素点上，3个预测框均预测出狗，但是我们只需要一个最准确的，因此其余2个要被抑制掉
            for c in unique_labels:
                detections_class = detections[detections[:, -1] == c]
                # ------------------------------------------#
                #   使用官方自带的非极大抑制会速度更快一些！
                #
                #   nms 函数是非极大值抑制（NMS）的实现。NMS 用于消除多个检测器可能产生的重叠或冗余的边界框。
                #   它通常接受三个参数：
                #         边界框的坐标（通常是 [x1, y1, x2, y2] 的格式，其中 (x1, y1) 是左上角坐标，(x2, y2) 是右下角坐标）。
                #         边界框的置信度得分（通常是一个表示模型对边界框内存在目标的确信程度的值）。
                #         非极大值抑制的阈值（nms_thres），它决定了两个边界框的重叠程度需要达到多少才会被视为冗余。
                #   NMS 函数的返回值通常是一个整数数组（或类似的索引列表），这些整数对应于输入候选框数组中的索引，
                #   表示被保留的（非被抑制的）候选框。这些被保留的候选框通常是置信度较高且与其他候选框重叠较少的框。
                # ------------------------------------------#
                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4] * detections_class[:, 5],  # 位置置信度x种类置信度
                    nms_thres
                )
                # 根据nms结果，选着保留的预测框
                max_detections = detections_class[keep]
                # ----------------------------------------------------------#
                # 这部分是另外的方法，自己编写nms
                # # 按照存在物体的置信度排序
                # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
                # detections_class = detections_class[conf_sort_index]
                # # 进行非极大抑制
                # max_detections = []
                # while detections_class.size(0):
                #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
                #     max_detections.append(detections_class[0].unsqueeze(0))
                #     if len(detections_class) == 1:
                #         break
                #     ious = bbox_iou(max_detections[-1], detections_class[1:])
                #     detections_class = detections_class[1:][ious < nms_thres]
                # # 堆叠
                # max_detections = torch.cat(max_detections).data

                # ---------------------------------------------------------#
                # 将保留的预测框保存，因为是遍历所有种类，所以都要存储进去
                # 根据output[i]的状态判断存储方式
                # 如果是空的，就将max_detections直接添加进去
                # 如果是非空，则将新的类别max_detections添加进去
                # max_detections的shape=[num_anchors, 7]，其中num_anchors数量又减少了，因为经历的nms
                # ---------------------------------------------------------#
                if output[i] is None:
                    output[i] = max_detections
                else:
                    torch.cat((output[i], max_detections))
            # 因为遍历了batch_size的图片，所以针对每个图片的结果均要存储
            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                # 将预测框由左上右下转换成中心宽高形式
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, (output[i][:, 2:4] - output[i][:, 0:2])
                # 注意：这里的box_xy和box_wh是归一化的数值，在方法def decode_box的最后已经进行了归一化处理
                output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

        return output

# class DecodeBoxNP():
