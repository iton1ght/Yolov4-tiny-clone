from random import sample, shuffle

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, epoch_length,
                 mosaic, mixup, mosaic_prob, mixup_prob, train, special_aug_ratio=0.7):
        """

        :param annotation_lines: 数据集列表
        :param input_shape: 输入图片的尺寸
        :param num_classes: 类别数量
        :param epoch_length: 训练总世代数
        :param mosaic: 是否进行mosaic增强，布尔值
        :param mixup: 是否进行mixup增强，布尔值
        :param mosaic_prob: 图片进行mosaic增强的比例
        :param mixup_prob: 图片进行mixup增强的比例（前提得进行mosaic增强）
        :param train: 训练模式
        :param special_aug_ratio: 进行数据增强的世代的比例
        """
        super().__init__()
        self.annotation_lines = annotation_lines
        self.input_shape      = input_shape
        self.num_classes      = num_classes
        self.epoch_length     = epoch_length
        self.mosaic           = mosaic
        self.mosaic_prob      = mosaic_prob
        self.mixup            = mixup
        self.mixup_prob       = mixup_prob
        self.train            = train
        self.special_aug_ration = special_aug_ratio

        self.epoch_now = -1
        self.length = len(self.annotation_lines)

    # 当你创建了一个这个类的实例，并尝试使用 len() 函数来获取其长度时，
    # __len__ 方法会被调用，并返回 self.length 的值
    def __len__(self):
        return self.length

    # __getitem__：这是一个特殊方法，它允许类的实例使用方括号索引操作，比如dataset[index]
    def __getitem__(self, index):
        # 取模运算来确保index在0到self.length - 1的范围内,
        # 当需要多次遍历整个数据集时, 通常用于实现数据集的循环加载
        index = index % self.length

        # ----------------------------------------#
        # 训练时进行数据的随机增强
        # 验证时不进行数据的随机增强
        # ----------------------------------------#
        # 判断是否进行mosaic增强
        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ration:
            # 从`self.annotation_lines`中随机选择3个样本
            lines = sample(self.annotation_lines, 3)
            # 将当前索引`index`对应的样本添加到`lines`中
            lines.append(self.annotation_lines[index])
            # 打乱`lines`的顺序
            shuffle(lines)
            # 使用函数进行Mosaic数据增强，得到增强后的图像`image`和对应的边界框`box`
            image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)

            # 如果进行mosaic增强，再判断是否进行mixup增强
            if self.mixup and self.rand() < self.mixup_prob:
                # 从'annotation_lines'中随机选择1个样本
                lines = sample(self.annotation_lines, 1)
                # 由于lines中只有1个元素，lines[0]则是这个样本本身，代入常规随机处理函数，返回新的图片和真实框
                image_2, box_2 = self.get_random_data(lines[0], self.input_shape, random=self.train)
                image, box = self.get_random_data_with_Mixup(image, box, image_2, box_2)
        # 如果不进行mosaic增强，则对数据进行常规随机处理
        else:
            image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)
        # 先将图像转换为浮点数数组，在进行归一化操作，最后利用转置将图像形状转化为（channels,height,width）
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box   = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return image, box

    # 用于生成一个在闭区间 [a, b] 内的随机浮点数
    def rand(self, a=0.0, b=1.0):
        return np.random.rand() * (b -a) +a

    # ------------------------------#
    # 常规随机处理函数，包括图像缩放、空白处填充
    # 函数里对训练集图片的处理和验证集图片的处理不同
    # 训练集图片缩放随机、宽高比也被扭曲，训练图片的真实框也同理；
    # 验证集缩放至与input_shape大小一致，宽高比也保持不变，验证图片的真实框也同理。
    def get_random_data(self, annotation_line, input_shape, jitter=0.3, hue=0.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()
        # ---------------------------#
        # 读取图像，并转换成RGB图像
        # ---------------------------#
        image = Image.open(line[0])
        image = cvtColor(image)
        # ---------------------------#
        # 获得图像的高宽与目标高宽
        # ---------------------------#
        iw, ih = image.size
        h, w   = input_shape
        # ---------------------------#
        # 获得预测框，将所有预测框转化成二维数组
        # ---------------------------#
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        # random = self.train，
        # 即验证集不参与训练，图片则直接进行以下处理（图片缩放和真实框缩放）
        if not random:
            # 对原始图像进行处理
            # 取原始图像较长的边作为缩放比例尺
            scale = min(w/iw, h/ih)
            # 对原始图像的宽高进行缩放
            # 较长边缩放与input_shape相同长度，较短边缩放比input_shape要短，存在空余部分
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w-nw) // 2
            dy = (h-nh) // 2

            # 将图像空余的部分加上灰条
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # 对原始图像的真实框进行处理,真实框采用左上右下坐标形式
            if len(box) > 0:
                np.random.shuffle(box)
                # 对真实框进行缩放，并且加上位置偏置
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                # 通过布尔数组的方式，对真实框左上右下坐标进行处理，使其在图像范围内
                # 对真实框左上坐标进行不小于0处理
                # 对真实框右下坐标进行不大于缩放后宽高处理
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                # 宽度列和高度列生产布尔数组
                # 对两个布尔数组进行逻辑与计算，生成新的布尔数据
                # 新的布尔数组用于选取宽高大于1的真实框作为有效真实框
                box = box[np.logical_and(box_w > 1, box_h > 1)]

            return image_data, box

        # ---------------------------------------------#
        # random = self.train，
        # 训练集图片进行处理，对图像进行缩放并且进行长和宽的扭曲
        # ---------------------------------------------#
        # 定义新的宽高比，如果new_ar>1，则宽长，反之则高长
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale  = self.rand(0.25, 2.0)
        if new_ar < 1:
            nh = int(ih * scale)
            nw = int(nh * new_ar)
        else:
            nw = int(iw * scale)
            nh = int(nw / new_ar)
        # 将图像空余的部分加上灰条，前提是有空余部分，因为缩放比例随机、宽高比也被调整
        image = image.resize((nw, nh), Image.BICUBIC)
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # ---------------------------------------------#
        # 翻转图像,翻转概率为50%
        # ---------------------------------------------#
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image_data = np.array(image, np.uint8)
        # ------------------------------------#
        # 对图像进行色域变换
        # 计算色域变换的参数
        # ------------------------------------#
        # 生成色调、饱和度、亮度的扰动值
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # 将图像转到HSV上
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # --------------------#
        # 应用变换
        # --------------------#
        # 用arange函数来创建一个一维数组 x。这个数组从 0 开始，到 256（不包括 256）结束，步长为 1
        # 数据类型是 np.uint8（无符号8位整数），那么 x 也将是一个数据类型为 np.uint8 的一维数组，包含了从 0 到 255 的整数
        x = np.arange(0, 256, dtype=dtype)
        # 计算查找表LUT
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        # 应用查找表
        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        # -------------------------------------#
        # 对真实框进行调整
        # -------------------------------------#
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box

    # ------------------------------------#
    # 图片增强处理过程中对真实框处理方法
    # 因为mosaic是将四张图片合并成一张图片，所以要将四张图片上的真实框合并到一个列表中
    # 合并过程中需要对原来的真实框进行截取
    # ------------------------------------#
    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if x1 > cutx or y1 > cuty:
                        continue
                    if y1 <= cuty and y2 >= cuty:
                        y2 = cuty
                    if x1 <= cutx and x2 >= cutx:
                        x2 = cutx

                if i == 1:
                    if x1 > cutx or y2 < cuty:
                        continue
                    if y1 <= cuty and y2 >= cuty:
                        y1 = cuty
                    if x1 <= cutx and x2 >= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox
    # ------------------------------------#
    # 定义mosaic数据增强函数
    # ------------------------------------#
    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        # min_offset_x和min_offset_y这两个值用于确定四张图片的中心位置在原input_shape的相对位置
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = []
        box_datas   = []
        # annotation_line中包括四张图片，从index=0开始索引
        index       = 0

        for line in annotation_line:
            # 对每一行进行分割，根据数据格式，按照'空格'进行分割
            line_content = line.split(' ')

            # 分割后形成字符串列表，对于列表第一个元素即为图片所在文件位置索引
            # 打开图片，并进行RGB三通道转换
            image = Image.open(line_content[0])
            image = cvtColor(image)

            # 获取图片的尺寸大小
            iw, ih = image.size

            # 获取真实框的数据
            box = np.array([np.array(list(map(lambda x: int(x), box.split(',')))) for box in line_content[1:]])

            # 判断是否翻转图片
            flip = self.rand() < 0.5
            if flip and len(box) > 0:
                # 将图片进行左右翻转（image.transpose用于旋转或翻转）
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                # 由于图片进行翻转，真实框的坐标发生变化（坐标系未变，仍是左上角为零点），所以得重新计算更新后的坐标
                # 由于是左右翻转，只有x方向的坐标发生变化
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            # ---------------------------------------#
            # 对图像进行缩放，并进行长和宽的扭曲
            # ---------------------------------------#
            new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(0.4, 1)
            if new_ar < 1:
                nh = int(ih * scale)
                nw = int(nh * new_ar)
            else:
                nw = int(iw * scale)
                nh = int(nw * new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            # ----------------------------------------#
            # 由于待处理的有四张图片，根据图片索引进行不同的操作
            # 先计算每张图片在最终合成图片中的放置位置
            # 四张图片围绕中心进行拼接
            # ----------------------------------------#
            if index == 0:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            elif index == 1:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y)
            elif index == 2:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y)
            elif index == 3:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y) - nh

            # 生成一张灰色底图，并将图片根据索引粘贴到相应位置
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)
            # 索引序号加1
            index = index + 1

            # ---------------------------------------#
            # 对该图片的真实框进行处理
            # ---------------------------------------#
            box_data = []
            if len(box) > 0:
                np.random.shuffle(box)
                # 根据图片的缩放变换和放置位置，对真实框进行处理
                # 先对真实框进行缩放，在进行位置偏置
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                # 考虑到缩放后真实框可能已经越界，所以要多数据进行截取处理
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box
            image_datas.append(image_data)
            box_datas.append(box_data)

        # --------------------------------------#
        # 将处理好的图片合并到一起，形成一张新的图片
        # --------------------------------------#
        # 拼接图片的中心点位置
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        # 生成一张空白的三通道图片的数组，注意这里是数组
        # 按照数组索引位置，将四张图片的数组数据合并在一起
        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image = np.array(new_image, np.uint8)

        # ---------------------------------------#
        # 对图像进行色域变换
        # 计算色域变换的参数
        # ---------------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # 现将图片由RGB颜色空间转换为HSV颜色空间
        # 再拆分成独立的通道，即色调、饱和度、亮度
        hue, sat, val =cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype

        # ----------------------------------------#
        # 利用LUT查找表，将色调、饱和度、亮度三个通道的数值进行变换
        # ----------------------------------------#
        # 生成0-255一维等差数组，数据格式与r相同
        x = np.arange(0, 256, dtype=r.dtype)
        # 根据色域变换参数r，生成色调、饱和度、亮度的变化查找表
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        # 应用查找表，对各个通道进行色域变换，并将通道合并
        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        # -----------------------------------------#
        # 对真实框进行进一步处理
        # -----------------------------------------#
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)


        return new_image, new_boxes

    # ------------------------------------#
    # 定义mixup增强函数
    # ------------------------------------#
    def get_random_data_with_Mixup(self, image_1, box_1, image_2, box_2):
         # 对两张图片进行像素点融合
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
             new_boxes = box_2
        elif len(box_2) == 0:
             new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], 0)
        return new_image, new_boxes


def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes