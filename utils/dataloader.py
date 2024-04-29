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

    # def merge_bboxes(self, bboxes, cutx, cuty):
    #
    #     return merge_bbox
    # 定义mosaic数据增强函数
    # def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
    #
    #     return new_image, new_boxes
    # ------------------------------------#
    # 定义mixup增强函数
    # ------------------------------------#
    # def get_random_data_with_Mixup(self, image_1, box_1, image_2, box_2):
    #
    #     return new_image, new_boxes


def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes