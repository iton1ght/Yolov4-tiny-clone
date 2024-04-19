from random import sample, shuffle

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor, preprocess_input
class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, epoch_length: \
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
        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ration:
            # 从`self.annotation_lines`中随机选择3个样本
            lines = sample(self.annotation_lines, 3)
            # 将当前索引`index`对应的样本添加到`lines`中
            lines.append(self.annotation_lines[index])
            # 打乱`lines`的顺序
            shuffle(lines)
            # 使用函数进行Mosaic数据增强，得到增强后的图像`image`和对应的边界框`box`
            image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)


        return image, box

    def rand(self, a=0, b=1):
        return np.random.rand() * (b -a) +a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):

        return image_data, box

    def merge_bboxes(self, bboxes, cutx, cuty):

        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):

        return new_image, new_boxes

    def get_random_data_with_Mixup(self, image_1, box_1, image_2, box_2):

        return new_image, new_boxes


def yolo_dataset_collate(batch):

    return images, bboxes