import os

import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal
from torch.utils.tensorboard import SummaryWriter

import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from utils.utils import cvtColor, preprocess_input
from utils.utils_bbox import DecodeBox
#from utils.utils_map import get_coco_map, get_map


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_losses = []

        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    # 将当前世代epoch的训练损失和验证损失写入到列表中和文件中，并绘制折线图
    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_losses.append(val_loss)

        with open(os.path.join(self.log_dir, 'epoch_loss.txt'), 'a') as f:
            f.write(str(loss))
            f.write('\n')
        with open(os.path.join(self.log_dir, 'epoch_val_loss.txt'), 'a') as f:
            f.write(str(val_loss))
            f.write('\n')
        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    # loss_plot 的方法，用于绘制训练损失（train loss）和验证损失（val loss）的曲线图
    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_losses, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_losses, num, 3), '#8B4513', linestyle='--', linewidth=2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

# ---------------------------------------#
# 构建评估类
# ---------------------------------------#
class Evalcallback():
    def __init__(self, net, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines, log_dir, cuda,
                 map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, nms_iou=0.5, letterbox_image=True, MINOVERLAP=0.5, eval_flag=True, period=1):
        """

        :param net: 模型
        :param input_shape: 输入尺寸(416,416)
        :param anchors:输入先验框，二维数组 shape=[6,2]
        :param anchors_mask:输入先验框掩码，二维数组，shape=[2,3]
        :param class_names:输入类名，字符串列表
        :param num_classes:输入类数量，80个
        :param val_lines:输入验证集索引，字符串列表，每个元素代表原索引文件的一行
        :param log_dir:目标文件夹位置
        :param cuda:是否使用cuda
        :param map_out_path:map_out_path的位置
        :param max_boxes:
        :param confidence:
        :param nms_iou:
        :param letterbox_image:是否填充
        :param MINOVERLAP:
        :param eval_flag:是否开启评估标志位
        :param period:
        """
        super().__init__()
        self.net = net
        self.input_shape = input_shape
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.class_names = class_names
        self.num_classes = num_classes
        self.val_lines = val_lines
        self.log_dir = log_dir
        self.cuda = cuda
        self.map_out_path = map_out_path
        self.max_boxes = max_boxes
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.letterbox_image = letterbox_image
        self.MINOVERLAP = MINOVERLAP
        self.eval_flag = eval_flag
        self.period = period

        # 定义解码实例
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        self.maps = [0]
        self.epoches = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write('\n')

    def get_map_txt(self, image_id, image, class_names, map_out_path):

        return


    def on_epoch_end(self, epoch, model_eval):

