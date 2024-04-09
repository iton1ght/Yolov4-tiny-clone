import math
import numpy as np
import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]], label_smoothing = 0):

    def clip_by_tensor(self, t, t_min, t_max):

    def MSELoss(self, pred, target):

    def BSELoss(self, pred, target, ):

    def box_ciou(self, box_1, box_2):

    def smooth_labels(self, y_ture, label_smoothing, num_classes):

    def forward(self, l, input, targets = None):

    def calculate_ciou(self, _box_a, _box_b):

    def get_target(self, l, target, anchors, in_h, in_w):

    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):

