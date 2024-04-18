import random
import numpy as np
import torch
from PIL import Image

# -------------------------------------- #
# 获得类
# -------------------------------------- #
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_name = f.readlines()
    class_name = [c.strip() for c in class_name]
    return class_name, len(class_name)

# -------------------------------------- #
# 获得先验框
# -------------------------------------- #
def get_anchors(anchors_path):
    with open(anchors_path,encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)


# ---------------------------------------#
# 打印出训练参数
# ---------------------------------------#
def show_config(**kwargs):
    print('Configurations:')
    print('-'*70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-'*70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print(('-'*70))