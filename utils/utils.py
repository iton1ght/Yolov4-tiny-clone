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
# 设置种子
# 这个函数的主要目的是通过设置多个库的随机种子和配置，确保在多次运行代码时，涉及随机性的部分能够产生相同的结果
# ---------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN 是 NVIDIA 提供的一个深度神经网络库，PyTorch 可以使用它进行
    # GPU 加速。这里，deterministic=True 确保 cuDNN 的操作是确定的，即
    # 每次使用相同的输入和权重时，都会得到相同的输出。这有助于确保实验的可复现性。
    # 然而，这可能会降低计算速度。benchmark=False 则禁用了 cuDNN 的自动基准测
    # 试功能，这有助于确保每次运行代码时都使用相同的算法，而不是根据运行时性能自动
    # 选择算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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