import random
import numpy as np
import torch
from PIL import Image


# ---------------------------------------#
# 将图像转换成RGB图像，防止灰度图在预测时报错
# 代码仅支持RGB图像的预测，所有其他类型的图像都会转化为RGB
# ---------------------------------------#

def cvtColor(image):
    if isinstance(image, Image.Image):
        image_array = np.array(image)
        if len(np.shape(image_array)) == 3 and np.shape(image_array)[2] == 3:
            return image
        else:
            image = image.convert('RGB')
            return image


# -------------------------------------- #
# 读取分类文件，获得类名和数量
# -------------------------------------- #
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

# -------------------------------------- #
# 读取先验框文件，获得先验框数组和先验框数量
# -------------------------------------- #
def get_anchors(anchors_path):
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)

# ---------------------------------------#
# 获得优化器中学习率
# ---------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
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
# --------------------------------------#
# 设置Dataloader的种子
# 用于在多进程数据加载时初始化每个工作进程的随机数生成器
# 确保不同进程之间数据加载和预处理的一致性
# --------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

# 对图像像素值进行归一化操作
def preprocess_input(image):
    image /= 255.0
    return image
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
    print('-'*70)