import torch
from torch.utils.data.dataset import Dataset
class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, epoch_length: \
                        mosaic, mixup, mosaic_prob, mixup_prob, traint, special_aug_ratio=0.7):