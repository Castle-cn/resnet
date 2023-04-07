import os
import struct
from torch.utils.data import Dataset
import numpy as np


def read_image(image_path):
    with open(image_path, 'rb') as imgpath:
        _, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(num, rows * cols)
    return rows, cols, images


def read_label(label_path):
    with open(label_path, 'rb') as lpath:
        _ = struct.unpack('>II', lpath.read(8))
        labels = np.fromfile(lpath, dtype=np.uint8)
    return labels


class MnistDataset(Dataset):
    # 类的初始化,没什么好说的,固定格式,微调即可
    def __init__(self, data_root, is_train=True, transform=None, target_transform=None):
        if is_train:
            img_dir = os.path.join(data_root, 'train-images.idx3-ubyte')
            label_dir = os.path.join(data_root, 'train-labels.idx1-ubyte')
        else:
            img_dir = os.path.join(data_root, 't10k-images.idx3-ubyte')
            label_dir = os.path.join(data_root, 't10k-labels.idx1-ubyte')

        self.labels = read_label(label_dir)
        self.rows, self.cols, self.images = read_image(img_dir)
        self.transform = transform
        self.target_transform = target_transform

    # 获取数据集大小
    def __len__(self):
        return len(self.labels)

    # 获取指定index的数据
    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx].reshape([self.rows, self.cols])
        # transform固定写法, 基本不变
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
