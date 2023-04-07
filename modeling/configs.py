from ml_collections import config_dict
from modeling import models

# 获取resnet18模型的参数
def get_resnet18_config(channels, classes_nums):
    config = config_dict.ConfigDict()
    config.block = models.BuildingBlock
    config.block_nums = [2, 2, 2, 2]
    config.classes_nums = classes_nums  # 需要修改
    config.channels = channels
    return config


def get_resnet34_config(channels,classes_nums):
    config = config_dict.ConfigDict()
    config.block = models.BuildingBlock
    config.block_nums = [3, 4, 6, 3]
    config.classes_nums = classes_nums  # 需要修改
    config.channels = channels
    return config


def get_resnet50_config(channels,classes_nums):
    config = config_dict.ConfigDict()
    config.block = models.BottleNeck
    config.block_nums = [3, 4, 6, 3]
    config.classes_nums = classes_nums  # 需要修改
    config.channels = channels
    return config


def get_resnet101_config(channels,classes_nums):
    config = config_dict.ConfigDict()
    config.block = models.BottleNeck
    config.block_nums = [3, 4, 23, 3]
    config.classes_nums = classes_nums  # 需要修改
    config.channels = channels

    return config


def get_resnet152_config(channels,classes_nums):
    config = config_dict.ConfigDict()
    config.block = models.BottleNeck
    config.block_nums = [3, 8, 36, 3]
    config.classes_nums = classes_nums  # 需要修改
    config.channels = channels
    return config

CONFIGS = {
    'resnet18':get_resnet18_config,
    'resnet34':get_resnet34_config,
    'resnet50':get_resnet50_config,
    'resnet101':get_resnet101_config,
    'resnet152':get_resnet152_config
}
