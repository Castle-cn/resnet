from modeling.configs import CONFIGS
from modeling.models import ResNet
import random
import numpy as np
import torch


# 设置下随机种子
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def setup(args):
    set_seed(args)

    if args.dataset == "mnist":
        num_classes = 10
        channels = 1
    else:
        raise Exception("There is no such dataset!")

    # Prepare model
    model_config = CONFIGS[args.model_type](channels, num_classes)
    model = ResNet(model_config.block, model_config.block_nums,
                   model_config.channels, model_config.classes_nums)

    return model
