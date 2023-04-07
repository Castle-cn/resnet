import torch
from torch import nn
from utils import dataset
from torchvision import transforms
from torch.utils.data import DataLoader

def get_loader(args):
    if args.dataset == 'mnist':
        train_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize([0.1307], [0.3081])])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.1307], [0.3081])])

        trainset = dataset.MnistDataset(data_root=args.data_root, transform=train_transform)
        testset = dataset.MnistDataset(data_root=args.data_root, is_train=False, transform=test_transform)
    else:
        raise Exception("no dataset called mnist")


    train_loader = DataLoader(trainset,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              drop_last=True,
                              # 这样将Tensor数据转移到GPU中速度就会快一些，默认为False。但是对于内存的大小要求比较高
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True)

    return train_loader, test_loader

def get_optim(args, model):
    if args.optim == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optim == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise Exception(f"there is no such optimizer named {args.optim}")


def get_loss_fn(args):
    if args.loss_fn == 'crossentropy':
        return nn.CrossEntropyLoss()
    elif args.loss_fn == 'mse':
        return nn.MSELoss()
    else:
        raise Exception(f"there is no such loss function named {args.loss_fn}")