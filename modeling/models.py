from torch import nn


class BottleNeck(nn.Module):
    expension = 4  # 主要用在最后全连接神经网络计算输入维度

    def __init__(self, in_channels, out_channels, stride, downsample):
        super().__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=stride, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expension,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels * self.expension)
        )
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        print(x.shape)
        identity = x if self.downsample is None else self.downsample(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x) + identity
        x = self.relu(x)
        return x


class BuildingBlock(nn.Module):
    expension = 1  # 主要用在最后全连接神经网络计算输入维度

    def __init__(self, in_channels, out_channels, stride, downsample):
        super().__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        x = self.conv_1(x)
        x = self.conv_2(x) + identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, block_nums, channels, classes_nums):
        super().__init__()

        self.layer_1 = self.make_layers(block, block_nums[0], 64, 64, 1)
        self.layer_2 = self.make_layers(block, block_nums[1], 64 * block.expension, 128, 2)
        self.layer_3 = self.make_layers(block, block_nums[2], 128 * block.expension, 256, 2)
        self.layer_4 = self.make_layers(block, block_nums[3], 256 * block.expension, 512, 2)

        self.stack = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *self.layer_1,
            *self.layer_2,
            *self.layer_3,
            *self.layer_4,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(block.expension * 512, classes_nums),
            nn.Softmax(dim=1)
        )

    def make_layers(self, block, block_nums, in_channels, out_channels, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expension:
            downsample = nn.Conv2d(in_channels, out_channels * block.expension, kernel_size=1,
                                   stride=stride, padding=0)

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(block_nums - 1):
            layers.append(block(out_channels * block.expension, out_channels, 1, None))
        return layers

    def forward(self, x):
        # print(x.shape)
        x = self.stack(x)
        return x
