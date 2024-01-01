import torch
import torch.nn as nn


# 定义模型
class CNN2DNet(nn.Module):

    def __init__(self, num_classes=109, num_channels=32):
        super(CNN2DNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, (num_channels, 1))
        self.bn1 = nn.BatchNorm2d(128)  # 这个数字永远抄上一行第二个参数

        self.conv2 = nn.Conv2d(1, 256, (128, 1))
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(1, 512, (256, 1))
        self.bn3 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(81920, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加通道维度
        # print(x.shape)
        # 64 1 num_channels 160 批量大小应该会始终在最外层
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = x.permute(0, 2, 1, 3)  # 重排维度
        # print(x.shape)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = x.permute(0, 2, 1, 3)

        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)
        x = x.permute(0, 2, 1, 3)

        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)
        x = self.fc(x)
        # x = self.softmax(x)
        return x
