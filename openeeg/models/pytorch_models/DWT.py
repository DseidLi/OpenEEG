import torch
import torch.nn as nn


# 定义模型
class CNN1DNetV2(nn.Module):

    def __init__(self, num_classes=109, num_channels=32):
        super(CNN1DNetV2, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 128, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(128)  # 这个数字永远抄上一行第二个参数

        self.conv2 = nn.Conv1d(128, 256, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(256, 512, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(512)

        self.fc = nn.Linear(64001, num_classes)

    def forward(self, x, task_type):
        # x = x.unsqueeze(1)  # 增加通道维度
        # print(x.shape)
        # 64 1 num_channels 160 批量大小应该会始终在最外层
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        # print(x.shape)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)

        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat([x, task_type.unsqueeze(1)], dim=1)
        # print(x.shape)
        x = self.fc(x)
        # x = self.softmax(x)
        return x
