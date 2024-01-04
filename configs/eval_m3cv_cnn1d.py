import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


from openeeg.datasets.m3cv import M3CVDataset
from openeeg.models.pytorch_models import CNN1DNet
from openeeg.trainers import TrainerV1
from openeeg.utils.device_utils import get_device

# M3CV数据集使用示例
root_dir = r'./data/M3CV'

dataset = M3CVDataset(root_dir=root_dir,
                      dataset_type='Enrollment')
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

device, device_message = get_device()
print(device_message)


# 分割数据集为训练集和测试集
def split_dataset(dataset, test_size=0.1, random_state=42):
    train_idx, test_idx = train_test_split(range(len(dataset)),
                                           test_size=test_size,
                                           random_state=random_state)
    return torch.utils.data.Subset(dataset,
                                   train_idx), torch.utils.data.Subset(
                                       dataset, test_idx)


# 数据集分割
train_dataset, test_dataset = split_dataset(dataset)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 实例化模型
model = CNN1DNet(num_classes=95, num_channels=64).to(device)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练过程
trainer = TrainerV1(
    model,
    device,
    train_loader,
    optimizer,
    criterion,
    epochs=10,
)
trainer.train()
