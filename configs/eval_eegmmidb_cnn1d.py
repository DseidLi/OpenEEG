import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from openeeg.datasets.eegmmidb import EEGMMIDBDataset
from openeeg.models.pytorch_models import CNN1DNet
from openeeg.trainers import TrainerV1
from openeeg.utils.device_utils import get_device

# EEGMMIDB数据集使用示例
root_dir = './data/EEGMMIDB/'
subject_list = ['%03d' % i for i in range(1, 110)]
record_list = ['%02d' % i for i in range(1, 15)]
channels32 = [
    0, 2, 3, 4, 6, 8, 10, 12, 14, 16, 17, 18, 20, 21, 22, 23, 26, 29, 31, 33,
    35, 37, 40, 41, 46, 48, 50, 52, 54, 57, 60, 62
]

dataset = EEGMMIDBDataset(subject_list, record_list, root_dir, channels32)
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


# 数据正规化
def normalize_data(x):
    max_val = x.max()
    return x / max_val


# 数据集分割
train_dataset, test_dataset = split_dataset(dataset)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# # 数据正规化 (示例：对训练集的一个batch进行操作)
# for x, y in train_loader:
#     print(x, y)
#     x_normalized = normalize_data(x)
#     # 使用 x_normalized 进行模型训练
#     break

# 实例化模型
model = CNN1DNet(num_classes=109, num_channels=32).to(device)

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

# # 评估模型 (示例)
# def evaluate(model, device, test_loader,
# model_path='models/EEGMMIDB/CNN/pytorch/best.pt'):
#     model.eval()
#     model.load_state_dict(torch.load(model_path))
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += criterion(output, target).item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
#     print(
#         f'Test set: Average loss: {test_loss:.4f}, '
#         f'Accuracy: {correct}/{len(test_loader.dataset)} '
#         f'({100. * correct / len(test_loader.dataset):.0f}%)'
#     )

# # 评估
# evaluate(model, device, test_loader, model_path='eeg_model_best.pt')
