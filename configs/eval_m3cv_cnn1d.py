import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split

from openeeg.datasets.m3cv import M3CVDataset
from openeeg.models.pytorch_models import CNN1DNetV2_RevGrad
from openeeg.trainers import TrainerV2
from openeeg.utils.device_utils import get_device


# 分割数据集为训练集和测试集
def split_dataset(dataset, test_size=0.1, random_state=42):
    train_idx, test_idx = train_test_split(range(len(dataset)),
                                           test_size=test_size,
                                           random_state=random_state)
    return (torch.utils.data.Subset(dataset, train_idx),
            torch.utils.data.Subset(dataset, test_idx))


def load_saved_dataset(filename):
    # Load the data from the file
    all_data, all_tasks, all_labels = torch.load(filename)

    # Create a TensorDataset with the loaded data
    return TensorDataset(all_data, all_tasks, all_labels)


def main():

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='Train a model on M3CV dataset.')
    parser.add_argument('-r',
                        '--resume',
                        type=str,
                        help='Path to the saved model to resume training.')
    parser.add_argument('--memory',
                        action='store_true',
                        help='Load everything into memory before training')
    parser.add_argument('--workers',
                        type=int,
                        help='Number of workers for data loading')

    # 解析命令行参数
    args = parser.parse_args()
    # M3CV数据集使用示例
    root_dir = r'./data/M3CV'

    device, device_message = get_device()
    print(device_message)

    if args.memory:
        # 加载完整的数据集
        full_dataset = load_saved_dataset(
            'data/M3CV/pytorch_data_v2/source_dataset_taskfromcsv.pt')

        # 定义训练集和测试集的大小
        train_size = int(0.9 * len(full_dataset))
        test_size = len(full_dataset) - train_size

        # 随机切分数据集
        train_dataset, test_dataset = random_split(full_dataset,
                                                   [train_size, test_size])

        # 创建 DataLoader
        train_loader = DataLoader(train_dataset,
                                  batch_size=1024,
                                  shuffle=True,
                                  num_workers=args.workers)
        test_loader = DataLoader(test_dataset,
                                 batch_size=1024,
                                 shuffle=False,
                                 num_workers=args.workers)

    else:
        dataset = M3CVDataset(root_dir=root_dir, dataset_type='Enrollment')
        # 数据集分割
        train_dataset, test_dataset = split_dataset(dataset)

        # 数据加载器
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)

    # 实例化模型
    model = CNN1DNetV2_RevGrad(num_classes=95, num_channels=64).to(device)

    # 检查是否指定了 --resume 参数
    if args.resume:
        # 加载模型
        model.load_state_dict(torch.load(args.resume))
        print(f'Resumed training from {args.resume}')

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 训练过程
    trainer = TrainerV2(
        model,
        device,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        epochs=100,
        save_path=os.path.dirname(args.resume) if args.resume else None)
    trainer.train()


if __name__ == '__main__':
    main()
