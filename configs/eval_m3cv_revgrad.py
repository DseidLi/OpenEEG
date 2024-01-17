import argparse

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm

from openeeg.datasets.m3cv import M3CVDataset
from openeeg.models.pytorch_models import CNN1DNetV2_RevGrad
from openeeg.models.pytorch_models.RevGrad import Discriminator
from openeeg.trainers import TrainerV3
from openeeg.utils.device_utils import get_device


def load_dataset_to_memory(dataset):
    all_data = []
    all_tasks = []
    all_labels = []

    for data, task, label in tqdm(dataset):
        data = torch.from_numpy(data)
        all_data.append(data)
        all_tasks.append(task)
        all_labels.append(label)

    all_data = torch.stack(all_data)
    all_tasks = torch.stack(all_tasks)
    all_labels = torch.stack(all_labels)

    return TensorDataset(all_data, all_tasks, all_labels)


def load_saved_dataset(filename):
    # Load the data from the file
    all_data, all_tasks, all_labels = torch.load(filename)

    # Create a TensorDataset with the loaded data
    return TensorDataset(all_data, all_tasks, all_labels)


# 分割数据集为训练集和测试集
def split_dataset(dataset, test_size=0.1, random_state=42):
    train_idx, test_idx = train_test_split(range(len(dataset)),
                                           test_size=test_size,
                                           random_state=random_state)
    return (torch.utils.data.Subset(dataset, train_idx),
            torch.utils.data.Subset(dataset, test_idx))


def main():

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='Train a model on M3CV dataset.')
    parser.add_argument('-p',
                        '--path_tosave',
                        type=str,
                        default=None,
                        help='Path to the saved model.')
    parser.add_argument('-w',
                        '--workers',
                        type=int,
                        default=14,
                        help='Num of workers of CPU in training process.')
    parser.add_argument('--in_memory',
                        action='store_true',
                        help='Load the entire dataset into memory for '
                        'faster training')

    # 解析参数
    args = parser.parse_args()
    num_workers = args.workers
    # M3CV数据集使用示例
    root_dir = r'./data/M3CV'

    device, device_message = get_device()
    print(device_message)

    pretrained_model = CNN1DNetV2_RevGrad(num_classes=95, num_channels=64)
    pretrained_model.load_state_dict(
        torch.load('outputs/20240117_002034/last.pt'))

    # # 使用 DataParallel 包装模型
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     pretrained_model = nn.DataParallel(pretrained_model)

    source_dataset = M3CVDataset(root_dir=root_dir, dataset_type='Enrollment')
    target_dataset = M3CVDataset(root_dir=root_dir, dataset_type='Calibration')

    pretrained_model = pretrained_model.to(device)

    # 数据集分割
    target_dataset, test_target_dataset = split_dataset(target_dataset)

    # target_dataset 是较短的数据集
    # 计算需要重复的次数
    num_samples = len(source_dataset)  # 使用较长数据集的长度
    sampler = RandomSampler(target_dataset,
                            replacement=True,
                            num_samples=num_samples)

    # 根据参数选择数据加载方式
    if args.in_memory:

        source_dataset_in_memory = load_saved_dataset(
            'data/M3CV/pytorch_data_v2/source_dataset_taskfromcsv.pt')
        target_dataset_in_memory = load_saved_dataset(
            'data/M3CV/pytorch_data_v2/target_dataset_taskfromcsv.pt')
        test_target_dataset_in_memory = load_saved_dataset(
            'data/M3CV/pytorch_data_v2/test_target_dataset_taskfromcsv.pt')

        # 创建内存 DataLoader
        source_loader = DataLoader(source_dataset_in_memory,
                                   batch_size=1024,
                                   shuffle=True,
                                   num_workers=num_workers)
        target_loader = DataLoader(target_dataset_in_memory,
                                   batch_size=1024,
                                   num_workers=10,
                                   sampler=sampler)
        target_test_loader = DataLoader(test_target_dataset_in_memory,
                                        batch_size=1024,
                                        num_workers=num_workers)
    else:
        # 使用动态 DataLoader
        source_loader = DataLoader(source_dataset,
                                   batch_size=1024,
                                   num_workers=num_workers,
                                   shuffle=True)
        target_loader = DataLoader(target_dataset,
                                   batch_size=1024,
                                   num_workers=10,
                                   sampler=sampler)
        target_test_loader = DataLoader(test_target_dataset,
                                        batch_size=1024,
                                        num_workers=num_workers)

    # Example usage
    discriminator = Discriminator().to(device)
    optim = torch.optim.Adam(list(discriminator.parameters()) +
                             list(pretrained_model.parameters()),
                             lr=0.0001)
    trainer = TrainerV3(pretrained_model,
                        discriminator,
                        device=device,
                        train_loader=[source_loader, target_loader],
                        test_loader=target_test_loader,
                        optimizer=optim,
                        criterion=None,
                        epochs=100,
                        save_path=args.path_tosave)
    trainer.train()


if __name__ == '__main__':
    main()
