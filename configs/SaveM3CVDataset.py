import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from openeeg.datasets.m3cv import M3CVDataset


def save_dataset(dataset, filename):
    all_data = []
    all_tasks = []
    all_labels = []

    for data, task, label in tqdm(dataset):
        data = torch.from_numpy(data)
        task = torch.tensor(task)
        label = torch.tensor(label)
        all_data.append(data)
        all_tasks.append(task)
        all_labels.append(label)

    all_data = torch.stack(all_data)
    all_tasks = torch.stack(all_tasks)
    all_labels = torch.stack(all_labels)
    # 保存数据集到文件
    torch.save((all_data, all_tasks, all_labels), filename)


# 分割数据集为训练集和测试集
def split_dataset(dataset, test_size=0.1, random_state=42):
    train_idx, test_idx = train_test_split(range(len(dataset)),
                                           test_size=test_size,
                                           random_state=random_state)
    return (torch.utils.data.Subset(dataset, train_idx),
            torch.utils.data.Subset(dataset, test_idx))


root_dir = r'./data/M3CV'
source_dataset = M3CVDataset(root_dir=root_dir, dataset_type='Enrollment')
target_dataset = M3CVDataset(root_dir=root_dir, dataset_type='Calibration')
testing_dataset = M3CVDataset(root_dir=root_dir, dataset_type='Testing')

# 数据集分割
target_dataset, test_target_dataset = split_dataset(target_dataset)

save_dataset(source_dataset, 'data/M3CV/source_dataset_taskfromcsv.pt')
save_dataset(target_dataset, 'data/M3CV/target_dataset_taskfromcsv.pt')
save_dataset(test_target_dataset,
             'data/M3CV/test_target_dataset_taskfromcsv.pt')
save_dataset(testing_dataset, 'data/M3CV/testing_dataset_taskfromcsv.pt')
