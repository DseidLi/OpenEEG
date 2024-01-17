import os

import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import Dataset


class M3CVDataset(Dataset):

    def __init__(self,
                 root_dir=r'D:\OpenEEG\data\M3CV',
                 dataset_type='Enrollment'):
        """
        root_dir: 数据集根目录
        dataset_type: 'Enrollment' 或 'Calibration' 或 'Testing'
        根据数据集类型决定前缀
        """
        info_file = os.path.join(root_dir, f'{dataset_type}_Info.csv')
        df = pd.read_csv(info_file)

        # 检查并适应不同的数据集格式
        if 'SubjectID' in df.columns:
            subject_col = 'SubjectID'
        else:
            subject_col = 'subject'

        # 检查并适应不同的数据集格式
        if 'EpochID' in df.columns:
            epoch_col = 'EpochID'
        else:
            epoch_col = 'Row'

        # 检查并适应不同的数据集格式
        if 'Task' in df.columns:
            task_col = 'Task'
        else:
            task_col = 'condition'

        self.img_list = df[epoch_col].values
        self.task_list = df[task_col].values
        self.labels = [self.extract_label(s) for s in df[subject_col].values]
        self.root_dir = root_dir
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        file_name = self.img_list[idx]
        mat_path = os.path.join(self.root_dir, self.dataset_type,
                                file_name + '.mat')
        data = sio.loadmat(mat_path)['epoch_data']
        eeg_data = data[:64, :]
        # task_type = np.unique(data[64, :])[0]
        task_type = self.task_list[idx]
        label = self.labels[idx]

        return eeg_data, task_type, torch.tensor(label, dtype=torch.long)

    @staticmethod
    def extract_label(s):
        try:
            # 尝试提取标签，只有当 s 是字符串类型时
            return int(s.replace('sub', '')) - 1 \
                if isinstance(s, str) else -1
        except ValueError:
            # 处理任何转换中的异常
            return -1
