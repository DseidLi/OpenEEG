import os

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from sklearn.utils import shuffle
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
        df = shuffle(df, random_state=0)

        self.img_list = df['EpochID'].values
        self.labels = [
            int(s.replace('sub', '')) - 1 for s in df['SubjectID'].values
        ]
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
        task_type = np.unique(data[64, :])[0]
        label = self.labels[idx]

        return eeg_data, task_type, torch.tensor(label, dtype=torch.long)
