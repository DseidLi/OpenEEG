import os

import mne
import numpy as np
import requests
import torch
from torch.utils.data import Dataset


class EEGMMIDBDataset(Dataset):

    def __init__(self,
                 subject_list,
                 record_list,
                 root_dir,
                 channels=None,
                 transform=None):
        # 初始化数据集
        self.data = []
        self.labels = []
        self.root_dir = root_dir
        self.channels = channels
        self.transform = transform

        # 遍历主题列表，加载或处理每个主题的数据
        for subj in subject_list:
            subj_data, subj_labels = self._load_or_process_subject(
                subj, record_list)
            self.data.append(subj_data)
            self.labels.append(subj_labels)

        # 将数据从列表转换为 NumPy 数组
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def _load_or_process_subject(self, subject, record_list):
        # 检查是否已处理并保存过数据
        processed_data_path = os.path.join(self.root_dir,
                                           f'S{subject}_processed_data.npy')
        processed_labels_path = os.path.join(
            self.root_dir, f'S{subject}_processed_labels.npy')

        # 如果已处理，直接加载；否则，处理数据并保存
        if os.path.exists(processed_data_path) and os.path.exists(
                processed_labels_path):
            data = np.load(processed_data_path)
            labels = np.load(processed_labels_path)
        else:
            data, labels = self._process_subject_data(subject, record_list)
            np.save(processed_data_path, data)
            np.save(processed_labels_path, labels)

        return data, labels

    def _process_subject_data(self, subject, record_list):
        subj_data = []
        subj_labels = []

        for rec in record_list:
            file_path = os.path.join(self.root_dir, f'S{subject}',
                                     f'S{subject}R{rec}.edf')

            try:
                data = mne.io.read_raw_edf(file_path, preload=True)
                raw_data = data.get_data()
            except Exception as e:
                print(f'Error loading {file_path}: {e}. '
                      'Attempting to re-download...')
                if self._download_file(subject, rec):
                    try:
                        data = mne.io.read_raw_edf(file_path, preload=True)
                        raw_data = data.get_data()
                    except Exception as e:
                        print(f'Failed to load {file_path} '
                              f'after re-downloading: {e}')
                        continue
                else:
                    continue

            # 如果指定了通道，则仅使用这些通道的数据
            if self.channels is not None:
                raw_data = raw_data[self.channels]

            # 切割数据并添加到列表中
            for k in range(raw_data.shape[1] // 160):
                segment = raw_data[:, k * 160:(k + 1) * 160]
                subj_data.append(segment.astype(np.float32))
                subj_labels.append(int(subject) - 1)

        return np.array(subj_data), np.array(subj_labels)

    def _download_file(self, subject, record):
        # 构建要下载的文件的 URL
        base_url = 'https://archive.physionet.org/pn4/eegmmidb'
        url = f'{base_url}/S{subject:03}/S{subject:03}R{record:02}.edf'

        # 目标路径
        target_path = os.path.join(self.root_dir, f'S{subject}',
                                   f'S{subject}R{record}.edf')
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # 开始下载
        print(f'Downloading {url}...')
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                f.write(response.content)
            print('Download completed.')
        else:
            print(f'Failed to download file from {url}')
        # 返回下载是否成功
        return response.status_code == 200

    def __len__(self):
        # 返回数据集的长度
        return len(self.data)

    def __getitem__(self, idx):
        # 获取单个样本
        x = self.data[idx]
        y = self.labels[idx]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        # 应用转换（如果有）
        if self.transform:
            x = self.transform(x)

        return x, y


# # EEGMMIDB数据集使用示例
# root_dir = 'D:/EEG-ESRT/datasets/EEGMMIDB/'
# subject_list = ['%03d' % i for i in range(1, 110)]
# record_list = ['%02d' % i for i in range(1, 15)]
# channels32 = [
#     0, 2, 3, 4, 6, 8, 10, 12, 14, 16, 17, 18, 20, 21, 22, 23, 26, 29, 31, 33,
#     35, 37, 40, 41, 46, 48, 50, 52, 54, 57, 60, 62
# ]
