import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device(
            'cuda'), f'有可用的 GPU：{torch.cuda.get_device_name(0)}'
    else:
        return torch.device('cpu'), '没有可用的 GPU，使用 CPU。'
