from datetime import datetime

import pandas as pd
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from openeeg.models.pytorch_models import CNN1DNetV2_RevGrad
from openeeg.utils.device_utils import get_device


def load_info(filename):
    df = pd.read_csv(filename)
    info_list = df.to_dict('records')  # 将每行转换为一个字典
    return info_list


def predict(model, device, test_loader, info_list, threshold=1):
    predictions = []
    with torch.no_grad():
        sample_index = 0  # 初始化样本索引
        for data, task, _ in tqdm(test_loader):
            data, task = data.to(device), task.to(device)
            outputs = model(data, task)
            # print(outputs)

            probs = softmax(outputs.data, dim=1)
            max_probs, predicted = torch.max(probs, 1)
            for j in range(data.size(0)):  # 迭代每个样本
                epoch_info = info_list[sample_index + j]
                usage = epoch_info['Usage']

                if usage == 3:
                    # 检查最大概率是否小于阈值
                    if max_probs[j] < threshold:
                        predictions.append(0)  # 标记为入侵者
                    else:
                        # 预测 SubjectID
                        predictions.append(predicted.cpu().numpy()[j] + 1)
                elif usage == 4:
                    # 判断真伪
                    actual_subj = epoch_info['SubjectID']
                    actual_subj = int(actual_subj.replace('sub', '')) - 1
                    # 检查模型的预测概率是否大于阈值，并且预测编号是否与声称的编号匹配
                    if max_probs[j] >= threshold and \
                            actual_subj == predicted.cpu().numpy()[j]:
                        predictions.append(1)  # 真实
                    else:
                        predictions.append(0)  # 有问题/入侵者
                else:
                    # 其他情况
                    predictions.append(-1)

            sample_index += data.size(0)  # 更新样本索引
    return predictions


def load_saved_dataset(filename):
    # Load the data from the file
    all_data, all_tasks, all_labels = torch.load(filename)

    # Create a TensorDataset with the loaded data
    return TensorDataset(all_data, all_tasks, all_labels)


# Main function
def main():

    # 在 main 函数中
    test_info_list = load_info('data/M3CV/Testing_Info.csv')
    test_dataset_in_memory = load_saved_dataset(
        'data/M3CV/pytorch_data_v2/testing_dataset_taskfromcsv.pt')

    test_loader = DataLoader(test_dataset_in_memory,
                             batch_size=1024,
                             num_workers=0)

    device, device_message = get_device()
    print(device_message)
    model = CNN1DNetV2_RevGrad(num_classes=95, num_channels=64)
    model.load_state_dict(
        torch.load('outputs/20240117_002034/best_label_loss.pt'))
    model = model.to(device)

    # Perform predictions
    predictions = predict(model, device, test_loader, test_info_list)

    # 生成预测文件
    epoch_ids = [info['Row']
                 for info in test_info_list]  # 从 test_info_list 获取 EpochID
    prediction_df = pd.DataFrame({
        'EpochID': epoch_ids,
        'Prediction': predictions
    })

    # 获取当前的日期和时间
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存文件，文件名包含时间戳
    prediction_df.to_csv(f'results/predictions_{current_time}.csv',
                         index=False)


if __name__ == '__main__':
    main()
