import datetime
import os
import os.path as osp

import torch
import torch.nn.functional as F
from tqdm import tqdm


class TrainerV3:

    def __init__(self,
                 model,
                 discriminator,
                 device,
                 train_loader,
                 test_loader,
                 optimizer,
                 criterion,
                 epochs,
                 save_path=None):
        self.model = model
        self.discriminator = discriminator
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs

        if save_path is None:
            current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.save_path = f'outputs/{current_time}'
        else:
            self.save_path = save_path

    def train(self):
        best_domain_loss = float('-inf')  # 初始化最大 domain loss
        best_label_loss = float('inf')  # 初始化最小 label loss

        # 打开文件以记录性能
        os.makedirs(self.save_path, exist_ok=True)
        with open(osp.join(self.save_path, 'Performances.csv'), 'w') as file:
            file.write('Epoch,Domain Loss,Label Loss,Source Dataset Accuracy,'
                       'Target Dataset Accuracy,Test Dataset Accuracy\n')
            for epoch in range(self.epochs):
                total_domain_loss = 0.0
                total_label_loss = 0.0
                total_batches = 0
                progress_bar = tqdm(total=len(self.train_loader[0]),
                                    desc=f'Epoch {epoch + 1}')
                for ((source_data, source_task, source_labels),
                     (target_data, target_task,
                      target_labels)) in zip(*self.train_loader):

                    source_data, source_task, source_labels = (source_data.to(
                        self.device), source_task.to(
                            self.device), source_labels.to(self.device))
                    target_data, target_task, target_labels = (target_data.to(
                        self.device), target_task.to(
                            self.device), target_labels.to(self.device))

                    # Feature extraction and post-processing
                    source_features = self.model.feature_extractor(source_data)
                    target_features = self.model.feature_extractor(target_data)
                    source_features = torch.flatten(source_features,
                                                    start_dim=1)
                    source_features = torch.cat(
                        [source_features,
                         source_task.unsqueeze(1)], dim=1)
                    target_features = torch.flatten(target_features,
                                                    start_dim=1)
                    target_features = torch.cat(
                        [target_features,
                         target_task.unsqueeze(1)], dim=1)

                    # Discriminator processing
                    source_domain_preds = self.discriminator(
                        source_features).squeeze()
                    target_domain_preds = self.discriminator(
                        target_features).squeeze()
                    domain_preds = torch.cat(
                        [source_domain_preds, target_domain_preds], dim=0)
                    domain_labels = torch.cat([
                        torch.ones(len(source_data)),
                        torch.zeros(len(target_data))
                    ],
                                              dim=0).to(self.device)

                    # Classifier processing for both source and target features
                    source_label_preds = self.model.classifier(source_features)
                    target_label_preds = self.model.classifier(target_features)

                    # Compute loss
                    domain_loss = F.binary_cross_entropy_with_logits(
                        domain_preds, domain_labels)

                    # Compute loss for both domains
                    source_label_loss = F.cross_entropy(
                        source_label_preds, source_labels)
                    target_label_loss = F.cross_entropy(
                        target_label_preds, target_labels)
                    label_loss = source_label_loss + target_label_loss

                    # sum for total_loss
                    total_loss = domain_loss + label_loss

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                    # 累积损失
                    total_domain_loss += domain_loss.item()
                    total_label_loss += label_loss.item()
                    total_batches += 1

                    progress_bar.update(1)
                    progress_bar.set_postfix(domain_loss=domain_loss.item(),
                                             label_loss=label_loss.item())

                progress_bar.close()
                # 检查并保存具有最大 domain loss 的模型
                if domain_loss.item() > best_domain_loss:
                    best_domain_loss = domain_loss.item()
                    self.save_model('best_domain_loss.pt')

                # 检查并保存具有最小 label loss 的模型
                if label_loss.item() < best_label_loss:
                    best_label_loss = label_loss.item()
                    self.save_model('best_label_loss.pt')

                # 计算每个 epoch 的平均损失
                avg_domain_loss = total_domain_loss / total_batches
                avg_label_loss = total_label_loss / total_batches

                # 保存模型状态
                self.save_model('last.pt')

                # 评估模型性能
                source_dataset_accuracy = self.evaluate(self.train_loader[0])
                target_dataset_accuracy = self.evaluate(self.train_loader[1])
                test_dataset_accuracy = self.evaluate(self.test_loader)

                # 记录每个 epoch 的性能
                logs = (f'{epoch + 1},{avg_domain_loss:.4f},'
                        f'{avg_label_loss:.4f},'
                        f'{source_dataset_accuracy:.2f}%,'
                        f'{target_dataset_accuracy:.2f}%,'
                        f'{test_dataset_accuracy:.2f}%\n')
                file.write(logs)
                file.flush()  # 立即将缓冲区内容写入文件

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, task_type, target in loader:
                data, task_type, target = (data.to(self.device),
                                           task_type.to(self.device),
                                           target.to(self.device))
                outputs = self.model(data, task_type)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100 * correct / total

    def save_model(self, filename):
        path = osp.join(self.save_path, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
