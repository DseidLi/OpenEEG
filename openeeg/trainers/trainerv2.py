import datetime
import os

import torch
from tqdm import tqdm


class TrainerV2:

    def __init__(self,
                 model,
                 device,
                 train_loader,
                 test_loader,
                 optimizer,
                 criterion,
                 epochs,
                 save_path=None):
        self.model = model
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
        self.model.train()
        best_train_loss = float('inf')
        best_test_accuracy = 0

        for epoch in range(self.epochs):
            progress_bar = tqdm(total=len(self.train_loader),
                                desc=f'Epoch {epoch + 1}')
            for batch_idx, (data, task_type, target) in \
                    enumerate(self.train_loader):
                data, task_type, target = (data.to(self.device),
                                           task_type.to(self.device),
                                           target.to(self.device))
                self.optimizer.zero_grad()
                output = self.model(data, task_type)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())

                if loss.item() < best_train_loss:
                    best_train_loss = loss.item()
                    save_path_best_train = os.path.join(
                        self.save_path, 'best_train.pt')
                    os.makedirs(os.path.dirname(save_path_best_train),
                                exist_ok=True)
                    torch.save(self.model.state_dict(), save_path_best_train)

            progress_bar.close()

            # Evaluate on the test set
            test_accuracy = self.evaluate()
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                save_path_best_test = os.path.join(self.save_path,
                                                   'best_test.pt')
                os.makedirs(os.path.dirname(save_path_best_test),
                            exist_ok=True)
                torch.save(self.model.state_dict(), save_path_best_test)

            print(f'Test Accuracy after Epoch {epoch + 1}: '
                  f'{test_accuracy:.2f}%')

            save_path_last = os.path.join(self.save_path, 'last.pt')
            os.makedirs(os.path.dirname(save_path_last), exist_ok=True)
            torch.save(self.model.state_dict(), save_path_last)

        print(f'\nBest Training Loss: {best_train_loss:.4f}')
        print(f'Best Test Accuracy: {best_test_accuracy:.2f}%')

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, task_type, target in self.test_loader:
                data, task_type, target = (data.to(self.device),
                                           task_type.to(self.device),
                                           target.to(self.device))
                outputs = self.model(data, task_type)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100 * correct / total
