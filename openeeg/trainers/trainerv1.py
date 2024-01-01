import datetime
import os

import torch


class TrainerV1:

    def __init__(self,
                 model,
                 device,
                 train_loader,
                 optimizer,
                 criterion,
                 epochs,
                 save_path=None):
        self.model = model
        self.device = device
        self.train_loader = train_loader
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
        best_loss = float('inf')

        for epoch in range(self.epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    save_path_best = f'{self.save_path}/best.pt'
                    os.makedirs(os.path.dirname(save_path_best), exist_ok=True)
                    torch.save(self.model.state_dict(), save_path_best)

                if batch_idx % 10 == 0:
                    print(
                        f'Train Epoch: {epoch+1} [{batch_idx * len(data)}/'
                        f'{len(self.train_loader.dataset)} '
                        f'({100. * batch_idx / len(self.train_loader):.0f}%)]'
                        f'\tLoss: {loss.item():.6f}')

            save_path_last = f'{self.save_path}/last.pt'
            os.makedirs(os.path.dirname(save_path_last), exist_ok=True)
            torch.save(self.model.state_dict(), save_path_last)
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
