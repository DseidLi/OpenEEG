import datetime
import os

import torch
from tqdm import tqdm


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
            progress_bar = tqdm(total=len(self.train_loader),
                                desc=f'Epoch {epoch + 1}')
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                progress_bar.update(1)  # Update the progress bar here

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    save_path_best = os.path.join(self.save_path, 'best.pt')
                    os.makedirs(os.path.dirname(save_path_best), exist_ok=True)
                    torch.save(self.model.state_dict(), save_path_best)

                if batch_idx % 10 == 0:
                    progress_bar.set_postfix(loss=loss.item())

            progress_bar.close()  # Close bar at the end of the epoch
            save_path_last = os.path.join(self.save_path, 'last.pt')
            os.makedirs(os.path.dirname(save_path_last), exist_ok=True)
            torch.save(self.model.state_dict(), save_path_last)
