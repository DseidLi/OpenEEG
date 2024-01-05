import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
# import argparse

from openeeg.datasets.m3cv import M3CVDataset
from openeeg.models.pytorch_models import CNN1DNetV2_RevGrad
from openeeg.trainers import TrainerV2
from openeeg.utils.device_utils import get_device
import os



class GradientReversalFunction(Function):
    """Gradient Reversal Layer from: Unsupervised Domain Adaptation by
    Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass, the upstream
    gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            GradientReversal(),
            nn.Linear(64001, 50),  # Adjust the input dimensions as needed
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        return self.layer(x)


def train(model, discriminator, source_loader, target_loader, optimizer,
          epochs, device):
    for epoch in range(epochs):
        total_loss, total_domain_loss, total_label_accuracy = 0, 0, 0

        for (source_data, source_labels), (target_data, _) in \
                zip(source_loader, target_loader):
            # Prepare data
            source_data, source_labels = (source_data.to(device), 
                                          source_labels.to(device))
            target_data = target_data.to(device)
            domain_labels = torch.cat([torch.ones(len(source_data)), 
                                       torch.zeros(len(target_data))], 
                                      dim=0).to(device)

            # Forward pass
            source_features = model.feature_extractor(source_data)
            target_features = model.feature_extractor(target_data)
            features = torch.cat([source_features, target_features], dim=0)
            domain_preds = discriminator(features).squeeze()
            label_preds = model.classfier(source_features)

            # Calculate losses
            domain_loss = F.binary_cross_entropy_with_logits(domain_preds, 
                                                             domain_labels)
            label_loss = F.cross_entropy(label_preds, source_labels)
            total_loss = domain_loss + label_loss

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Log metrics
            total_domain_loss += domain_loss.item()
            total_label_accuracy += (label_preds.max(1)[1] == source_labels).float().mean().item()

        # Log epoch results
        print(f'Epoch {epoch+1}: Domain Loss: {total_domain_loss/len(source_loader)}, Label Accuracy: {total_label_accuracy/len(source_loader)}')


def main():

    # # 创建命令行参数解析器
    # parser = argparse.ArgumentParser(
    #     description='Train a model on M3CV dataset.')
    # parser.add_argument('-r', '--resume',
    #                     type=str,
    #                     help='Path to the saved model to resume training.')

    # M3CV数据集使用示例
    root_dir = r'./data/M3CV'
    
    pretrained_model = CNN1DNetV2_RevGrad(num_classes=95, num_channels=64)
    pretrained_model.load_state_dict(torch.load('path_to_pretrained_model.pt'))

    source_dataset = M3CVDataset(root_dir=root_dir,
                                 dataset_type='Enrollment')
    target_dataset = M3CVDataset(root_dir=root_dir,
                                 dataset_type='Calibratioon')
    device, device_message = get_device()
    print(device_message)

    # 分割数据集为训练集和测试集
    def split_dataset(dataset, test_size=0.1, random_state=42):
        train_idx, test_idx = train_test_split(range(len(dataset)),
                                               test_size=test_size,
                                               random_state=random_state)
        return (torch.utils.data.Subset(dataset, train_idx),
                torch.utils.data.Subset(dataset, test_idx))

    # 数据集分割
    target_dataset, test_target_dataset = split_dataset(target_dataset)

    # 数据加载器
    source_loader = DataLoader(source_dataset, batch_size=64, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=64)

    # Example usage
    discriminator = Discriminator().to(device)
    optim = torch.optim.Adam(list(discriminator.parameters()) + list(pretrained_model.parameters()), lr=0.001)
    train(pretrained_model, discriminator, source_loader, target_loader, optim,
          epochs=10, device=device)




        

if __name__ == "__main__":
    main()



# """
# Implements RevGrad:
# Unsupervised Domain Adaptation by Backpropagation, Ganin & Lemptsky (2014)
# Domain-adversarial training of neural networks, Ganin et al. (2016)
# """
# import argparse

# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torchvision.datasets import MNIST
# from torchvision.transforms import Compose, ToTensor
# from tqdm import tqdm

# import config
# from data import MNISTM
# from models import Net
# from utils import GrayscaleToRgb, GradientReversal


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def main(args):
#     model = Net().to(device)
#     model.load_state_dict(torch.load(args.MODEL_FILE))
#     feature_extractor = model.feature_extractor
#     clf = model.classifier

#     discriminator = nn.Sequential(
#         GradientReversal(),
#         nn.Linear(320, 50),
#         nn.ReLU(),
#         nn.Linear(50, 20),
#         nn.ReLU(),
#         nn.Linear(20, 1)
#     ).to(device)

#     half_batch = args.batch_size // 2
#     source_dataset = MNIST(config.DATA_DIR/'mnist', train=True, download=True,
#                           transform=Compose([GrayscaleToRgb(), ToTensor()]))
#     source_loader = DataLoader(source_dataset, batch_size=half_batch,
#                                shuffle=True, num_workers=1, pin_memory=True)
    
#     target_dataset = MNISTM(train=False)
#     target_loader = DataLoader(target_dataset, batch_size=half_batch,
#                                shuffle=True, num_workers=1, pin_memory=True)

#     optim = torch.optim.Adam(list(discriminator.parameters()) + list(model.parameters()))

#     for epoch in range(1, args.epochs+1):
#         batches = zip(source_loader, target_loader)
#         n_batches = min(len(source_loader), len(target_loader))

#         total_domain_loss = total_label_accuracy = 0
#         for (source_x, source_labels), (target_x, _) in tqdm(batches, leave=False, total=n_batches):
#                 x = torch.cat([source_x, target_x])
#                 x = x.to(device)
#                 domain_y = torch.cat([torch.ones(source_x.shape[0]),
#                                       torch.zeros(target_x.shape[0])])
#                 domain_y = domain_y.to(device)
#                 label_y = source_labels.to(device)

#                 features = feature_extractor(x).view(x.shape[0], -1)
#                 domain_preds = discriminator(features).squeeze()
#                 label_preds = clf(features[:source_x.shape[0]])
                
#                 domain_loss = F.binary_cross_entropy_with_logits(domain_preds, domain_y)
#                 label_loss = F.cross_entropy(label_preds, label_y)
#                 loss = domain_loss + label_loss

#                 optim.zero_grad()
#                 loss.backward()
#                 optim.step()

#                 total_domain_loss += domain_loss.item()
#                 total_label_accuracy += (label_preds.max(1)[1] == label_y).float().mean().item()

#         mean_loss = total_domain_loss / n_batches
#         mean_accuracy = total_label_accuracy / n_batches
#         tqdm.write(f'EPOCH {epoch:03d}: domain_loss={mean_loss:.4f}, '
#                    f'source_accuracy={mean_accuracy:.4f}')

#         torch.save(model.state_dict(), 'trained_models/revgrad.pt')


# if __name__ == '__main__':
#     arg_parser = argparse.ArgumentParser(description='Domain adaptation using RevGrad')
#     arg_parser.add_argument('MODEL_FILE', help='A model in trained_models')
#     arg_parser.add_argument('--batch-size', type=int, default=64)
#     arg_parser.add_argument('--epochs', type=int, default=15)
#     args = arg_parser.parse_args()
#     main(args)
