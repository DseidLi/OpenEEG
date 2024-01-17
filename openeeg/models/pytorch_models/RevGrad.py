import torch.nn as nn
from torch.autograd import Function


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

    def __init__(self, lambda_=10):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            GradientReversal(),
            nn.Linear(64001, 100),  # Adjust the input dimensions as needed
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.Linear(20, 1))

    def forward(self, x):
        return self.layer(x)
