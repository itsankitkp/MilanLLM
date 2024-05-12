
from ..core.base import Module
from torch import nn
import torch

class LinearRegression(Module):
    """The linear regression model implemented with high-level APIs.

    Defined in :numref:`sec_linear_concise`"""
    def __init__(self, lr):
        super().__init__()
        self.lr=lr
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def forward(self, X):

        return self.net(X)

    def loss(self, y_hat, y):

        fn = nn.MSELoss()
        return fn(y_hat, y)

    def configure_optimizers(self):

        return torch.optim.SGD(self.parameters(), self.lr)

    def get_w_b(self):

        return (self.net.weight.data, self.net.bias.data)