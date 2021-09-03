from ray import tune
import torch
import torch.nn as nn
from einops import rearrange
from models.layers import Linear


LR_CONFIG = {
    # training config
    'lr': tune.grid_search([1e-3, 3e-3, 1e-2, 3e-2, 1e-1]),
}


class LRModel(torch.nn.Module):
    """
    Model:  Logistic Regression
    """

    def __init__(self, nclass, nfeat):
        super().__init__()
        self.linear = nn.ModuleList(Linear(nfeat) for _ in range(nclass))

    def forward(self, x):
        """
        :param x:   FloatTensor B*F
        :return:    y of size B, Regression and Classification (+softmax)
        """
        y = [layer(x) for layer in self.linear]
        return rearrange(y, 'nclass b -> b nclass')