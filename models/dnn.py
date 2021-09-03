from ray import tune
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
from models.layers import Embedding, MLP


DNN_CONFIG = {
    # training config
    'lr': 1e-3,
    # model config
    'nemb': tune.grid_search([0, 1, 4, 16, 32]),
    'mlp_layers': tune.grid_search([2, 4, 6, 8]),
    'mlp_hid': tune.grid_search([64, 128, 256, 512]),
    'dropout': tune.grid_search([0.0, 0.05, 0.1]),
}


class DNNModel(torch.nn.Module):
    """
    Model:  Deep Neural Networks
    """
    def __init__(self, nclass, nfield, nfeat, nemb, mlp_layers, mlp_hid, dropout):
        super().__init__()
        if nemb > 0:
            self.classifier = nn.Sequential(
                Embedding(nfeat, nemb),
                Rearrange('b f e -> b (f e)'),
                MLP(nfield*nemb, mlp_layers, mlp_hid, dropout, noutput=nclass)
            )
        else:
            self.classifier = nn.Sequential(
                MLP(nfeat, mlp_layers, mlp_hid, dropout, noutput=nclass)
            )

    def forward(self, x):
        """
        :param x:   FloatTensor B*F
        :return:    y of size B, Regression and Classification (+softmax)
        """
        return self.classifier(x)   # B*nclass