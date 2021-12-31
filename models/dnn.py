from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
from models.layers import MLP


class DNNModel(torch.nn.Module):
    """ Model:  Deep Neural Networks """
    def __init__(self, nclass: int, nfeat: int, mlp_nlayer: int, mlp_nhid: int, dropout: float):
        super().__init__()
        self.classifier = MLP(nfeat, mlp_nlayer, mlp_nhid, dropout, noutput=nclass)

    def forward(self, x):
        """
        :param x:   [bsz*nfield], FloatTensor
        :return:    [bsz*nclass], FloatTensor
        """
        if x.dtype != torch.float: x = x.float()
        return self.classifier(x)
