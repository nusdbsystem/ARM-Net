import torch
import torch.nn as nn
from einops import rearrange


class Linear(nn.Module):

    def __init__(self, nfeat):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(nfeat))
        nn.init.normal_(self.weight)
        self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, x):
        """
        :param x:   [bsz, nfield], FloatTensor
        :return:    [bsz], FloatTensor
        """
        x = torch.einsum('bf, f -> b', x, self.weight)  # bsz
        return x + self.bias                            # bsz


class LR(torch.nn.Module):
    """ Model:  Logistic Regression """
    def __init__(self, nclass: int, nfield: int):
        super().__init__()
        self.linears = nn.ModuleList(Linear(nfield) for _ in range(nclass))

    def forward(self, features):
        """
        :param x:   [bsz*nfield], FloatTensor
        :return:    [bsz*nclass], FloatTensor
        """
        x = features['quantitative']
        if x.dtype != torch.float: x = x.float()
        y = [layer(x) for layer in self.linears]        # [nclass, ...]
        return rearrange(y, 'nclass b -> b nclass')     # bsz*nclass
