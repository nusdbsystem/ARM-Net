import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, ninput: int, nlayers: int, nhid: int,
                 dropout: float = 0., noutput: int = 1):
        super().__init__()
        layers = list()
        for i in range(nlayers):
            layers.append(nn.Linear(ninput, nhid))
            layers.append(nn.BatchNorm1d(nhid))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            ninput = nhid
        if nlayers==0: nhid = ninput
        layers.append(nn.Linear(nhid, noutput))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x:   B*ninput
        :return:    B*nouput
        """
        return self.mlp(x)
