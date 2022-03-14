import torch
from models.layers import MLP


class DNN(torch.nn.Module):
    """ Model:  Deep Neural Networks """
    def __init__(self, nclass: int, nfield: int, mlp_nlayer: int, mlp_nhid: int, dropout: float):
        super().__init__()
        self.classifier = MLP(nfield, mlp_nlayer, mlp_nhid, dropout, noutput=nclass)

    def forward(self, features):
        """
        :param x:   [bsz*nfield], FloatTensor
        :return:    [bsz*nclass], FloatTensor
        """
        x = features['quantitative']
        if x.dtype != torch.float: x = x.float()
        return self.classifier(x)
