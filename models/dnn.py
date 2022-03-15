import torch
from models.layers import MLP


class DNN(torch.nn.Module):
    """ Model:  Deep Neural Networks """
    def __init__(self, nclass: int, nfield: int, mlp_nlayer: int, mlp_nhid: int, dropout: float):
        super().__init__()
        self.classifier = MLP(nfield, mlp_nlayer, mlp_nhid, dropout, noutput=nclass)

    def forward(self, features):
        """
        :param quantitative:    [bsz*nfield], FloatTensor
        :return:                [bsz*nclass], FloatTensor
        """
        quantitative = features['quantitative']
        if quantitative.dtype != torch.float: quantitative = quantitative.float()
        return self.classifier(quantitative)
