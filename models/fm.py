import torch
from models.layers import Embedding, Linear, FactorizationMachine

class FMModel(torch.nn.Module):
    """
    Model:  Factorization Machine
    Ref:    S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, nfeat, nemb):
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)
        self.linear = Linear(nfeat)
        self.fm = FactorizationMachine(reduce_dim=True)

    def forward(self, x):
        """
        :param x:   {'id': LongTensor B*F, 'value': FloatTensor B*F}
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        y = self.linear(x) + self.fm(self.embedding(x))
        return y