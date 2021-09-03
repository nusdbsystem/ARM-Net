from ray import tune
import torch
from einops import rearrange
from models.layers import Embedding, Linear, FactorizationMachine, MLP


FM_CONFIG = {
    # training config
    'lr': tune.grid_search([1e-3, 3e-3, 1e-2, 3e-2, 1e-1]),
    # model config
    'nemb': tune.grid_search([1, 4, 8, 16, 32, 64, 128, 256]),
}


class FMModel(torch.nn.Module):
    """
    Model:  Factorization Machine
    Ref:    S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, nclass, nfeat, nemb):
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)
        self.linear = Linear(nfeat)
        self.fm = FactorizationMachine(reduce_dim=False)
        self.classifier = MLP(nemb, nlayers=0, nhid=None, dropout=0, noutput=nclass)

    def forward(self, x):
        """
        :param x:   FloatTensor B*F
        :return:    y of size B, Regression and Classification (+softmax)
        """
        y = rearrange(self.linear(x), 'b -> b 1') \
            + self.fm(self.embedding(x))                # B*E
        y = self.classifier(y)                          # B*nclass
        return y