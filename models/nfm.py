import torch
from models.layers import Embedding, Linear, FactorizationMachine, MLP

class NFMModel(torch.nn.Module):
    """
    Model:  Neural Factorization Machine
    Ref:    X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.
    """

    def __init__(self, nfeat, nemb, mlp_layers, mlp_hid, dropout):
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)
        self.linear = Linear(nfeat)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(reduce_dim=False),
            torch.nn.BatchNorm1d(nemb),
            torch.nn.Dropout(dropout)
        )
        self.mlp = MLP(nemb, mlp_layers, mlp_hid, dropout, noutput=1)

    def forward(self, x):
        """
        :param x:   {'id': LongTensor B*F, 'value': FloatTensor B*F}
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        bi_interaction = self.fm(self.embedding(x))                 # B*E
        y = self.linear(x) + self.mlp(bi_interaction).squeeze(1)    # B
        return y