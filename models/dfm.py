import torch
from models.layers import Embedding, Linear, FactorizationMachine, MLP

class DeepFMModel(torch.nn.Module):
    """
    Model:  DeepFM
    Ref:    H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """
    def __init__(self, nfield, nfeat, nemb, mlp_layers, mlp_hid, dropout):
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)
        self.linear = Linear(nfeat)
        self.fm = FactorizationMachine(reduce_dim=True)
        self.mlp_ninput = nfield*nemb
        self.mlp = MLP(self.mlp_ninput, mlp_layers, mlp_hid, dropout)

    def forward(self, x):
        """
        :param x:   {'ids': LongTensor B*F, 'vals': FloatTensor B*F}
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        x_emb = self.embedding(x)                                   # B*F*E
        y = self.linear(x)+self.fm(x_emb)+\
            self.mlp(x_emb.view(-1, self.mlp_ninput)).squeeze(1)    # B
        return y