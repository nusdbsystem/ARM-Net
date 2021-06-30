import torch
from models.layers import Embedding, MLP

class DNNModel(torch.nn.Module):
    """
    Model:  Deep Neural Networks
    """
    def __init__(self, nfield, nfeat, nemb, mlp_layers, mlp_hid, dropout):
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)
        self.mlp_ninput = nfield*nemb
        self.mlp = MLP(self.mlp_ninput, mlp_layers, mlp_hid, dropout)

    def forward(self, x):
        """
        :param x:   {'ids': LongTensor B*F, 'vals': FloatTensor B*F}
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        x_emb = self.embedding(x)                           # B*F*E
        y = self.mlp(x_emb.view(-1, self.mlp_ninput))       # B*1
        return y.squeeze(1)