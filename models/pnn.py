import torch
from models.layers import Embedding, MLP, get_triu_indices

class InnnerProduct(torch.nn.Module):

    def forward(self, x):
        """
        :param x:   FloatTensor B*F*E
        :return:    FloatTensor B*(Fx(F-1))
        """
        nfield = x.size(1)
        vi_indices, vj_indices = get_triu_indices(nfield)
        vi, vj = x[:, vi_indices], x[:, vj_indices]             # B*(Fx(F-1)/2)*E
        inner_product = torch.sum(vi * vj, dim=-1)              # B*(Fx(F-1)/2)
        return inner_product

class IPNNModel(torch.nn.Module):
    """
    Model:  Inner Product based Neural Network
    Ref:    Y Qu, et al. Product-based Neural Networks for User Response Prediction
                over Multi-field Categorical Data, 2016.
    """
    def __init__(self, nfield, nfeat, nemb, mlp_layers, mlp_hid, dropout):
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)
        self.ninput = nfield * nemb
        self.pnn = InnnerProduct()
        self.mlp = MLP(self.ninput + nfield*(nfield-1)//2,
                       mlp_layers, mlp_hid, dropout)

    def forward(self, x):
        """
        :param x:   {'id': LongTensor B*F, 'value': FloatTensor B*F}
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        x_emb = self.embedding(x)                                               # B*F*E
        x_prod = self.pnn(x_emb)
        y = self.mlp(torch.cat([x_emb.view(-1, self.ninput), x_prod], dim=1))   # B*1
        return y.squeeze(1)

class KernelProduct(torch.nn.Module):

    def __init__(self, nfield, nemb):
        super().__init__()
        self.kernel = torch.nn.Parameter(torch.ones(nemb, nfield*(nfield-1)//2, nemb))
        torch.nn.init.xavier_uniform_(self.kernel.data)

    def forward(self, x):
        """
        :param x:   FloatTensor B*F*E
        :return:    FloatTensor B*(Fx(F-1))
        """
        nfield = x.size(1)
        vi_indices, vj_indices = get_triu_indices(nfield)
        vi, vj = x[:, vi_indices], x[:, vj_indices]             # B*(Fx(F-1)/2)*E
        kernel_product = torch.einsum('bki,ikj,bkj->bk',
                                      vi, self.kernel, vj)      # B*(Fx(F-1)/2)
        return kernel_product

class KPNNModel(torch.nn.Module):
    """
    Model:  Kernel Product based Neural Network
    Ref:    Y Qu, et al. Product-based Neural Networks for User Response Prediction
                over Multi-field Categorical Data, 2016.
    """
    def __init__(self, nfield, nfeat, nemb, mlp_layers, mlp_hid, dropout):
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)
        self.ninput = nfield * nemb
        self.pnn = KernelProduct(nfield, nemb)
        self.mlp = MLP(self.ninput + nfield*(nfield-1)//2,
                       mlp_layers, mlp_hid, dropout)

    def forward(self, x):
        """
        :param x:   {'id': LongTensor B*F, 'value': FloatTensor B*F}
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        x_emb = self.embedding(x)                                               # B*(FxE)
        x_prod = self.pnn(x_emb)                                                # B*(Fx(F-1)/2)
        y = self.mlp(torch.cat([x_emb.view(-1, self.ninput), x_prod], dim=1))   # B*1
        return y.squeeze(1)