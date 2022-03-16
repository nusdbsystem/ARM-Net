import torch
from models.layers import Embedding, Linear, FactorizationMachine

class AnovaKernel(torch.nn.Module):

    def __init__(self, order):
        super().__init__()
        self.order = order

    def forward(self, x):
        """
        :param x: FloatTensor B*F*E
        """
        bsz, nfiled, nemb = x.shape
        a_prev = torch.ones((bsz, nfiled+1, nemb), dtype=torch.float).to(x.device)
        for order in range(self.order):
            a = torch.zeros((bsz, nfiled+1, nemb), dtype=torch.float).to(x.device)
            a[:, order+1:, :] += x[:, order:, :] * a_prev[:, order:-1, :]
            a = torch.cumsum(a, dim=1)
            a_prev = a

        return torch.sum(a[:, -1, :], dim=-1)                   # B

class HOFMModel(torch.nn.Module):
    """
    Model:  Higher-Order Factorization Machines
    Ref:    M Blondel, et al. Higher-Order Factorization Machines, 2016.
    """

    def __init__(self, nfeat, nemb, order):
        super().__init__()
        assert order >= 2, 'invalid order'

        self.order = int(order)
        self.nemb = nemb
        self.embedding = Embedding(nfeat, nemb*(order-1))
        self.linear = Linear(nfeat)
        self.fm = FactorizationMachine(reduce_dim=True)
        if order >= 3:
            self.kernels = torch.nn.ModuleList([
                AnovaKernel(order=i) for i in range(3, order+1)
            ])

    def forward(self, x):
        """
        :param x:   {'id': LongTensor B*F, 'value': FloatTensor B*F}
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        x_emb = self.embedding(x)                               # B*F*Ex(order-1)
        y = self.linear(x) + self.fm(x_emb[:, :, :self.nemb])   # B
        for i in range(self.order-2):
            emb = x_emb[:, :, (i+1)*self.nemb: (i+2)*self.nemb] # B*F*E
            y += self.kernels[i](emb)                           # B
        return y