import torch
from models.layers import Embedding, MLP

class CrossNetwork(torch.nn.Module):

    def __init__(self, ninput, nlayers):
        super().__init__()
        self.nlayers = nlayers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(ninput, 1, bias=False) for _ in range(nlayers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((ninput, ))) for _ in range(nlayers)
        ])

    def forward(self, x):
        """
        :param x:   FloatTensor B*(FxE)
        :return:    FloatTensor B*(FxE)
        """
        x0 = x
        for l in range(self.nlayers):
            xw = self.w[l](x)
            x = x0*xw + self.b[l] + x
        return x

class CrossNetModel(torch.nn.Module):
    """
    Model:  Deep & Cross (w/o a neural network)
    Ref:    R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.
    """
    def __init__(self, nfield, nfeat, nemb, cn_layers):
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)
        self.ninput = nfield * nemb
        self.cross_net = CrossNetwork(self.ninput, cn_layers)
        self.w = torch.nn.Linear(self.ninput, 1, bias=False)

    def forward(self, x):
        """
        :param x:   {'ids': LongTensor B*F, 'vals': FloatTensor B*F}
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        x_emb = self.embedding(x).view(-1, self.ninput)     # B*(FxE)
        xl1 = self.cross_net(x_emb)                         # B*(FxE)
        y = self.w(xl1)                                     # B*1
        return y.squeeze(1)

class DCNModel(torch.nn.Module):
    """
    Model:  Deep & Cross (w/ a neural network)
    Ref:    R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.
    """
    def __init__(self, nfield, nfeat, nemb, cn_layers, mlp_layers, mlp_hid, dropout):
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)
        self.ninput = nfield * nemb
        self.cross_net = CrossNetwork(self.ninput, cn_layers)
        self.mlp = MLP(self.ninput, mlp_layers, mlp_hid, dropout, noutput=mlp_hid)
        self.w = torch.nn.Linear(mlp_hid+self.ninput, 1, bias=False)

    def forward(self, x):
        """
        :param x:   {'ids': LongTensor B*F, 'vals': FloatTensor B*F}
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        x_emb = self.embedding(x).view(-1, self.ninput)         # B*(FxE)
        xl1 = self.cross_net(x_emb)                             # B*(FxE)
        hl2 = self.mlp(x_emb)                                   # B*mlp_hid
        y = self.w(torch.cat([xl1, hl2], dim=1))                # B*1
        return y.squeeze(1)