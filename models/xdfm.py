import torch
import torch.nn.functional as F
from models.layers import Embedding, Linear, MLP

class CompressedInteraction(torch.nn.Module):

    def __init__(self, nfield, nlayers, nfilter):
        super().__init__()
        self.nlayers = nlayers
        self.filters = torch.nn.ModuleList()
        n_prev, n_fc = nfield, 0
        for k in range(nlayers):
            self.filters.append(torch.nn.Conv1d(
                nfield*n_prev, nfilter, kernel_size=1, bias=False))
            n_prev = nfilter
            n_fc += n_prev

        self.affine = torch.nn.Linear(n_fc, 1, bias=False)

    def forward(self, x):
        """
        :param x:   FloatTensor B*F*E
        :return:    FloatTensor B
        """
        xlist = list()
        x0, xk = x.unsqueeze(2), x                              # x0: B*F*1*E
        for k in range(self.nlayers):                           # xk: B*F/nfilter*E
            h = x0 * xk.unsqueeze(1)                            # B*F*F/nfilter*E
            bsz, nfield, nfilter, nemb = h.size()
            xk = F.relu(self.filters[k](h.view(bsz, -1, nemb))) # B*nfilter*E
            xlist.append(torch.sum(xk, dim=-1))                 # B*nfilter

        y = self.affine(torch.cat(xlist, dim=1))                # B*1
        return y.squeeze(1)                                     # B


class CINModel(torch.nn.Module):
    """
    Model:  CIN (w/o a neural network)
    Ref:    J Lian, et al. xDeepFM: Combining Explicit and
                Implicit Feature Interactions for Recommender Systems, 2018.
    """
    def __init__(self, nfield, nfeat, nemb, cin_layers, nfilter):
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)
        self.linear = Linear(nfeat)
        self.cin = CompressedInteraction(nfield, cin_layers, nfilter)

    def forward(self, x):
        """
        :param x:   {'ids': LongTensor B*F, 'vals': FloatTensor B*F}
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        x_emb = self.embedding(x)                       # B*F*E
        y = self.linear(x)+self.cin(x_emb)              # B
        return y

class xDeepFMModel(torch.nn.Module):
    """
    Model:  xDeepFM (w/ a neural network)
    Ref:    J Lian, et al. xDeepFM: Combining Explicit and
                Implicit Feature Interactions for Recommender Systems, 2018.
    """
    def __init__(self, nfield, nfeat, nemb, cin_layers, nfilter, mlp_layers, mlp_hid, dropout):
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)
        self.linear = Linear(nfeat)
        self.cin = CompressedInteraction(nfield, cin_layers, nfilter)
        self.ninput = nfield*nemb
        self.mlp = MLP(self.ninput, mlp_layers, mlp_hid, dropout)

    def forward(self, x):
        """
        :param x:   {'ids': LongTensor B*F, 'vals': FloatTensor B*F}
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        x_emb = self.embedding(x)                                                   # B*F*E
        y = self.linear(x)+self.cin(x_emb)+\
            self.mlp(x_emb.view(-1, self.ninput)).squeeze(1)                        # B
        return y