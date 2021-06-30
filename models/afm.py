import torch
import torch.nn.functional as F
from models.layers import Embedding, Linear, get_triu_indices

class AttentionalFactorizationMachine(torch.nn.Module):

    def __init__(self, nemb, nattn, dropout):
        super().__init__()
        self.attn_w = torch.nn.Linear(nemb, nattn)
        self.attn_h = torch.nn.Linear(nattn, 1)
        self.attn_p = torch.nn.Linear(nemb, 1)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        """
        :param x:   FloatTensor B*F*E
        """
        nfield = x.size(1)
        vi_indices, vj_indices = get_triu_indices(nfield)
        vi, vj = x[:, vi_indices], x[:, vj_indices]                 # B*(Fx(F-1)/2)*E
        hadamard_prod = vi * vj
        attn_weights = F.relu(self.attn_w(hadamard_prod))           # B*(Fx(F-1)/2)*nattn
        attn_weights = F.softmax(self.attn_h(attn_weights), dim=1)  # B*(Fx(F-1)/2)*1
        attn_weights = self.dropout(attn_weights)
        afm = torch.sum(attn_weights*hadamard_prod, dim=1)          # B*E
        afm = self.dropout(afm)
        return self.attn_p(afm).squeeze(1)                          # B

class AFMModel(torch.nn.Module):
    """
    Model:  Attentional Factorization Machine
    Ref:    J Xiao, et al. Attentional Factorization Machines:
                Learning the Weight of Feature Interactions via Attention Networks, 2017.
    """

    def __init__(self, nfeat, nemb, nattn, dropout):
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)
        self.linear = Linear(nfeat)
        self.afm = AttentionalFactorizationMachine(nemb, nattn, dropout)

    def forward(self, x):
        """
        :param x:   {'ids': LongTensor B*F, 'vals': FloatTensor B*F}
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        y = self.linear(x) + self.afm(self.embedding(x))
        return y