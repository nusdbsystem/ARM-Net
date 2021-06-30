import torch
import torch.nn as nn
from utils.entmax import EntmaxBisect
from models.layers import Embedding, MLP

class SparseAttLayer(nn.Module):
    def __init__(self, nfield, nemb, nhid, alpha=1.5):
        """ Sparse Attention Layer w/o bilinear weight"""
        super(SparseAttLayer, self).__init__()
        if alpha == 1.:
            self.sparsemax = nn.Softmax(dim=-1)
        else:
            self.sparsemax = EntmaxBisect(alpha, dim=-1)

        self.Q = nn.Parameter(torch.zeros(nhid, nemb))              # O*E
        nn.init.xavier_uniform_(self.Q, gain=1.414)

        self.values = nn.Parameter(torch.zeros(nhid, nfield))       # O*F
        nn.init.xavier_uniform_(self.values, gain=1.414)

    def forward(self, x):
        """
        :param x:       B*F*E
        :return:        Att_weights (B*O*F), Key (B*F*E) <-> Q (O*E) -> W (O*F)
        """
        keys = x                                                    # B*F*E

        # sparse gates
        att_gates = torch.einsum('bfe,oe->bof', keys, self.Q)       # B*O*F
        sparse_gates = self.sparsemax(att_gates)                    # B*O*F

        return torch.einsum('bof,of->bof', sparse_gates, self.values)

class ARMNetModel(nn.Module):
    """
    Model:  Adaptive Relation Modeling Network (w/o bilinear weight => One-Head)
    """

    def __init__(self, nfield, nfeat, nemb, nhead, alpha, arm_hid, mlp_layers, mlp_hid,
                 dropout, ensemble, deep_layers, deep_hid):
        super().__init__()
        if nhead > 1: print(f'{nhead} attention heads of {arm_hid} neurons '
                            f'<===> 1 head of {nhead*arm_hid} neurons')
        self.nfield, self.nfeat = nfield, nfeat
        self.nemb, self.arm_hid = nemb, nhead*arm_hid
        self.ensemble = ensemble

        # embedding
        self.embedding = Embedding(nfeat, nemb)
        self.emb_bn = nn.BatchNorm1d(nfield)

        # arm
        self.attn_layer = SparseAttLayer(nfield, nemb, self.arm_hid, alpha)
        self.arm_bn = nn.BatchNorm1d(self.arm_hid)

        # MLP
        self.mlp = MLP(self.arm_hid*nemb, mlp_layers, mlp_hid, dropout)

        if ensemble:
            self.deep_embedding = Embedding(nfeat, nemb)
            self.deep_mlp = MLP(nfield*nemb, deep_layers, deep_hid, dropout)
            self.ensemble_layer = nn.Linear(2, 1)
            nn.init.constant_(self.ensemble_layer.weight, 0.5)
            nn.init.constant_(self.ensemble_layer.bias, 0.)

    def forward(self, x):
        """
        :param x:   {'ids': LongTensor B*F, 'vals': FloatTensor B*F}
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        x['vals'].clamp_(0.001, 1.)
        x_emb = self.embedding(x)                                       # B*F*E

        arm_weights = self.attn_layer(x_emb)                            # B*O*F

        arm = torch.einsum('bfe,bof->boe', x_emb, arm_weights)          # B*O*E
        arm = torch.exp(arm)                                            # B*O*E
        arm = arm.view(-1, self.arm_hid, self.nemb)                     # B*O*E
        arm = self.arm_bn(arm).view(arm.size(0), -1)                    # B*(OxE)

        y = self.mlp(arm)                                               # B*1

        if self.ensemble:
            deep_emb = self.deep_embedding(x)
            y_deep = self.deep_mlp(
                deep_emb.view(-1, self.nfield*self.nemb))               # B*1

            y = torch.cat([y, y_deep], dim=1)                           # B*2
            y = self.ensemble_layer(y)                                  # B*1

        return y.squeeze(1)                                             # B