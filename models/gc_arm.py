import torch
import torch.nn as nn
from utils.entmax import EntmaxBisect
from models.layers import Embedding, MLP

class GC_SparseAttLayer(nn.Module):
    def __init__(self, nhead, nfield, nemb, nhid, alpha=1.5):
        """ Sparse Attention Layer with Global Context Info"""
        super(GC_SparseAttLayer, self).__init__()
        if alpha == 1.:
            self.sparsemax = nn.Softmax(dim=-1)
        else:
            self.sparsemax = EntmaxBisect(alpha, dim=-1)

        # bilinear attn
        self.Q = nn.Parameter(torch.zeros(nhead, nhid, nemb))               # K*O*E
        nn.init.xavier_uniform_(self.Q, gain=1.414)

        self.bilinear = nn.Parameter(torch.zeros(nhead, nemb, nemb))        # K*E*E
        nn.init.xavier_uniform_(self.bilinear, gain=1.414)

        self.values = nn.Parameter(torch.zeros(nhead, nhid, nfield))        # K*O*F
        nn.init.xavier_uniform_(self.values, gain=1.414)

    def forward(self, x):
        """
        :param x:   B*F*E
        :return:    Att_weights (B*O*F), Key (B*F*E) <-> Q (K*O*E) -> W (K*O*F)
        """
        keys = x                                                    # B*F*E

        ## bilinear
        bilinear = torch.einsum('bfx,kxy,koy->bkof',
                                 keys, self.bilinear, self.Q)       # B*K*O*F

        ## global context
        context = torch.sum(x, dim=1)                               # B*E
        global_context = torch.einsum('bx,kxy,koy->bko',
                                 context, self.bilinear, self.Q)    # B*K*O

        attn_gates = bilinear + global_context.unsqueeze(-1)        # B*K*O*F

        sparse_gates = self.sparsemax(attn_gates)                   # B*K*O*F

        attn_weights = torch.einsum('bkof,kof->bkof',
                                     sparse_gates, self.values)

        return attn_weights

class GC_ARMModel(nn.Module):
    """
    Model:  Adaptive Relation Modeling Network + Global Context
    Ref:    [Global Context] B Yang, et al. Context-Aware Self-Attention Networks, 2019
    """
    def __init__(self, nfield, nfeat, nemb, nhead, alpha, arm_hid, mlp_layers, mlp_hid,
                 dropout, ensemble, deep_layers, deep_hid):
        super().__init__()
        self.nfield, self.nfeat, self.nemb = nfield, nfeat, nemb
        self.nhead, self.arm_hid = nhead, arm_hid
        self.ensemble = ensemble
        self.dropout = nn.Dropout(p=dropout)

        # embedding
        self.embedding = Embedding(nfeat, nemb)
        self.emb_bn = nn.BatchNorm1d(nfield)

        # arm
        self.attn_layers = GC_SparseAttLayer(nhead, nfield, nemb, arm_hid, alpha)
        self.arm_bn = nn.BatchNorm1d(nhead*arm_hid)

        # MLP
        self.mlp = MLP(nhead*arm_hid*nemb, mlp_layers, mlp_hid, dropout)

        if ensemble:
            self.deep_embedding = Embedding(nfeat, nemb)
            self.deep_mlp = MLP(nfield*nemb, deep_layers, deep_hid, dropout)
            self.ensemble_layer = nn.Linear(2, 1)
            nn.init.constant_(self.ensemble_layer.weight, 0.5)
            nn.init.constant_(self.ensemble_layer.bias, 0.)

    def forward(self, x):
        """
        :param x:   {'id': LongTensor B*F, 'value': FloatTensor B*F}
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        x['value'].clamp_(0.001, 1.)
        x_emb = self.embedding(x)                                       # B*F*E

        x_exp = self.emb_bn(torch.exp(x_emb))                           # B*F*E
        arm_weights = self.attn_layers(x_emb)                           # B*K*O*F

        arm = torch.einsum('bfe,bkof->bkoe', x_exp, arm_weights)        # B*K*O*E
        arm = arm.view(arm.size(0), -1, self.nemb)                      # B*(KxO)*E
        arm = self.arm_bn(arm).view(arm.size(0), -1)                    # B*(KxOxE)

        y = self.mlp(arm)                                               # B*1

        if self.ensemble:
            deep_emb = self.deep_embedding(x)
            y_deep = self.deep_mlp(
                deep_emb.view(-1, self.nfield*self.nemb))               # B*1

            y = torch.cat([y, y_deep], dim=1)                           # B*2
            y = self.ensemble_layer(y)                                  # B*1

        return y.squeeze(1)                                             # B