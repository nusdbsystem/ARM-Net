from ray import tune
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


ARM_CONFIG = {
    # training config
    'batch_size': 64,
    'lr': tune.grid_search([1e-3, 3e-3, 1e-2, 3e-2]),
    # model config
    'dropout': 0.0,
    'mlp_layer': 2,
    'mlp_hid': 128,
    # model config - grid search
    'nemb': tune.grid_search([1, 2, 4, 8, 16, 32]),
    'alpha': tune.grid_search([1.0, 1.3, 1.5, 1.7, 2.0]),
    'arm_hid': tune.grid_search([8, 16, 32, 64, 128, 256]),
    'ensemble': True
}


class ARMNetModel(nn.Module):
    """
    Model:  Adaptive Relation Modeling Network (w/o bilinear weight => One-Head)
    """

    def __init__(self, nclass, nfield, nfeat, nemb, alpha, arm_hid, mlp_layers, mlp_hid,
                 dropout, ensemble, deep_layers, deep_hid):
        super().__init__()
        self.nfield, self.nfeat = nfield, nfeat
        self.nemb, self.arm_hid = nemb, arm_hid
        self.ensemble = ensemble

        # embedding
        self.embedding = Embedding(nfeat, nemb)
        self.emb_bn = nn.BatchNorm1d(nfield)

        # arm
        self.attn_layer = SparseAttLayer(nfield, nemb, self.arm_hid, alpha)
        self.arm_bn = nn.BatchNorm1d(self.arm_hid)

        # MLP
        self.mlp = MLP(self.arm_hid*nemb, mlp_layers, mlp_hid, dropout, noutput=nclass)

        if ensemble:
            self.deep_embedding = Embedding(nfeat, nemb)
            self.deep_mlp = MLP(nfield*nemb, deep_layers, deep_hid, dropout, noutput=nclass)
            self.ensemble_layer = nn.Linear(2, 1)
            nn.init.constant_(self.ensemble_layer.weight, 0.5)
            nn.init.constant_(self.ensemble_layer.bias, 0.)

    def forward(self, x):
        """
        :param x:   FloatTensor B*F
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        x_emb = self.embedding(x)                                       # B*F*E

        arm_weights = self.attn_layer(x_emb)                            # B*O*F

        arm = torch.einsum('bfe,bof->boe', x_emb, arm_weights)          # B*O*E
        arm = torch.exp(arm)                                            # B*O*E
        arm = arm.view(-1, self.arm_hid, self.nemb)                     # B*O*E
        arm = self.arm_bn(arm).view(arm.size(0), -1)                    # B*(OxE)

        y = self.mlp(arm)                                               # B*nclass

        if self.ensemble:
            deep_emb = self.deep_embedding(x)
            y_deep = self.deep_mlp(
                deep_emb.view(-1, self.nfield*self.nemb))               # B*nclass

            y = torch.stack([y, y_deep], dim=2)                         # B*nclass*2
            y = self.ensemble_layer(y).squeeze(2)                       # B*nclass

        return y                                                        # B*nclass