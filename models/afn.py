import torch
import torch.nn as nn
from models.layers import Embedding, MLP

AFN_CONFIG = {
    'lr': [0.003, 0.01, 0.03, 0.1],
    'nemb': [1, 4, 8, 16],
    'afn_hid': [20, 50, 100, 200],
    'mlp_layer': [1, 2, 3],
    'mlp_hid': [50, 100, 200],
    'dropout': [0.0, 0.05],
    'ensemble': [True, False],
}

class AFNModel(nn.Module):
    """
    Model:  Adaptive Factorization Network
    Ref:    W Cheng, et al. Adaptive Factorization Network:
                Learning Adaptive-Order Feature Interactions, 2020.
    """
    def __init__(self, nclass, nfield, nfeat, nemb, afn_hid, mlp_layers, mlp_hid, dropout,
                 ensemble, deep_layers, deep_hid):
        super().__init__()
        self.nfield, self.nfeat = nfield, nfeat
        self.nemb, self.afn_hid = nemb, afn_hid
        self.ensemble = ensemble
        self.dropout = nn.Dropout(p=dropout)

        # embedding
        self.embedding = Embedding(nfeat, nemb)
        self.emb_bn = nn.BatchNorm1d(nfield)

        # afn
        self.afn = nn.Linear(nfield, afn_hid)
        self.afn_bn = nn.BatchNorm1d(afn_hid)
        nn.init.normal_(self.afn.weight, std=0.1)
        nn.init.constant_(self.afn.bias, 0.)

        # MLP
        ninput = afn_hid * nemb
        self.mlp = MLP(ninput, mlp_layers, mlp_hid, dropout, noutput=nclass)

        # ensemble with a neural network
        if ensemble:
            self.deep_embedding = Embedding(nfeat, nemb)
            self.deep_mlp = MLP(nfield*nemb, deep_layers, deep_hid,
                                dropout, noutput=nclass)
            self.ensemble_layer = nn.Linear(2, 1)
            nn.init.constant_(self.ensemble_layer.weight, 0.5)
            nn.init.constant_(self.ensemble_layer.bias, 0.)

    def forward(self, x):
        """
        :param x:   FloatTensor B*F
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        x.clamp_(0.001, 1.)

        # embedding weight clamp
        self.embedding_clip()
        x_emb = self.embedding(x)                                       # B*F*E

        x_log = self.emb_bn(torch.log(x_emb))                           # B*F*E
        x_log = x_log.transpose(1, 2)                                   # B*E*F
        afn = torch.exp(self.afn(x_log))                                # B*E*O
        afn = self.afn_bn(afn.transpose(1, 2))                          # B*O*E
        afn = afn.view(-1, self.afn_hid*self.nemb)                      # B*(OxE)

        afn = self.dropout(afn)
        y = self.mlp(afn)                                               # B*nclass

        if self.ensemble:
            deep_emb = self.deep_embedding(x)                           # B*F*E
            y_deep = self.deep_mlp(
                deep_emb.view(-1, self.nfield*self.nemb))               # B*nclass

            y = torch.stack([y, y_deep], dim=2)                         # B*nclass*2
            y = self.ensemble_layer(y).squeeze(2)                       # B*nclass

        return y                                                        # B*nclass

    def embedding_clip(self):
        ''' keep AFN embeedings positive'''
        with torch.no_grad():
            self.embedding.embedding.abs_()
            self.embedding.embedding.clamp_(min=1e-4)