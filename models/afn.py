import torch
import torch.nn as nn
from models.layers import Embedding, MLP

class AFNModel(nn.Module):
    """
    Model:  Adaptive Factorization Network
    Ref:    W Cheng, et al. Adaptive Factorization Network:
                Learning Adaptive-Order Feature Interactions, 2020.
    """
    def __init__(self, nfield, nfeat, nemb, afn_hid, mlp_layers, mlp_hid, dropout,
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
        self.mlp = MLP(ninput, mlp_layers, mlp_hid, dropout)
        # TODO: original AFN adopt order:   Linear->ReLU->BN->Dropout
        #  recent reserach recommend:       Linear->BN->ReLU->Dropout

        # ensemble with a neural network
        if ensemble:
            self.deep_embedding = Embedding(nfeat, nemb)
            # TODO: original AFN adopt nn.init.normal_(feat_emb_mlp.weight, std=0.1)
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

        # embedding weight clamp
        self.embedding_clip()
        x_emb = self.embedding(x)                                       # B*F*E

        x_log = self.emb_bn(torch.log(x_emb))                           # B*F*E
        x_log = x_log.transpose(1, 2)                                   # B*E*F
        afn = torch.exp(self.afn(x_log))                                # B*E*O
        afn = self.afn_bn(afn.transpose(1, 2))                          # B*O*E
        afn = afn.view(-1, self.afn_hid*self.nemb)                      # B*(OxE)

        afn = self.dropout(afn)
        y = self.mlp(afn)                                               # B*1

        if self.ensemble:
            deep_emb = self.deep_embedding(x)                           # B*F*E
            y_deep = self.deep_mlp(
                deep_emb.view(-1, self.nfield*self.nemb))               # B*1

            y = torch.cat([y, y_deep], dim=1)                           # B*2
            y = self.ensemble_layer(y)                                  # B*1

        return y.squeeze(1)                                             # B

    def embedding_clip(self):
        ''' keep AFN embeedings positive'''
        with torch.no_grad():
            self.embedding.embedding.weight.abs_()
            self.embedding.embedding.weight.clamp_(min=1e-4)