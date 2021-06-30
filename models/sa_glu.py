import torch
import torch.nn as nn
from models.layers import Embedding, MLP, SelfAttnLayer

class SA_GLUModel(nn.Module):
    """
    Model:  Self Attention + Gated Linear Unit
    Ref:    Y N. Dauphin, et al. Language Modeling with Gated Convolutional Networks, 2017
    """
    def __init__(self, nfield, nfeat, nemb, mlp_layers, mlp_hid,
                 dropout, ensemble, deep_layers, deep_hid):
        super().__init__()
        self.nfield, self.nfeat, self.nemb = nfield, nfeat, nemb
        self.ensemble = ensemble
        self.dropout = nn.Dropout(p=dropout)

        # embedding
        self.embedding = Embedding(nfeat, nemb)
        self.emb_bn = nn.BatchNorm1d(nfield)

        # self attn
        self.self_attn_w = SelfAttnLayer(nemb)
        self.self_attn_v = SelfAttnLayer(nemb)
        self.w_b = nn.Parameter(torch.zeros(nemb,))
        self.v_b = nn.Parameter(torch.zeros(nemb,))

        # MLP
        self.mlp = MLP(nfield*nemb, mlp_layers, mlp_hid, dropout)

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
        x_emb = self.embedding(x)                               # B*F*E

        xw = self.self_attn_w(x_emb)[0]+self.w_b                # B*F*E
        xv = self.self_attn_v(x_emb)[0]+self.v_b                # B*F*E
        glu = xw * torch.sigmoid(xv)                            # B*F*E

        glu = self.dropout(glu.view(xw.size(0), -1))            # B*(FxE)
        y = self.mlp(glu)                                       # B*1

        if self.ensemble:
            deep_emb = self.deep_embedding(x)
            y_deep = self.deep_mlp(
                deep_emb.view(-1, self.nfield*self.nemb))       # B*1

            y = torch.cat([y, y_deep], dim=1)                   # B*2
            y = self.ensemble_layer(y)                          # B*1

        return y.squeeze(1)                                     # B