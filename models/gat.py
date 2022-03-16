import torch
from torch import nn
import torch.nn.functional as F
from models.layers import get_all_indices, Embedding, MLP

class GraphAttentionLayer(nn.Module):

    def __init__(self, ninfeat, noutfeat, nhead, dropout, alpha):
        super().__init__()
        self.nhead = nhead

        self.W = nn.ParameterList()
        self.a = nn.ModuleList()
        for _ in range(nhead):
            self.W.append(nn.Parameter(torch.zeros(size=(ninfeat, noutfeat))))
            nn.init.xavier_uniform_(self.W[-1].data, gain=1.414)
            self.a.append(nn.Linear(2*noutfeat, 1, bias=False))

        self.dropout = nn.Dropout(p=dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        '''
        :param x:       FloatTensor B*F*E1
        :param adj:     FloatTensor F*F
        :return:        FloatTensor B*F*(headxE2)
        '''
        nfield = x.size(1)
        zero_vec = -9e15 * torch.ones_like(adj)
        mask = torch.where(adj > 0, adj, zero_vec)                                  # F*F

        h_list = []
        vi_indices, vj_indices = get_all_indices(nfield)
        for head in range(self.nhead):
            h = torch.einsum('bfi,io->bfo', x, self.W[head])                        # B*F*E2
            vi, vj = h[:, vi_indices], h[:, vj_indices]
            hh = torch.cat([vi, vj], dim=2)                                         # B*(FxF)*(2xE2)

            e = self.leakyrelu(self.a[head](hh))                                    # B*(FxF)*1

            attn = torch.einsum('bxy,xy->bxy', e.view(-1, nfield, nfield), mask)    # B*F*F
            attn = self.dropout(F.softmax(attn, dim=-1))                            # B*F*F

            h_prime = torch.einsum('bxy,bye->bxe', attn, h)                         # B*F*E2
            h_list.append(h_prime)

        return torch.cat(h_list, dim=2)                                             # B*F*(headxE2)

class GATModel(nn.Module):
    """
        Model:  Graph Attention Networks
        Ref:    P Veličković, et al. Graph Attention Networks, 2018.
        """
    def __init__(self, nfield, nfeat, nemb, gat_layers, gat_hid, mlp_layers, mlp_hid, dropout, alpha=0.2, nhead=8):
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)

        self.gat_layers = gat_layers
        self.gats = torch.nn.ModuleList()
        ninfeat = nemb
        for _ in range(gat_layers):
            self.gats.append(GraphAttentionLayer(ninfeat, gat_hid, nhead, dropout, alpha))
            ninfeat = nhead*gat_hid

        self.dropout = nn.Dropout(p=dropout)
        self.affine = MLP(nfield*ninfeat, mlp_layers, mlp_hid, dropout)

    def forward(self, x, adj=None):
        """
        :param x:   {'id': LongTensor B*F, 'value': FloatTensor B*F}
        :param adj:     FloatTensor F*F, default fully connected
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        h = self.embedding(x)                                                   # B*F*E
        if adj is None:
            adj = torch.ones((h.size(1), h.size(1)), dtype=h.dtype, device=h.device)
        for l in range(self.gat_layers):
            h = self.gats[l](h, adj)                                            # B*F*(nheadxgat_hid)
            h = F.elu(self.dropout(h))

        y = self.affine(h.view(h.size(0), -1))                                  # B*1
        return y.squeeze(1)                                                     # B