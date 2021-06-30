import torch
from torch import nn
import torch.nn.functional as F
from models.layers import Embedding, MLP

def normalize_adj(adj):
    """normalize and return a adjacency matrix (pytorch 2d tensor)"""
    rowsum = torch.sum(adj, dim=-1)
    d_inv_sqrt = rowsum ** (-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(adj, d_mat_inv_sqrt).transpose(0, 1), d_mat_inv_sqrt)

class GraphConvolutionLayer(nn.Module):

    def __init__(self, ninfeat, noutfeat, bias, dropout):
        super().__init__()
        if bias: self.bias = nn.Parameter(torch.zeros(size=(noutfeat,)))
        else: self.register_parameter('bias', None)

        self.weight = nn.Parameter(torch.zeros(size=(ninfeat, noutfeat)))
        nn.init.xavier_uniform_(self.weight.data)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        '''
        :param x:       FloatTensor B*F*E1
        :param adj:     FloatTensor F*F
        :return:        FloatTensor B*F*(headxE2)
        '''
        x_drop = self.dropout(x)                                # B*F*E1
        support = torch.einsum('bfx,xy', x_drop, self.weight)   # B*F*E2
        output = torch.einsum('xy,bye->bxe', adj, support)      # B*F*E2
        if self.bias is not None:
            output = output + self.bias
        return output                                           # B*F*E2

class GCNModel(nn.Module):
    """
        Model:  Graph Convolutional Networks
        Ref:    T N. Kipf, et al. Semi-Supervised Classification with Graph Convolutional Networks, 2017.
        """
    def __init__(self, nfield, nfeat, nemb, gcn_layers, gcn_hid, mlp_layers, mlp_hid, dropout):
        super().__init__()
        self.embedding = Embedding(nfeat, nemb)

        self.gcn_layers = gcn_layers
        self.gcns = torch.nn.ModuleList()
        ninfeat = nemb
        for _ in range(gcn_layers):
            self.gcns.append(GraphConvolutionLayer(
                ninfeat, gcn_hid, bias=True, dropout=dropout))
            ninfeat = gcn_hid

        self.dropout = nn.Dropout(p=dropout)
        self.affine = MLP(nfield*ninfeat, mlp_layers, mlp_hid, dropout)

    def forward(self, x, adj=None):
        """
        :param x:   {'ids': LongTensor B*F, 'vals': FloatTensor B*F}
        :param adj: np array F*F, default fully connected
        :return:    y of size B, Regression and Classification (+sigmoid)
        """
        h = self.embedding(x)                                   # B*F*E
        if adj is None:
            adj = torch.ones((h.size(1), h.size(1)), dtype=h.dtype, device=h.device)
        adj = normalize_adj(adj)                                # F*F

        for l in range(self.gcn_layers):
            h = self.gcns[l](h, adj)                            # B*F*gcn_hid
            h = F.relu(self.dropout(h))                         # B*F*gcn_hid

        y = self.affine(h.reshape(h.size(0), -1))               # B*1
        return y.squeeze(1)                                     # B