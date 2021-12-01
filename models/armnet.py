import torch
import torch.nn as nn
from utils.entmax import EntmaxBisect


class SparseAttention(nn.Module):
    def __init__(self, nfield: int, d_k: int, nhid: int,
                 nemb: int, alpha: float = 1.5):
        """ Sparse Attention Layer w/o bilinear weight"""
        super(SparseAttention, self).__init__()
        self.sparsemax = nn.Softmax(dim=-1) if alpha == 1. \
            else EntmaxBisect(alpha, dim=-1)

        self.w_k = nn.Linear(nemb, d_k, bias=False)                     # nemb*d_k
        self.query = nn.Parameter(torch.zeros(nhid, d_k))               # nhid*d_k
        self.values = nn.Parameter(torch.zeros(nhid, nfield))           # nhid*nfield
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query, gain=1.414)
        nn.init.xavier_uniform_(self.values, gain=1.414)

    def forward(self, x):
        """
        :param x:       [bsz, nfield, nemb]
        :return:        Att_weights [bsz, nhid, nfield]
        """
        keys = self.w_k(x)                                              # bsz*nfield*d_k
        att_gates = torch.einsum('bfe,oe->bof', keys, self.query)       # bsz*nhid*nfield
        sparse_gates = self.sparsemax(att_gates)                        # bsz*nhid*nfield
        return torch.einsum('bof,of->bof', sparse_gates, self.values)   # bsz*nhid*nfield


class ARMModule(nn.Module):
    """
    Model:  Adaptive Relation Modeling Network (w/o bilinear weight => One-Head)
    """

    def __init__(self, nfield: int, nemb: int, d_k: int, alpha: float, nhid: int):
        super().__init__()

        # arm
        self.attn_layer = SparseAttention(nfield, d_k, nhid, nemb, alpha)
        self.bn = nn.BatchNorm1d(nhid)

    def forward(self, x):
        """
        :param x:       [bsz, nfield, nemb]
        :return:        [bsz, nhid, nemb]
        """
        att_weights = self.attn_layer(x)                                # bsz*nhid*nfield
        interaction = torch.exp(torch.einsum(
            'bfe,bof->boe', x, att_weights))                            # bsz*nhid*nfield
        return self.bn(interaction)                                     # bsz*nhid*nfield
