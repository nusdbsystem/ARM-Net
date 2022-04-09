import torch
import torch.nn as nn
from utils.entmax import EntmaxBisect


class SparseAttention(nn.Module):
    def __init__(self, nfield: int, d_k: int, nhid: int, nemb: int, alpha: float = 1.5):
        """ Sparse Attention Layer w/o bilinear weight"""
        super(SparseAttention, self).__init__()
        self.sparsemax = nn.Softmax(dim=-1) if alpha == 1. \
            else EntmaxBisect(alpha, dim=-1)

        self.scale = d_k ** -0.5
        self.w_k = nn.Linear(nemb, d_k, bias=False)                     # nemb*d_k
        self.query = nn.Parameter(torch.zeros(nhid, d_k))               # nhid*d_k
        self.value = nn.Parameter(torch.zeros(nhid, nfield))            # nhid*nfield
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query, gain=1.414)
        nn.init.xavier_uniform_(self.value, gain=1.414)

    def forward(self, x):
        """
        :param x:       [bsz, nfield, nemb], FloatTensor
        :return:        Att_weights [bsz, nhid, nfield], FloatTensor
        """
        keys = self.w_k(x)                                              # bsz*nfield*d_k
        att_gates = torch.einsum('bfe,oe->bof',
                                 keys, self.query) * self.scale         # bsz*nhid*nfield
        sparse_gates = self.sparsemax(att_gates)                        # bsz*nhid*nfield
        return torch.einsum('bof,of->bof', sparse_gates, self.value)    # bsz*nhid*nfield


class ARMModule(nn.Module):
    """
    Model:      Adaptive Relation Modeling Network (w/o bilinear weight => One-Head, no position)
    Reference:  https://dl.acm.org/doi/10.1145/3448016.3457321
    """

    def __init__(self, nfield: int, nemb: int, d_k: int, alpha: float, nhid: int):
        super().__init__()
        self.attn_layer = SparseAttention(nfield, d_k, nhid, nemb, alpha)
        self.bn = nn.BatchNorm1d(nhid)

    def forward(self, x):
        """
        :param x:       [bsz, nfield, nemb], FloatTensor
        :return:        [bsz, nhid, nemb], FloatTensor
        """
        att_weight = self.attn_layer(x)                                 # bsz*nhid*nfield
        if not self.training:                                           # for tabular weight visualization
            self.attn_weight = att_weight.detach()                      # bsz*nhid*nfield
        interaction = torch.exp(torch.einsum(
            'bfe,bof->boe', x, att_weight))                             # bsz*nhid*nfield
        return self.bn(interaction)                                     # bsz*nhid*nfield
