from models.utils import default
from einops import rearrange

import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F

class Attention(nn.Module):
    """" Self-Attention or Cross-Attention """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        """
        :param query_dim:       dimension of each query vector
        :param context_dim:     dimension of each context vector (key & value)
        :param heads:           Number of attention heads
        :param dim_head:        dimension of each head
        """
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, q, context=None):
        """
        :param q:           [bsz, nhid_q, d_q], FloatTensor
        :param context:     [bsz, nhid_k, d_k], FloatTensor
        :return:            [bsz, nhid_q, d_q], FloatTensor
        """
        h = self.heads

        q = self.to_q(q)
        context = default(context, q)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class TCN(nn.Module):
    """ Temporal Convolution Network """
    def __init__(self, nemb: int, nlayer: int = 2,
                 kernel_size: int = 3, bias: bool = False):
        """
        :param nemb:            input dimension of each timestep
        :param nlayer:          Number of temporal convolution layers
        :param kernel_size:     kernel size of each layer
        :param bias:            convolution bias, default False
        """
        super().__init__()
        self.nlayer = nlayer
        self.kernel_size = kernel_size

        self.residual_convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        for _ in range(nlayer):
            self.residual_convs.append(
                nn.Conv1d(nemb, nemb, kernel_size=1, bias=bias))
            self.filter_convs.append(
                nn.Conv1d(nemb, nemb, kernel_size=kernel_size, bias=bias))
            self.gate_convs.append(
                nn.Conv1d(nemb, nemb, kernel_size=kernel_size, bias=bias))

    def forward(self, x):
        """
        :param x:           [bsz, nemb, seq_len], FloatTensor
        :return:            [bsz, nemb, seq_len], FloatTensor
        """
        for idx in range(self.nlayer):
            residual = self.residual_convs[idx](x)          # bsz*nemb*seq_len

            # left pad x to keep constant seq_len
            x = F.pad(x, [self.kernel_size-1, 0])           # bsz*nemb*(seq_len+ksz-1)
            filter = torch.tanh(self.filter_convs[idx](x))  # bsz*nemb*seq_len
            gate = torch.sigmoid(self.gate_convs[idx](x))   # bsz*nemb*seq_len
            x = filter * gate                               # bsz*nemb*seq_len

            x += residual                                   # bsz*nemb*seq_len

        return x                                            # bsz*nemb*seq_len


class MLP(nn.Module):

    def __init__(self, ninput: int, nlayers: int, nhid: int,
                 dropout: float = 0., noutput: int = 1):
        """
        :param ninput:      dimension of input
        :param nlayers:     Number of hidden layers
        :param nhid:        hidden dimension of each layer
        :param dropout:     dropout rate, default 0.
        :param noutput:     dimension of output
        """
        super().__init__()
        layers = list()
        for i in range(nlayers):
            layers.append(nn.Linear(ninput, nhid))
            layers.append(nn.BatchNorm1d(nhid))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            ninput = nhid
        if nlayers==0: nhid = ninput
        layers.append(nn.Linear(nhid, noutput))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x:   [bsz, ninput], FloatTensor
        :return:    [bsz, nouput], FloatTensor
        """
        return self.mlp(x)
