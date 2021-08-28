import math
import numpy as np
from einops import rearrange, reduce
import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    ''' simplified version for ALL numerical features '''

    def __init__(self, nfeat, nemb):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(nfeat, nemb)) # F*E
        nn.init.xavier_uniform_(self.embedding)

    def forward(self, x):
        """
        :param x:   FloatTensor B*F
        :return:    embeddings B*F*E
        """
        return torch.einsum('bf,fe->bfe', x, self.embedding)    # B*F*E

class Linear(nn.Module):

    def __init__(self, nfeat):
        super().__init__()
        self.weight = Embedding(nfeat, 1)
        self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, x):
        """
        :param x:   FloatTensor B*F
        :return:    linear transform of x
        """
        linear = self.weight(x).squeeze(2)                      # B*F
        return torch.sum(linear, dim=1) + self.bias             # B

class FactorizationMachine(nn.Module):

    def __init__(self, reduce_dim=True):
        super().__init__()
        self.reduce_dim = reduce_dim

    def forward(self, x):
        """
        :param x:   FloatTensor B*F*E
        """
        square_of_sum = torch.sum(x, dim=1)**2                  # B*E
        sum_of_square = torch.sum(x**2, dim=1)                  # B*E
        fm = square_of_sum - sum_of_square                      # B*E
        if self.reduce_dim:
            fm = torch.sum(fm, dim=1)                           # B
        return 0.5 * fm                                         # B*E/B

def get_triu_indices(n, diag_offset=1):
    """get the row, col indices for the upper-triangle of an (n, n) array"""
    return np.triu_indices(n, diag_offset)

def get_all_indices(n):
    """get all the row, col indices for an (n, n) array"""
    return map(list, zip(*[(i, j) for i in range(n) for j in range(n)]))

class MLP(nn.Module):

    def __init__(self, ninput, nlayers, nhid, dropout, noutput=1):
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
        :param x:   FloatTensor B*ninput
        :return:    FloatTensor B*nouput
        """
        return self.mlp(x)

def normalize_adj(adj):
    """normalize and return a adjacency matrix (numpy array)"""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


class SelfAttnLayer(nn.Module):
    def __init__(self, nemb):
        """ Self Attention Layer (scaled dot-product)"""
        super(SelfAttnLayer, self).__init__()
        self.Wq = nn.Linear(nemb, nemb, bias=False)
        self.Wk = nn.Linear(nemb, nemb, bias=False)
        self.Wv = nn.Linear(nemb, nemb, bias=False)

    def forward(self, x):
        """
        :param x:   B*F*E
        :return:    B*F*E
        """
        query, key, value = self.Wq(x), self.Wk(x), self.Wv(x)
        d_k = query.size(-1)
        scores = torch.einsum('bxe,bye->bxy', query, key)               # B*F*F
        attn_weights = F.softmax(scores / math.sqrt(d_k), dim=-1)       # B*F*F
        return torch.einsum('bxy,bye->bxe', attn_weights, value), attn_weights

class MultiHeadAttention(nn.Module):
    '''Multi-head Attention Module'''
    def __init__(self, nhead, ninput, d_k, d_v, dropout=0.0):
        super().__init__()
        self.nhead = nhead

        self.w_qs = nn.Linear(ninput, nhead * d_k)
        self.w_ks = nn.Linear(ninput, nhead * d_k)
        self.w_vs = nn.Linear(ninput, nhead * d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (ninput + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (ninput + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (ninput + d_v)))

        self.fc = nn.Linear(nhead * d_v, ninput)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(ninput)

    def forward(self, q, k, v, mask=None):
        '''
        :param q:       B*F_l*E_k
        :param k:       B*F_t*E_k
        :param v:       B*F_t*E_k
        :param mask:    B*F_l*F_t
        :return:        B*F_l*(nhead * d_v)
        '''
        residual = q
        q = rearrange(self.w_qs(q), 'b l (head k) -> head b l k', head=self.nhead)
        k = rearrange(self.w_ks(k), 'b t (head k) -> head b t k', head=self.nhead)
        v = rearrange(self.w_vs(v), 'b t (head v) -> head b t v', head=self.nhead)
        attn = torch.einsum('hblk,hbtk->hblt', [q, k]) / np.sqrt(q.shape[-1])
        if mask is not None:
            attn = attn.masked_fill(mask[None], -np.inf)
        attn = torch.softmax(attn, dim=3)
        output = torch.einsum('hblt,hbtv->hblv', [attn, v])
        output = rearrange(output, 'head b l v -> b l (head v)')
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn