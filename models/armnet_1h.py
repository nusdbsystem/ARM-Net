from einops import rearrange
import torch
import torch.nn as nn
from utils.entmax import EntmaxBisect
from models.layers import Embedding, MLP


class SparseAttention(nn.Module):
    def __init__(self, nfield: int, d_k: int, nhid: int, nemb: int, alpha: float = 1.5):
        """ Sparse Attention Layer w/ shared bilinear weight -> one-head """
        super(SparseAttention, self).__init__()
        self.sparsemax = nn.Softmax(dim=-1) if alpha == 1. \
            else EntmaxBisect(alpha, dim=-1)

        self.scale = d_k ** -0.5
        self.bilinear_w = nn.Linear(nemb, d_k, bias=False)              # nemb*d_k
        self.query = nn.Parameter(torch.zeros(nhid, d_k))               # nhid*d_k
        self.values = nn.Parameter(torch.zeros(nhid, nfield))           # nhid*nfield
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query, gain=1.414)
        nn.init.xavier_uniform_(self.values, gain=1.414)

    def forward(self, x):
        """
        :param x:       [bsz, nfield, nemb], FloatTensor
        :return:        Att_weights [bsz, nhid, nfield], FloatTensor
        """
        keys = self.bilinear_w(x)                                       # bsz*nfield*d_k
        att_gates = torch.einsum('bfe,oe->bof',
                                 keys, self.query) * self.scale         # bsz*nhid*nfield
        sparse_gates = self.sparsemax(att_gates)                        # bsz*nhid*nfield
        return torch.einsum('bof,of->bof', sparse_gates, self.values)   # bsz*nhid*nfield


class ARMNetModel(nn.Module):
    """
        Model:  Adaptive Relation Modeling Network (w/ shared bilinear weight => One-Head)
        Important Hyper-Params: alpha (sparsity), nhid (exponential neurons)
    """
    def __init__(self, nfield: int, nfeat: int, nemb: int, alpha: float, nhid: int, d_k: int,
                 mlp_nlayer: int, mlp_nhid: int, dropout: float, ensemble: bool,
                 deep_nlayer: int, deep_nhid: int, noutput: int = 1):
        '''
        :param nfield:          Number of Fields
        :param nfeat:           Total Number of Features
        :param nemb:            Feature Embedding size
        :param alpha:           Sparsity hyper-parameter for ent-max
        :param nhid:            Number of Exponential Neuron
        :param d_k:             Inner Query/Key dimension in Attention (default: nemb)
        :param mlp_nlayer:      Number of layers for prediction head
        :param mlp_nhid:        Number of hidden neurons for prediction head
        :param dropout:         Dropout rate
        :param ensemble:        Whether to Ensemble with a DNN
        :param deep_nlayer:     Number of layers for Ensemble DNN
        :param deep_nhid:       Number of hidden neurons for Ensemble DNN
        :param noutput:         Number of prediction output, e.g., 1 for binary cls
        '''
        super().__init__()
        # embedding
        self.embedding = Embedding(nfeat, nemb)
        # arm
        self.attn_layer = SparseAttention(nfield, d_k, nhid, nemb, alpha)
        self.arm_bn = nn.BatchNorm1d(nhid)
        # MLP
        self.mlp = MLP(nhid * nemb, mlp_nlayer, mlp_nhid, dropout, noutput=noutput)

        if ensemble:
            self.deep_embedding = Embedding(nfeat, nemb)
            self.deep_mlp = MLP(nfield * nemb, deep_nlayer, deep_nhid, dropout, noutput=noutput)
            self.ensemble_layer = nn.Linear(2*noutput, noutput)
            nn.init.constant_(self.ensemble_layer.weight, 0.5)
            nn.init.constant_(self.ensemble_layer.bias, 0.)

    def forward(self, x):
        """
        :param x:   {'id': [bsz, nfield], LongTensor, 'value': [bsz, nfield], FloatTensor}
        :return:    y: [bsz], FloatTensor of size B, for Regression or Classification
        """
        x['value'].clamp_(1e-3, 1.)
        x_arm = self.embedding(x)                                       # bsz*nfield*nemb

        arm_weight = self.attn_layer(x_arm)                             # bsz*nhid*nfield
        x_arm = self.arm_bn(torch.exp(
            torch.einsum('bfe, bof->boe', x_arm, arm_weight)))          # bsz*nhid*nemb
        x_arm = rearrange(x_arm, 'b o e -> b (o e)')                    # bsz*(nhid*nemb)

        y = self.mlp(x_arm)                                             # B*noutput
        if hasattr(self, 'ensemble_layer'):
            x_deep = self.deep_embedding(x)                             # bsz*nfield*nemb
            x_deep = rearrange(x_deep, 'b f e -> b (f e)')              # bsz*(nfield*nemb)
            y_deep = self.deep_mlp(x_deep)                              # bsz*noutput

            y = torch.cat([y, y_deep], dim=1)                           # bsz*(2*noutput)
            y = self.ensemble_layer(y)                                  # bsz*noutput

        return y.squeeze()                                              # bsz*noutput, squeezed if ndim=1
