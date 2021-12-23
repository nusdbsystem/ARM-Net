from einops import rearrange, repeat
from torch import Tensor
import torch
import torch.nn as nn
from models.armnet import ARMModule
from models.layers import Attention


class TabularSeqEncoder(nn.Module):
    def __init__(self, nstep: int, nfield: int, nfeat: int, nemb: int, alpha: float,
                 nhid: int, nquery: int = 4):
        """ Time-series Tabular Data Encoder """
        super(TabularSeqEncoder, self).__init__()
        # TODO: add global embedding
        self.feature_embedding = nn.Embedding(nfeat, nemb)
        self.arm = ARMModule(nfield, nemb, nemb, alpha, nhid)

        # TODO: replace cross-attn with MLP for dim reduction
        # output cross-attn, aggregating info within each step
        self.query = nn.Parameter(torch.randn(nstep, nquery, nemb))
        self.step_query_attn = Attention(query_dim=nemb, context_dim=nemb)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x:       [bsz, nstep, nfield], LongTensor
        :return:        [bsz, nstep, nemb], FloatTensor
        """
        bsz = x.size(0)
        x = self.feature_embedding(x)                                   # bsz*nstep*nfield*nemb
        x = rearrange(x, 'b t f e -> (b t) f e')                        # (bsz*nstep)*nfield*nemb
        x = self.arm(x)                                                 # (bsz*nstep)*nhid*nemb

        query = repeat(self.query, 't q e -> (b t) q e', b=bsz)         # (bsz*nstep)*nquery*nemb
        x = self.step_query_attn(query, context=x)                      # (bsz*nstep)*nquery*nemb
        x = rearrange(x, '(b t) q e -> b t (q e)', b=bsz)               # bsz*nstep*(nquery*nemb)
        return x
