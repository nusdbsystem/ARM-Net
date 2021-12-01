from torch import Tensor
from typing import Optional, Tuple

import torch
import torch.nn as nn


class TabularSeqEncoder(nn.Module):
    def __init__(self, nstep: int, nfield: int, nfeat: int,
                 nemb: int, nhid: int, d_hid: int):
        """ Time-series Tabular Data Encoder """
        super(TabularSeqEncoder, self).__init__()
        self.total_fields: int = nstep * nfield

        self.global_embedding = nn.Embedding(self.total_fields, nemb)
        self.feat_embedding = nn.Embedding(nfeat, nemb)


    def forward(self, x: Tensor) -> Tensor:
        """
        :param x:       bsz*nstep*nfeat
        :return:        bsz*nhid*nemb
        """
        pass

