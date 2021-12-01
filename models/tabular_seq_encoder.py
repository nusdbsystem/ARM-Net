from torch import Tensor
from typing import Optional, Tuple

import torch
import torch.nn as nn


class TabularSeqEncoder(nn.Module):
    def __init__(self, nstep: int, nfeat: int, nemb: int, nhid: int, d_hid: int):
        """ Time-series Tabular Data Encoder """
        super(TabularSeqEncoder, self).__init__()
        # self.pos_embedding =
        # self.feat_embedding


    def forward(self, x: Tensor) -> Tensor:
        """
        :param x:       bsz*nstep*nfeat
        :return:        bsz*nhid*nemb
        """
        pass

