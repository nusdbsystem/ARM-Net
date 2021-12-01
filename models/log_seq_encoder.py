from torch import Tensor
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.tabular_seq_encoder import TabularSeqEncoder
from models.text_seq_encoder import  TextSeqEncoder

class LogSeqEncoder(nn.Module):
    def __init__(self, nstep: int, nfield: int, nemb: int, nhid: int, d_hid: int,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 d_feedforward: int = 2048,
                 dropout: float = 0.1,
                 d_classifier: int = 512,
                 nclass: int = 1
                 ):
        """ Tabular & Text Sequences Data Encoder """
        super(LogSeqEncoder, self).__init__()
        self.tabular_encoder = TabularSeqEncoder(nstep, nfield, nemb, nhid, d_hid)
        self.text_encoder =TextSeqEncoder(nstep, nemb, nhead,
                                          num_encoder_layers, d_feedforward, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(nhid * nemb + nfield * nemb + nstep * nemb, d_classifier),
            nn.Dropout(dropout),
            nn.Linear(d_classifier, nclass)
        )

    def forward(self, tabular_seq: Tensor, text_seq: Tensor) -> Tensor:
        """
        :param tabular_seq:     bsz*nstep*nfeat*nemb
        :param text_seq:        bsz*nstep*nemb
        :return:
        """



