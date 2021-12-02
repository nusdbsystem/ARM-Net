import torch
from einops import rearrange, reduce

from torch import Tensor
import torch.nn as nn

from models.layers import MLP, TCN
from models.tabular_seq_encoder import TabularSeqEncoder
from models.text_seq_encoder import  TextSeqEncoder


class LogSeqEncoder(nn.Module):
    """ Tabular & Text Sequences Data Encoder """
    def __init__(self, nstep: int, nfield: int, nfeat: int,
                 nemb: int, alpha: float, nhid: int, d_hid: int,
                 d_model: int, ntokens: int, pad_idx: int,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 tcn_layers: int = 3,
                 l_predictor: int = 2,
                 d_predictor: int = 512,
                 noutput: int = 1
                 ):
        """
        :param nstep:               Number of time steps
        :param nfield:              Number of fields of tabular data
        :param nfeat:               Total number of tabular features
        :param nemb:                Tabular feature embedding size
        :param alpha:               Sparsity of ARM-Module
        :param nhid:                Number of Cross Features in ARM-Module
        :param d_hid:               Inner Query/Key dimension in ARM-Module
        :param d_model:             Text Token embedding size
        :param ntokens:             Total number of text tokens
        :param pad_idx:             Padding idx for Text
        :param nhead:               Number of attention head for Text Encoder
        :param num_layers:          Number of layers for Text Encoder
        :param dim_feedforward:     FFN dimension for Text Encoder
        :param dropout:             Dropout rate for Text Encoder and predictor
        :param tcn_layers:          Number of TCN layers for Tabular & Text
        :param l_predictor:         FFN layers for predictor
        :param d_predictor:         FFN dimension for predictor
        :param noutput:              Number of prediction output for predictor
        """
        super(LogSeqEncoder, self).__init__()
        self.tabular_encoder = TabularSeqEncoder(
            nstep, nfield, nfeat, nemb, alpha, nhid, d_hid)
        self.text_encoder = TextSeqEncoder(
            d_model, ntokens, pad_idx, nhead, num_layers, dim_feedforward, dropout)

        self.tabular_tcn = TCN(nemb, nlayer=tcn_layers)
        self.text_tcn = TCN(d_model, nlayer=tcn_layers)

        self.predictor = MLP(nemb + d_model, nlayers=l_predictor,
                             nhid=d_predictor, dropout=dropout, noutput=noutput)


    def forward(self, tabular_seq: Tensor, text_seq: Tensor) -> Tensor:
        """
        :param tabular_seq:     [bsz, nstep, nfield], LongTensor
        :param text_seq:        [bsz, nstep, seq_len], LongTensor
        :return:
        """
        tabular = self.tabular_encoder(tabular_seq)                 # bsz*nstep*nemb
        tabular = rearrange(tabular, 'b t e -> b e t')              # bsz*nemb*nstep
        tabular = self.tabular_tcn(tabular)                         # bsz*nemb*nstep
        tabular = reduce(tabular, 'b e t -> b e', 'mean')           # bsz*nemb

        text = self.text_encoder(text_seq)                          # bsz*nstep*d_model
        text = rearrange(text, 'b t d -> b d t')                    # bsz*d_model*nstep
        text = self.text_tcn(text)                                  # bsz*d_model*nstep
        text = reduce(text, 'b d t -> b d', 'mean')                 # bsz*d_model

        context = torch.cat([tabular, text], dim=1)                 # bsz*(nemb+d_model)
        y = self.predictor(context)                                 # bsz*noutput
        return y
