from einops import rearrange

import torch
from torch import Tensor
import torch.nn as nn

from models.layers import MLP
from models.tabular_seq_encoder import TabularSeqEncoder
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class LogSeqEncoder(nn.Module):
    """ Log Sequence Encoder """
    def __init__(self, nstep: int, nfield: int, nfeat: int,
                 nemb: int, alpha: float, nhid: int, nquery: int,
                 nhead: int = 8, num_layers: int = 6, dim_feedforward: int = 2048,
                 dropout: float = 0.0, predictor_nlayer: int = 2, d_predictor: int = 256,
                 noutput: int = 2):
        """
        :param nstep:               Number of time steps
        :param nfield:              Number of fields of tabular data
        :param nfeat:               Total number of tabular features
        :param nemb:                Tabular feature embedding size
        :param alpha:               Sparsity of ARM-Module
        :param nhid:                Number of Cross Features in ARM-Module
        :param nquery:              Number of query vectors in Tabular output
        :param nhead:               Number of attention heads for Log Seq Encoder
        :param num_layers:          Number of layers for Log Seq Encoder
        :param dim_feedforward:     FFN dimension for Log Seq Encoder
        :param dropout:             Dropout rate for Log Seq Encoder and predictor
        :param predictor_nlayer:    FFN layers for predictor
        :param d_predictor:         FFN dimension for predictor
        :param noutput:             Number of prediction output for predictor, e.g., nclass
        """
        super(LogSeqEncoder, self).__init__()
        self.tabular_encoder = TabularSeqEncoder(nstep, nfield, nfeat, nemb, alpha, nhid, nquery)
        d_model = nquery * nemb
        # positional embedding
        # TODO: nstep -> max_len (or using positional encoding) for supporting flexible log seq len
        self.pos_embedding = nn.Parameter(torch.randn(1, nstep, d_model))
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.log_seq_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        # TODO: using class token for prediction (-> more flexible log seq len during inference)
        self.predictor = MLP(nstep*d_model, nlayers=predictor_nlayer, nhid=d_predictor,
                             dropout=dropout, noutput=noutput)

    def forward(self, tabular_seq: Tensor) -> Tensor:
        """
        :param tabular_seq:     [bsz, nstep, nfield], LongTensor
        :return:                [bsz, noutput], FloatTensor
        """
        log_seq = self.tabular_encoder(tabular_seq)                 # bsz*nstep*d_model
        log_seq = log_seq + self.pos_embedding                      # bsz*nstep*d_model
        log_seq = self.log_seq_encoder(log_seq)                     # bsz*nstep*d_model
        log_seq = rearrange(log_seq, 'b t d -> b (t d)')            # bsz*(nstep*d_model)
        y = self.predictor(log_seq)                                 # bsz*noutput
        return y
