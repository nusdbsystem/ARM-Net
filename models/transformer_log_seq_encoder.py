from einops import reduce

import torch
from torch import Tensor
import torch.nn as nn

from models.layers import MLP
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class TransformerLogSeqEncoder(nn.Module):
    """ Log Sequence Encoder using Transformer """
    def __init__(self, nstep: int, nfeat: int, nemb: int,
                 nhead: int = 8, num_layers: int = 6, dim_feedforward: int = 2048,
                 dropout: float = 0.0, predictor_nlayer: int = 2, d_predictor: int = 256,
                 noutput: int = 2):
        """
        :param nstep:               Number of time steps
        :param nfield:              Number of fields of tabular data
        :param nfeat:               Total number of tabular features
        :param nemb:                Tabular feature embedding size
        :param nhead:               Number of attention heads for Log Seq Encoder
        :param num_layers:          Number of layers for Log Seq Encoder
        :param dim_feedforward:     FFN dimension for Log Seq Encoder
        :param dropout:             Dropout rate for Log Seq Encoder and predictor
        :param predictor_nlayer:    FFN layers for predictor
        :param d_predictor:         FFN dimension for predictor
        :param noutput:             Number of prediction output for predictor, e.g., nclass
        """
        super(TransformerLogSeqEncoder, self).__init__()
        self.feature_embedding = nn.Embedding(nfeat, nemb)
        # positional embedding
        # TODO: nstep -> max_len (or using positional encoding) for supporting flexible log seq len
        self.pos_embedding = nn.Parameter(torch.randn(1, nstep, nemb))

        encoder_layer = TransformerEncoderLayer(
            d_model=nemb, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.log_seq_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        # TODO: using class token for prediction (-> more flexible log seq len during inference)
        # current: mean
        self.predictor = MLP(nemb, nlayers=predictor_nlayer, nhid=d_predictor,
                             dropout=dropout, noutput=noutput)

    def forward(self, tabular_seq: Tensor) -> Tensor:
        """
        :param tabular_seq:     [bsz, nstep, nfield], LongTensor
        :return:                [bsz, noutput], FloatTensor
        """
        # use only eventID field for eventID prediction
        log_seq = tabular_seq[:, :, -1]                                 # bsz*nstep
        log_seq = self.feature_embedding(log_seq)+self.pos_embedding    # bsz*nstep*nemb

        log_seq = self.log_seq_encoder(log_seq)                         # bsz*nstep*nemb
        log_seq = reduce(log_seq, 'b t e -> b e', 'mean')               # bsz*nemb
        y = self.predictor(log_seq)                                     # bsz*noutput
        return y
