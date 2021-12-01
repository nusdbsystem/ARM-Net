import math

from einops import rearrange
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        :param x:       [*]
        :return:        [*, d_model]
        """
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)                                      # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)     # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))                  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)                            # [max_len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x:   [*, x_len, d_model]
        :return:    [*, x_len, d_model]
        """
        x = x + self.pe[:x.size(-2), :]                                         # [*, x_len, d_model]
        return self.dropout(x)


class TextSeqEncoder(nn.Module):
    def __init__(self, d_model: int, ntokens: int, pad_idx: int,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1
                 ):
        """ Text Sequences Encoder """
        super(TextSeqEncoder, self).__init__()
        assert dim_feedforward % nhead == 0, \
            f"dim_ff {dim_feedforward} must be divisible by nhead {nhead}"
        self.pad_idx = pad_idx

        self.token_embedding = Embeddings(d_model=d_model, vocab=ntokens)
        self.pos_embedding = PositionalEncoding(d_model=d_model, dropout=dropout)

        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, text: Tensor) -> Tensor:
        """
        :param text:    bsz*nstep*seq_len
        :return:        bsz*nstep*d_model
        """
        bsz, nstep, seq_len = text.size()
        text = rearrange(text, 'b t l -> (b t) l')                      # (bsz*nstep)*seq_len

        padding_mask = (text == self.pad_idx)                           # (bsz*nstep)*seq_len
        text_emb = self.token_embedding(text)                           # (bsz*nstep)*seq_len*d_model

        encoded_text = self.encoder(text_emb,
                                    src_key_padding_mask=padding_mask)  # (bsz*nstep)*seq_len*d_model
        encoded_text = encoded_text.view(bsz, nstep, seq_len, -1)       # bsz*nstep*seq_len*d_model

        # only return the context encoded in the prepended EOS token
        return encoded_text[:, :, 0]                                    #  bsz*nstep*d_model
