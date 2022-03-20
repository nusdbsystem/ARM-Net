from einops import reduce
import torch
import torch.nn as nn
from models.layers import PositionalEncoding, MLP
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class TransWindow(torch.nn.Module):
    """
        Model:  LogTransformer Model (window-based; [sequential])
        Reference:  https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """
    def __init__(self, nevent: int, nemb: int, nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.0, mlp_nlayer: int = 2, mlp_nhid: int = 256):
        """
        :param nevent:              Total number of events
        :param nemb:                event embedding size
        :param nhead:               Number of attention heads
        :param num_layers:          Number of transformer layers
        :param dim_feedforward:     FFN dimension of transformer
        :param dropout:             dropout rate for transformer and MLP
        :param mlp_nlayer:          FFN layers for next event prediction
        :param mlp_nhid:            FFN dimention for next event prediction
        """
        super().__init__()
        # event embedding & positional encoding
        # TODO: class embedding for aggregating seq info
        self.event_embedding = nn.Embedding(nevent, nemb)
        self.positional_encoding = PositionalEncoding(nemb, dropout=dropout)
        # transformer
        encoder_layer = TransformerEncoderLayer(d_model=nemb, nhead=nhead, dim_feedforward=dim_feedforward,
                                                dropout=dropout, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = MLP(nemb, nlayers=mlp_nlayer, nhid=mlp_nhid, dropout=dropout, noutput=nevent)

    def forward(self, features):
        """
        :param sequential:  [nwindow*nstep], LongTensor
        :return:            [nwindow*nevent], FloatTensor
        """
        sequential = features['sequential']                             # nwindow*nstep
        sequential = self.positional_encoding(
            self.event_embedding(sequential))                           # nwindow*nstep*nemb
        sequential = self.transformer(sequential)                       # nwindow*nstep*nemb
        sequential = reduce(sequential, 'b t e -> b e', 'mean')         # nwindow*nemb
        return self.classifier(sequential)                              # nwindow*nevent
