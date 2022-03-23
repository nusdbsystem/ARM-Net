import torch
import torch.nn as nn
from models.layers import PositionalEncoding, MLP
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class TransSession(torch.nn.Module):
    """
        Model:  LogTransformer Model (session-based; [sequential])
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
        self.event_embedding = nn.Embedding(nevent+1, nemb)
        self.pos_enc = PositionalEncoding(nemb, dropout=dropout)
        # transformer
        encoder_layer = TransformerEncoderLayer(d_model=nemb, nhead=nhead, dim_feedforward=dim_feedforward,
                                                dropout=dropout, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = MLP(nemb, nlayer=mlp_nlayer, nhid=mlp_nhid, dropout=dropout, noutput=nevent)

    def forward(self, features):
        """
        :param sequential:  [bsz*max_len], LongTensor
        :return:            [bsz*2], FloatTensor
        """
        sequential = features['sequential']                                 # bsz*max_len
        sequential = sequential + 1                                         # bsz*max_len
        seq_len = features['seq_len']                                       # bsz
        bsz, max_len = sequential.size()
        sequential = self.pos_enc(self.event_embedding(sequential))         # bsz*max_len*nemb
        # src_key_padding_mask for TransformerEncoder
        mask = torch.zeros((bsz, max_len),
                           dtype=torch.bool, device=sequential.device)      # bsz*max_len
        for seq_idx in range(bsz):
            mask[seq_idx, seq_len[seq_idx]:] = True
        sequential = self.transformer(sequential,
                                      src_key_padding_mask=mask)            # bsz*max_len*nemb
        # mean aggregation, ignore padding tokens
        aggregation = torch.zeros((bsz, sequential.size(2)),
                            dtype=torch.float, device=sequential.device)    # bsz*nemb
        for seq_idx in range(bsz):
            aggregation[seq_idx] = torch.mean(
                sequential[seq_idx, :seq_len[seq_idx]], dim=0)
        return self.classifier(aggregation)                                 # bsz*2
