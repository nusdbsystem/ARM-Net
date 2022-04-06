from einops import rearrange, repeat, reduce
import torch
from torch import Tensor
import torch.nn as nn
from data_loader import __event_emb_size__
from models.armnet import ARMModule
from models.layers import MLP, Attention, PositionalEncoding


class TabularLog(nn.Module):
    """ Log Sequence Encoder """
    def __init__(self, nfield: int, nfeat: int, nemb: int,
                 alpha: float, nhid: int, lstm_nlayer: int, nquery: int, nhead: int = 8,
                 dropout: float = 0.0, mlp_nlayer: int = 2, mlp_nhid: int = 256, noutput: int = 2):
        """
        :param nfield:          Number of fields of tabular data
        :param nfeat:           Total number of tabular features
        :param nemb:            Tabular feature embedding size
        :param alpha:           Sparsity for ARM-Module
        :param nhid:            Number of Cross Features in ARM-Module
        :param lstm_nlayer:     Number of LSTM layers for semantic modeling
        :param nquery:          Number of query vectors in Cross Attention
        :param nhead:           Number of attention heads for Cross Attention
        :param dropout:         Dropout rate for the whole model
        :param mlp_nlayer:      Number of MLP layers for classifier
        :param mlp_nhid:        Number of MLP hidden units for classifier
        :param noutput:         Number of prediction output (nclass)
        """
        super(TabularLog, self).__init__()
        # 1. tabular - arm-module
        self.tabular_embedding = nn.Embedding(nfeat+1, nemb)
        self.arm = ARMModule(nfield-1, nemb, nemb, alpha, nhid)
        self.positional_encoding = PositionalEncoding(nemb, dropout=dropout)
        # 2. semantic - lstm
        self.semantic_lstm = nn.LSTM(__event_emb_size__, nemb, lstm_nlayer, batch_first=True, dropout=dropout)
        # 3. sequence modeling - cross-attention
        self.query = nn.Parameter(torch.randn(nquery, nemb))
        self.cross_attn = Attention(query_dim=nemb, context_dim=nemb+nemb, heads=nhead, dim_head=nemb)
        self.classifier = MLP(nemb, nlayer=mlp_nlayer, nhid=mlp_nhid, dropout=dropout, noutput=noutput)

    def forward(self, features) -> Tensor:
        """
        :param features:    [semantic, tabular], of [bsz, max_len, SEM_EMB/nfield], FloatTensor/LongTensor
        :return:            [bsz, noutput], FloatTensor, anomaly prediction
        """
        seq_len, semantic, tabular = features['seq_len'], features['semantic'], features['tabular']
        bsz, max_len = tabular.size()[:2]
        # 1. tabular
        tabular = tabular + 1                                               # bsz*max_len*nfield, padding -1 to 0
        tabular = self.tabular_embedding(tabular)                           # bsz*max_len*nfield*nemb
        tabular = rearrange(tabular, 'b t f e -> (b t) f e')                # (bsz*max_len)*nfield*nemb
        tabular = tabular[:, :-1]                                           # (bsz*max_len)*(nfield-1)*nemb
        tabular = self.arm(tabular)                                         # (bsz*max_len)*nhid*nemb
        tabular = reduce(tabular, '(b t) h e -> b t e', 'mean', b=bsz)      # bsz*max_len*nemb
        tabular = self.positional_encoding(tabular)                         # bsz*max_len*nemb
        # 2. semantic
        semantic, _ = self.semantic_lstm(semantic)                          # bsz*max_len*nemb
        tabular = torch.cat([tabular, semantic], dim=2)                     # bsz*max_len*(nemb+nemb)
        # 3. cross-attn
        mask = torch.zeros((bsz, max_len), dtype=torch.bool,
                           device=tabular.device)                           # bsz*max_len
        for seq_idx in range(bsz):
            mask[seq_idx, seq_len[seq_idx]:] = True
        query = repeat(self.query, 'q e -> b q e', b=bsz)                   # bsz*nquery*nemb
        seq_feature = self.cross_attn(query, context=tabular, mask=mask)    # bsz*nquery*nemb
        seq_feature = reduce(seq_feature, 'b q e -> b e', 'mean')           # bsz*nemb
        if not self.training:
            self.representation = seq_feature.detach()                      # for representation visualization
        return self.classifier(seq_feature)                                 # bsz*noutput
