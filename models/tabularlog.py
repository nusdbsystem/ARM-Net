from einops import rearrange, repeat, reduce
import torch
from torch import Tensor
import torch.nn as nn
from models.layers import MLP
from data_loader import decode_feature_code, __event_emb_size__
from models.armnet import ARMModule
from models.layers import Attention, PositionalEncoding


class TabularSeqEncoder(nn.Module):
    def __init__(self, nfield: int, nfeat: int, nemb: int, alpha: float, nhid: int, nquery: int, dropout: float):
        """ Time-series Tabular Data Encoder """
        super(TabularSeqEncoder, self).__init__()
        self.feature_embedding = nn.Embedding(nfeat, nemb)
        self.arm = ARMModule(nfield, nemb, nemb, alpha, nhid)
        # cross-attn for querying all logs across time
        # TODO: ablation study - positional encoding
        self.pos_enc = PositionalEncoding(nemb, dropout=dropout)
        self.query = nn.Parameter(torch.randn(nquery, nemb))
        self.seq_query = Attention(query_dim=nemb, context_dim=nemb, heads=8, dim_head=nemb)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x:       [bsz, nstep, nfield], LongTensor
        :return:        [bsz, nquery, nemb], FloatTensor
        """
        bsz = x.size(0)
        # arm
        x = self.feature_embedding(x)                                   # bsz*nstep*nfield*nemb
        x = rearrange(x, 'b t f e -> (b t) f e')                        # (bsz*nstep)*nfield*nemb
        x = self.arm(x)                                                 # (bsz*nstep)*nhid*nemb
        x = reduce(x, '(b t) h e -> b t e', 'mean', b=bsz)              # bsz*nstep*nemb
        # positional encoding
        x = self.pos_enc(x)                                             # bsz*nstep*nemb
        # cross-attn
        query = repeat(self.query, 'q e -> b q e', b=bsz)               # bsz*nquery*nemb
        x = self.seq_query(query, context=x)                            # bsz*nquery*nemb
        return x


class TabularLog(nn.Module):
    """ Log Sequence Encoder """
    def __init__(self, nevent: int, feature_code: int, nfield: int, nfeat: int, nemb: int,
                 alpha: float, nhid: int, nquery: int, lstm_nlayer: int = 2, dropout: float = 0.0,
                 mlp_nlayer: int = 2, mlp_nhid: int = 256):
        """
        :param nfield:              Number of fields of tabular data
        :param nfeat:               Total number of tabular features
        :param nemb:                Tabular feature embedding size
        :param alpha:               Sparsity of ARM-Module
        :param nhid:                Number of Cross Features in ARM-Module/hidden features
        :param nquery:              Number of query vectors in Tabular output
        :param lstm_nlayer:         Number of LSTM layers for sequential, semantic modeling
        :param dropout:             Dropout rate for Log Seq Encoder and predictor
        :param mlp_nlayer:          Number of MLP layers for classifier
        :param mlp_nhid:            Number of MLP hidden units for classifier
        :param nevent:              Number of prediction output, i.e., eventID to be predict
        """
        super(TabularLog, self).__init__()
        classifier_ndim = 0
        self.use_sequential, self.use_quantitative, self.use_semantic, self.use_tabular = \
            decode_feature_code(feature_code)
        if self.use_sequential:
            self.sequential_embedding = nn.Embedding(nevent, nemb)
            self.sequential_lstm = nn.LSTM(nemb, nhid, lstm_nlayer,
                                           batch_first=True, bidirectional=False, dropout=dropout)
            classifier_ndim += nhid
        if self.use_quantitative:
            self.quantitative_mlp = MLP(nevent, mlp_nlayer, mlp_nhid, dropout, noutput=nhid)
            classifier_ndim += nhid
        if self.use_semantic:
            self.semantic_lstm = nn.LSTM(__event_emb_size__, nhid, lstm_nlayer,
                                         batch_first=True, bidirectional=False, dropout=dropout)
            classifier_ndim += nhid
        if self.use_tabular:
            self.tabular_encoder = TabularSeqEncoder(nfield, nfeat, nemb, alpha, nhid, nquery, dropout)
            # TODO: introduce an MLP for reduction
            classifier_ndim += nemb
        self.classifier = MLP(classifier_ndim, nlayers=mlp_nlayer, nhid=mlp_nhid, dropout=dropout, noutput=nevent)

    def forward(self, features) -> Tensor:
        """
        :param features:    [sequential, quantitative, semantic, tabular], each of [nwindow, *], LongTensor
        :return:            [nwindow, nevent], FloatTensor, prediction of next eventID
        """
        deep_features = []
        if self.use_sequential:
            sequential = features['sequential']                     # nwindow*nstep
            sequential = self.sequential_embedding(sequential)      # nwindow*nstep*nemb
            sequential, _ = self.sequential_lstm(sequential)        # nwindow*nstep*nhid
            deep_features.append(sequential[:, -1])                 # nwindow*nhid
        if self.use_quantitative:
            quantitative = features['quantitative'].float()         # nwindow*nevent
            quantitative = self.quantitative_mlp(quantitative)      # nwindow*nhid
            deep_features.append(quantitative)                      # nwindow*nhid
        if self.use_semantic:
            semantic = features['semantic']                         # nwindow*nstep*E_nemb
            semantic, _ = self.semantic_lstm(semantic)              # nwindow*nstep*nhid
            deep_features.append(semantic[:, -1])                   # nwindow*nhid
        if self.use_tabular:
            tabular = features['tabular']                           # nwindow*nstep*nfield
            tabular = self.tabular_encoder(tabular)                 # nwindow*nquery*nemb
            tabular = reduce(tabular, 'b q e -> b e', 'mean')       # nwindow*nemb
            deep_features.append(tabular)                           # nwindow*nemb

        deep_features = torch.cat(deep_features, dim=1)             # nwindow*classifier_ndim
        return self.classifier(deep_features)                       # nwindow*nevent
