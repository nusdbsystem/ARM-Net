from einops import rearrange
import numpy as np

import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor


class RobustLog(torch.nn.Module):
    """
        Model:  RobustLog Model (session-based; [semantic])
        Reference:  https://dl.acm.org/doi/10.1145/3338906.3338931
                    https://github.com/donglee-afar/logdeep (wrong implementation, no attention)
                    https://github.com/LogIntelligence/LogADEmpirical (wrong implementation, padding should be ignored)
    """
    def __init__(self, lstm_nlayer: int, nhid: int, bidirectional: bool = True, nemb: int = 300):
        super().__init__()
        self.num_directions = bidirectional + 1
        self.lstm = nn.LSTM(nemb, nhid, lstm_nlayer, batch_first=True, bidirectional=bidirectional, dropout=0.5)
        self.classifier = nn.Sequential(
            nn.Linear(nhid*self.num_directions, nhid),
            nn.Linear(nhid, 2)
        )
        self.w_omega = nn.Parameter(torch.randn(self.num_directions*nhid, nhid))        # (D*nhid)*nhid
        self.u_omega = nn.Parameter(torch.randn(nhid))                                  # nhid

    def attention(self, lstm_out: FloatTensor, seq_len: LongTensor) -> FloatTensor:
        """
        :param lstm_out:    [bsz*max_len*(D*nhid)]
        :param seq_len:     [bsz]
        :return:            [bsz*(D*nhid)]
        """
        bsz, max_len = lstm_out.size(0), lstm_out.size(1)
        out = rearrange(lstm_out, 'b l h -> (b l) h')                       # (bsz*max_len)*(D*nhid)
        attn_tanh = torch.tanh(torch.einsum('b m, m n -> b n',
                                            out, self.w_omega))             # (bsz*max_len)*nhid
        attn = torch.einsum('b h, h -> b', attn_tanh, self.u_omega)         # (bsz*max_len)
        attn = rearrange(attn, '(b l)->b l', b=bsz)                         # bsz*max_len
        for seq_idx in range(bsz):
            attn[seq_idx][seq_len[seq_idx]:] = -np.inf                      # attn mask
        alpha = torch.softmax(attn, dim=-1)                                 # bsz*max_len
        return torch.einsum('b l h, b l -> b h', lstm_out, alpha)           # bsz*(D*nhid)

    def forward(self, features):
        """
        :param semantic:    [bsz*max_len*nemb], FloatTensor
        :param seq_len:     [bsz], LongTensor
        :return:            [bsz*2], FloatTensor
        """
        semantic = features['semantic']                     # bsz*max_len*nemb
        out, _ = self.lstm(semantic)                        # bsz*max_len*(D*nhid)
        out = self.attention(out, features['seq_len'])      # bsz*(D*nhid)
        return self.classifier(out)                         # bsz*2
