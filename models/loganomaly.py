
from einops import rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn


class LogAnomaly(torch.nn.Module):
    """
        Model:  LogAnomaly Model (window-based; [sequential, quantative], original implementation nemb=1)
        Reference:  https://www.ijcai.org/proceedings/2019/0658.pdf
                    https://github.com/donglee-afar/logdeep (wrong implementation, correct one refers to the link below)
        https://github.com/nusdbsystem/ARM-Net/blob/53975c8d1b5378cc997a40f0dfd136d1863a3b8f/models/loganomaly.py
    """
    def __init__(self, nevent: int, lstm_nlayer: int, nhid: int, nemb: int = 1):
        super().__init__()
        self.embedding = Rearrange('b t -> b t 1') if nemb == 1 else nn.Embedding(nevent, nemb)
        self.sequential_lstm = nn.LSTM(nemb, nhid, lstm_nlayer, batch_first=True, bidirectional=False)
        self.quantitative_lstm = nn.LSTM(1, nhid, lstm_nlayer, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(nhid*2, nevent)

    def forward(self, features):
        """
        :param sequential:      [nwindow*nstep], LongTensor
        :param quantitative:    [nwindow*nstep], LongTensor
        :return:                [nwindow*nevent], FloatTensor
        """
        sequential = features['sequential']                         # nwindow*nstep
        sequential = self.embedding(sequential).float()             # nwindow*nstep*nemb
        sequential_out, _ = self.sequential_lstm(sequential)        # nwindow*nstep*nhid

        quantitative = features['quantitative'].float()             # nwindow*nstep
        quantitative = rearrange(quantitative, 'b t -> b t 1')      # nwindow*nstep*1
        quantitative_out, _ = self.quantitative_lstm(quantitative)  # nwindow*nstep*nhid

        out = torch.cat([sequential_out[:, -1],
                         quantitative_out[:, -1]], dim=1)           # nwindow*2nhid
        return self.classifier(out)                                 # nwindow*nevent