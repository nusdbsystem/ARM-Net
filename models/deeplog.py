import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class DeepLog(torch.nn.Module):
    """
        Model:  DeepLog Model (window-based; [sequential])
        Reference:  https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf
                    https://github.com/donglee-afar/logdeep
    """
    def __init__(self, nevent: int, lstm_nlayer: int, nhid: int, nemb: int = 1):
        super().__init__()
        self.embedding = Rearrange('b t -> b t 1') if nemb == 1 else nn.Embedding(nevent, nemb)
        self.lstm = nn.LSTM(nemb, nhid, lstm_nlayer, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(nhid, nevent)

    def forward(self, features):
        """
        :param sequential:  [nwindow*nstep], LongTensor
        :return:            [nwindow*nevent], FloatTensor
        """
        sequential = features['sequential']                 # nwindow*nstep
        sequential = self.embedding(sequential).float()     # nwindow*nstep*nemb
        out, _ = self.lstm(sequential)                      # nwindow*nstep*nhid
        return self.classifier(out[:, -1])                  # nwindow*nevent
