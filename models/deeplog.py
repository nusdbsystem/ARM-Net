import torch
import torch.nn as nn


class DeepLog(torch.nn.Module):
    """
        Model:  DeepLog Model (window-based; sequentials)
        Reference:  https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf
                    https://github.com/donglee-afar/logdeep
    """
    def __init__(self, nevent: int, lstm_nlayer: int, nhid: int, nemb: int):
        super().__init__()
        self.embedding =nn.Embedding(nevent, nemb)
        self.lstm = nn.LSTM(nemb, nhid, lstm_nlayer, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(nhid, nevent)

    def forward(self, features):
        """
        :param x:   [nwindow*nstep], FloatTensor
        :return:    [nwindow*nevent], FloatTensor
        """
        x = features['sequential']                                  # nwindow*nstep
        x = self.embedding(x)                                       # nwindow*nstep*nemb
        out, _ = self.lstm(x)                                       # nwindow*nstep*nhid
        return self.classifier(out[:, -1])                          # nwindow*nevent
