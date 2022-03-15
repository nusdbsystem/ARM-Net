from collections import Counter
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
from torch import Tensor, LongTensor


def sequential_to_quantitative_seq(sequential: LongTensor, nevent: int) -> LongTensor:
    """
    :param sequential:  [nwindow*nstep], LongTensor
    :return:            [nwindow*nstep*nevent], LongTensor
    """
    def sliding_window_to_squantitative(sliding_window: LongTensor) -> Tensor:
        """
        :param sliding_window:  nwindow*(t+1)
        :return:                nwindow*nevent
        """
        quantitative = torch.zeros((sliding_window.size(0), nevent), dtype=torch.int64)
        for window_idx in range(sliding_window.size(0)):
            counter = Counter(sliding_window[window_idx].tolist())
            for eventID in counter:
                quantitative[window_idx][eventID] = counter[eventID]
        return quantitative

    nwindow, nstep = sequential.size()
    quantitative = torch.zeros((nwindow, nstep, nevent), dtype=torch.int64)     # nwindow*nstep*nevent
    for t in range(nstep):
        sliding_window = sequential[:, :t+1]                                    # nwindow*(t+1)
        quantitative[:, t] = sliding_window_to_squantitative(sliding_window)    # nwindow*nevent
    return quantitative.to(sequential.device)


class LogAnomaly(torch.nn.Module):
    """
        Model:  LogAnomaly Model (window-based; [sequential, quantative])
        Reference:  https://www.ijcai.org/proceedings/2019/0658.pdf
    """
    def __init__(self, nevent: int, lstm_nlayer: int, nhid: int, nemb: int = 1):
        super().__init__()
        self.nevent = nevent
        self.embedding = Rearrange('b t -> b t 1') if nemb == 1 else nn.Embedding(nevent, nemb)
        self.sequential_lstm = nn.LSTM(nemb, nhid, lstm_nlayer, batch_first=True, bidirectional=False)
        self.quantitative_lstm = nn.LSTM(nevent, nhid, lstm_nlayer, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(nhid*2, nevent)

    def forward(self, features):
        """
        :param sequential:      [nwindow*nstep], LongTensor
        :return:                [nwindow*nevent], FloatTensor
        """
        sequential = features['sequential']                             # nwindow*nstep
        sequential = self.embedding(sequential).float()                 # nwindow*nstep*nemb
        sequential_out, _ = self.sequential_lstm(sequential)            # nwindow*nstep*nhid

        quantitative = sequential_to_quantitative_seq(
            features['sequential'].cpu(), self.nevent).float().cuda()   # nwindow*nstep*nevent
        quantitative_out, _ = self.quantitative_lstm(quantitative)      # nwindow*nstep*nhid

        out = torch.cat([sequential_out[:, -1],
                         quantitative_out[:, -1]], dim=1)               # nwindow*2nhid
        return self.classifier(out)                                     # nwindow*nevent
