import torch
from torch import nn


class robustlog(nn.Module):
    def __init__(self, ninput: int = 300, nhid: int = 128, nlayer: int = 2, nclass: int = 2):
        super(robustlog, self).__init__()
        self.nhid = nhid
        self.nlayer = nlayer
        self.lstm = nn.LSTM(ninput, nhid, nlayer, batch_first=True)
        self.fc = nn.Linear(nhid, nclass)

    def forward(self, features, device):
        input0 = features[0]                                    # bsz*nseq*nemb
        h0 = torch.zeros(self.nlayer, input0.size(0),
                         self.nhid).to(device)
        c0 = torch.zeros(self.nlayer, input0.size(0),
                         self.nhid).to(device)
        out, _ = self.lstm(input0, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out