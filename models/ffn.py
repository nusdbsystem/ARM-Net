import math
import torch.nn as nn

FFN_CONFIGS = {
    'lr': [0.01, 0.1, 1.0],
    'unit': [512, 256, 128],
    'layer': [2, 4, 8, 16],
    'dropout': [0.0, 0.01, 0.05, 0.1],
    'activation': ['selu', 'relu']
}

ACTIVATIONS = {
    'selu': nn.SELU,
    'relu': nn.ReLU
}


def init_tensor(tensor, nonlinearity, mode='fan_in'):
    if nonlinearity == 'selu':
        nonlinearity = 'linear'
    fan = nn.init._calculate_correct_fan(tensor, mode)
    gain = nn.init.calculate_gain(nonlinearity)
    std = gain / math.sqrt(fan)
    return tensor.data.normal_(0, std)

class Layer(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', dropout=0.):
        super(Layer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        init_tensor(self.linear.weight, activation)
        if activation=='selu': self.dropout = nn.AlphaDropout(p=dropout)
        else: self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.activation = ACTIVATIONS[activation]()

    def forward(self, x):
        return self.activation(self.linear(self.dropout(x)))

class FFN(nn.Module):
    def __init__(self, in_features, n_classes, config, softmax=True):
        super(FFN, self).__init__()
        self.in_features = in_features
        self.config = config
        self.layers = nn.ModuleList([Layer(in_features, config['unit'],
                    config['activation'], config['dropout'])])      # NOTE: 1st layer

        for _ in range(config['layer']-1):
            self.layers.append(Layer(config['unit'], config['unit'],
                    config['activation'], config['dropout']))       # NOTE: config['dropout']
        self.layers.append(nn.Linear(config['unit'], n_classes))
        if softmax: self.layers.append(nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = x.view(-1, self.in_features)
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x