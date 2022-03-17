import torch
from models.lr import LR
from models.dnn import DNN
from models.robustlog import RobustLog

from models.deeplog import DeepLog
from models.loganomaly import LogAnomaly
from models.tabularlog import TabularLog
from data_loader import __event_emb_size__

# set default configs for args, will always overwrite args
default_config = {
    # session-based
    'lr': {
        'session_based': True,
        'feature_code': 4,                      # [quantitative]
    },
    'dnn': {
        'session_based': True,
        'feature_code': 4,                      # [quantitative]
        'mlp_nlayer': 2,
        'mlp_nhid': 256,
    },
    'robustlog': {
        'session_based': True,
        'feature_code': 2,                      # [semantic]
        'nemb': __event_emb_size__,
    },
    # window-based
    'deeplog': {
        'only_normal': True,
        'feature_code': 8,                      # [sequential]
    },
    'loganomaly': {
        'only_normal': True,
        'feature_code': 12,                     # [sequential, quantitative]
    },
    'tabularlog': {
        'only_normal': True
    }

}


def update_default_config(args):
    args_dict = vars(args)
    if args.model in default_config:
        config = default_config[args.model]
        for argument in config.keys():
            if argument in args_dict:
                args_dict[argument] = config[argument]


def create_model(args, logger, vocab_sizes):
    if logger: logger.info(f'=> creating model {args.model}')
    # nevent -> vocab last feature
    nevent, nfield, nvocab = vocab_sizes[-1].item(), len(vocab_sizes), sum(vocab_sizes).item()
    # session-based, supervised training
    if args.session_based:
        if args.model == 'lr':
            model = LR(2, nevent)
        elif args.model == 'dnn':
            model = DNN(2, nevent, args.mlp_nlayer, args.mlp_nhid, args.dropout)
        elif args.model == 'robustlog':
            model = RobustLog(args.nlayer, args.nhid, bidirectional=True, nemb=args.nemb)
        else:
            raise NotImplementedError
    # window-based, unsupervised training
    else:
        if args.model == 'deeplog':
            model = DeepLog(nevent, args.nlayer, args.nhid, args.nemb)
        elif args.model == 'loganomaly':
            model = LogAnomaly(nevent, args.nlayer, args.nhid, args.nemb)
        elif args.model == 'tabularlog':
            model = TabularLog(nevent, args.feature_code, nfield, nvocab, args.nemb, args.alpha, args.nhid,
                               args.nquery, args.nlayer, args.dropout, args.mlp_nlayer, args.mlp_nhid)
        else:
            raise NotImplementedError

    model = torch.nn.DataParallel(model).cuda()
    if logger: logger.info(f'{model}\nmodel parameters: {sum([p.data.nelement() for p in model.parameters()])}')
    return model
