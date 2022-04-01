import torch
from models.lr import LR
from models.dnn import DNN
from models.robustlog import RobustLog
from models.transsession import TransSession
from models.ensemble import Ensemble
from models.tabularlog import TabularLog

from models.deeplog import DeepLog
from models.loganomaly import LogAnomaly
from models.transwindow import TransWindow
from data_loader import __event_emb_size__

# set default configs for args, always OVERWRITE args
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
    'transformerlog-session': {
        'session_based': True,
        'feature_code': 8,                      # [sequential]
    },
    'ensemble': {
        'session_based': True,                  # [semantic, tabular]
    },
    'tabularlog': {
        'session_based': True,
        'feature_code': 3,                      # [semantic, tabular]
    },
    # window-based
    'deeplog': {
        'only_normal': True,
        'feature_code': 8,                      # [sequential]
        # 'nemb': 1,
    },
    'loganomaly': {
        'only_normal': True,
        'feature_code': 12,                     # [sequential, quantitative]
        # 'nemb': 1,
    },
    'transformerlog-window': {
        'only_normal': True,
        'feature_code': 8,                      # [sequential]
    },
}


def update_default_config(args):
    args_dict = vars(args)
    if args.model in default_config:
        config = default_config[args.model]
        for argument in config.keys():
            if argument in args_dict:
                args_dict[argument] = config[argument]


def create_model(args, logger, vocab_sizes):
    logger.info(f'=> creating model {args.model}')
    # nevent -> vocab last feature
    nevent, nfield, nvocab = vocab_sizes[-1].item(), len(vocab_sizes), sum(vocab_sizes).item()
    # session-based, predict anomaly
    if args.session_based:
        if args.model == 'lr':
            model = LR(2, nevent)
        elif args.model == 'dnn':
            model = DNN(2, nevent, args.mlp_nlayer, args.mlp_nhid, args.dropout)
        elif args.model == 'robustlog':
            model = RobustLog(args.nlayer, args.nhid, bidirectional=True, nemb=args.nemb)
        elif args.model == 'transformerlog-session':
            model = TransSession(nevent, args.nemb, args.nhead, args.nlayer, args.dim_feedforward,
                                args.dropout, args.mlp_nlayer, args.mlp_nhid)
        elif args.model == 'ensemble':
            model = Ensemble(args.feature_code, nfield, nvocab, args.nemb, args.alpha, args.nhid,
                             args.nlayer, args.nquery, args.nhead, args.dropout, args.mlp_nlayer, args.nhid)
        elif args.model == 'tabularlog':
            model = TabularLog(nfield, nvocab, args.nemb, args.alpha, args.nhid,
                             args.nlayer, args.nquery, args.nhead, args.dropout, args.mlp_nlayer, args.nhid)
        else:
            raise NotImplementedError
    # window-based, predict next-event
    else:
        if args.model == 'deeplog':
            model = DeepLog(nevent, args.nlayer, args.nhid, args.nemb)
        elif args.model == 'loganomaly':
            model = LogAnomaly(nevent, args.nlayer, args.nhid, args.nemb)
        elif args.model == 'transformerlog-window':
            model = TransWindow(nevent, args.nemb, args.nhead, args.nlayer, args.dim_feedforward,
                                args.dropout, args.mlp_nlayer, args.mlp_nhid)
        else:
            raise NotImplementedError

    model = torch.nn.DataParallel(model).cuda()
    logger.info(f'{model}\nmodel parameters: {sum([p.data.nelement() for p in model.parameters()])}')
    return model
