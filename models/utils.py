import torch
from models.lr import LR
from models.dnn import DNN

from models.deeplog import DeepLog
from models.logSeqEnc import LogSeqEncoder
from models.transformerLogSeqEnc import TransformerLogSeqEncoder

# set default configs for args, will always overwrite args
default_config = {
    # session-based
    'lr': {
        'feature_code': 4,                      # quantitative
    },
    'dnn': {
        'feature_code': 4,                      # quantitative
        'mlp_nlayer': 2,
        'mlp_nhid': 256,
    },
    # window-based
    'deeplog': {
        'only_normal': True,                    # sequential
        'feature_code': 8,
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
    nevent, nvocab = vocab_sizes[-1].item(), sum(vocab_sizes).item()
    # session-based, supervised training
    if args.session_based:
        if args.model == 'lr':
            model = LR(2, nevent)
        elif args.model == 'dnn':
            model = DNN(2, nevent, args.mlp_nlayer, args.mlp_nhid, args.dropout)
        elif args.model == 'robustlod':
            # [semantics]; bsz*nseq*nemb; nseq pad to 50
            raise NotImplementedError
        else:
            raise ValueError(f'unknown model {args.model}')
    # window-based, unsupervised training
    else:
        if args.model == 'deeplog':
            model = DeepLog(nevent, args.nlayer, args.nhid, args.nemb)
        elif args.model == 'loganomaly':
            # [sequentials, quantatives]; bsz*nstep*1; bsz*nvent*1
            raise NotImplementedError
        elif args.model == 'armnet':
            model = LogSeqEncoder(args.nstep, len(vocab_sizes), nvocab, args.nemb, args.alpha, args.nhid,
                                  args.nquery, args.nhead, args.nlayer, args.dim_feedforward, args.dropout,
                                  args.mlp_nlayer, args.mlp_nhid, nevent)
        elif args.model == 'transformer':
            model = TransformerLogSeqEncoder(args.nstep, nvocab, args.nemb, args.nhead, args.nlayer,
                                 args.dim_feedforward, args.dropout, args.mlp_nlayer, args.mlp_nhid, nevent)
        else:
            raise ValueError(f'unknown model {args.model}')

    model = torch.nn.DataParallel(model).cuda()
    if logger: logger.info(f'{model}\nmodel parameters: {sum([p.data.nelement() for p in model.parameters()])}')
    return model
