import torch
from models.logSeqEnc import LogSeqEncoder
from models.transformerLogSeqEnc import TransformerLogSeqEncoder
from models.lr import LRModel
from models.dnn import DNNModel

def create_model(args, logger, vocab_sizes):
    if logger: logger.info(f'=> creating model {args.model}')
    # nevent -> vocab last feature
    args.nevent = vocab_sizes[-1]
    # session-based, supervised training
    if args.session_based:
        if args.model == 'lr':
            model = LRModel(2, args.nevent)
        elif args.model == 'dnn':
            model = DNNModel(2, args.nevent, args.mlp_nlayer, args.mlp_nhid, args.dropout)
        elif args.model == 'robustlod':
            # [semantics]; bsz*nseq*nemb; nseq pad to 50
            raise NotImplementedError
        else:
            raise ValueError(f'unknown model {args.model}')
    # window-based, unsupervised training
    else:
        if args.model == 'deeplog':
            # [sequentials]; bsz*nstep*1
            raise NotImplementedError
        elif args.model == 'loganomaly':
            # [sequentials, quantatives]; bsz*nstep*1; bsz*nvent*1
            raise NotImplementedError
        elif args.model == 'armnet':
            model = LogSeqEncoder(args.nstep, len(vocab_sizes), sum(vocab_sizes).item(), args.nemb, args.alpha, args.nhid,
                                  args.nquery, args.nhead, args.nlayer, args.dim_feedforward, args.dropout,
                                  args.mlp_nlayer, args.mlp_nhid, args.nevent)
        elif args.model == 'transformer':
            model = TransformerLogSeqEncoder(args.nstep, sum(vocab_sizes).item(), args.nemb, args.nhead, args.nlayer,
                                             args.dim_feedforward, args.dropout, args.mlp_nlayer, args.mlp_nhid,
                                             args.nevent)
        else:
            raise ValueError(f'unknown model {args.model}')

    model = torch.nn.DataParallel(model).cuda()
    if logger: logger.info(f'{model}\nmodel parameters: {sum([p.data.nelement() for p in model.parameters()])}')
    return model
