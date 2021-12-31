import torch
from models.log_seq_encoder import LogSeqEncoder
from models.transformer_log_seq_encoder import TransformerLogSeqEncoder
from models.lr import LRModel
from models.dnn import DNNModel

def create_model(args, logger, vocab_sizes):
    logger.info(f'=> creating model {args.model}')
    if args.session_based:
        if args.model == 'lr':
            model = LRModel(2, vocab_sizes[-1])
        elif args.model == 'dnn':
            model = DNNModel(2, vocab_sizes[-1], args.nlayer, args.nhid, args.dropout)
        else:
            raise ValueError(f'unknown model {args.model}')
    else:
        if args.model == 'armnet':
            model = LogSeqEncoder(args.nstep, len(vocab_sizes), sum(vocab_sizes).item(), args.nemb, args.alpha, args.nhid,
                                  args.nquery, args.nhead, args.nlayer, args.dim_feedforward, args.dropout,
                                  args.predictor_nlayer, args.d_predictor, vocab_sizes[-1])
        elif args.model == 'transformer':
            model = TransformerLogSeqEncoder(args.nstep, sum(vocab_sizes).item(), args.nemb, args.nhead, args.nlayer,
                                             args.dim_feedforward, args.dropout, args.predictor_nlayer, args.d_predictor,
                                             vocab_sizes[-1])
        else:
            raise ValueError(f'unknown model {args.model}')

    model = torch.nn.DataParallel(model).cuda()
    logger.info(f'{model}\nmodel parameters: {sum([p.data.nelement() for p in model.parameters()])}')
    return model
