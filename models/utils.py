import torch
from models.log_seq_encoder import LogSeqEncoder
from models.transformer_log_seq_encoder import TransformerLogSeqEncoder
from models.lr import LRModel
from models.dnn import DNNModel

def create_model(args, logger, vocab_sizes):
    if logger: logger.info(f'=> creating model {args.model}')
    if args.session_based:
        if args.model == 'lr':
            model = LRModel(2, vocab_sizes[-1])
        elif args.model == 'dnn':
            model = DNNModel(2, vocab_sizes[-1], args.mlp_nlayer, args.mlp_nhid, args.dropout)
        else:
            raise ValueError(f'unknown model {args.model}')
    else:
        if args.model == 'armnet':
            model = LogSeqEncoder(args.nstep, len(vocab_sizes), sum(vocab_sizes).item(), args.nemb, args.alpha, args.nhid,
                                  args.nquery, args.nhead, args.nlayer, args.dim_feedforward, args.dropout,
                                  args.mlp_nlayer, args.mlp_nhid, vocab_sizes[-1])
        elif args.model == 'transformer':
            model = TransformerLogSeqEncoder(args.nstep, sum(vocab_sizes).item(), args.nemb, args.nhead, args.nlayer,
                                             args.dim_feedforward, args.dropout, args.mlp_nlayer, args.mlp_nhid,
                                             vocab_sizes[-1])
        else:
            raise ValueError(f'unknown model {args.model}')

    model = torch.nn.DataParallel(model).cuda()
    if logger: logger.info(f'{model}\nmodel parameters: {sum([p.data.nelement() for p in model.parameters()])}')
    return model
