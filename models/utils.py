import torch
from models.log_seq_encoder import LogSeqEncoder
from models.transformer_log_seq_encoder import TransformerLogSeqEncoder


def create_model(args, logger, vocab_sizes):
    logger.info(f'=> creating model {args.model}')
    if args.model == 'armnet':
        model = LogSeqEncoder(args.nstep, len(vocab_sizes), sum(vocab_sizes).item(), args.nemb, args.alpha, args.nhid,
                              args.nquery, args.nhead, args.num_layers, args.dim_feedforward, args.dropout,
                              args.predictor_layers, args.d_predictor, vocab_sizes[-1])
    elif args.model == 'transformer':
        model = TransformerLogSeqEncoder(args.nstep, sum(vocab_sizes).item(), args.nemb, args.nhead, args.num_layers,
                                         args.dim_feedforward, args.dropout, args.predictor_layers, args.d_predictor,
                                         vocab_sizes[-1])
    else:
        raise ValueError(f'unknown model {args.model}')

    model = torch.nn.DataParallel(model).cuda()
    logger.info(f'{model}\nmodel parameters: {sum([p.data.nelement() for p in model.parameters()])}')
    return model
