import torch
from models.lr import LRModel
from models.fm import FMModel
from models.hofm import HOFMModel
from models.afm import AFMModel
from models.dcn import CrossNetModel
from models.xdfm import CINModel

from models.dnn import DNNModel
from models.gcn import GCNModel
from models.gat import GATModel

from models.wd import WDModel
from models.pnn import IPNNModel
from models.pnn import KPNNModel
from models.nfm import NFMModel
from models.dfm import DeepFMModel
from models.dcn import DCNModel
from models.xdfm import xDeepFMModel

from models.afn import AFNModel
from models.armnet import ARMNetModel
from models.armnet_1h import ARMNetModel as ARMNet1H
from models.gc_arm import GC_ARMModel
from models.sa_glu import SA_GLUModel

def create_model(args, logger):
    logger.info(f'=> creating model {args.model}')
    if args.model == 'lr':
        model = LRModel(args.nfeat)
    elif args.model == 'fm':
        model = FMModel(args.nfeat, args.nemb)
    elif args.model == 'hofm':
        model = HOFMModel(args.nfeat, args.nemb, args.k)
    elif args.model == 'afm':
        model = AFMModel(args.nfeat, args.nemb, args.h, args.dropout)
    elif args.model == 'dcn':
        model = CrossNetModel(args.nfield, args.nfeat, args.nemb, args.k)
    elif args.model == 'cin':
        model = CINModel(args.nfield, args.nfeat, args.nemb, args.k, args.h)
    elif args.model == 'afn':
        model = AFNModel(args.nfield, args.nfeat, args.nemb, args.h, args.mlp_nlayer, args.mlp_nhid,
                    args.dropout, args.ensemble, args.dnn_nlayer, args.dnn_nhid)
    elif args.model == 'armnet':
        model = ARMNetModel(args.nfield, args.nfeat, args.nemb, args.nattn_head, args.alpha, args.h,
                    args.mlp_nlayer, args.mlp_nhid, args.dropout, args.ensemble, args.dnn_nlayer, args.dnn_nhid)
    elif args.model == 'armnet_1h':
        model = ARMNet1H(args.nfield, args.nfeat, args.nemb, args.alpha, args.h, args.nemb, args.mlp_nlayer,
                         args.mlp_nhid, args.dropout, args.ensemble, args.dnn_nlayer, args.dnn_nhid)

    elif args.model == 'dnn':
        model = DNNModel(args.nfield, args.nfeat, args.nemb, args.mlp_nlayer, args.mlp_nhid, args.dropout)
    elif args.model == 'gcn':
        model = GCNModel(args.nfield, args.nfeat, args.nemb, args.k, args.h, args.mlp_nlayer,
                         args.mlp_nhid, args.dropout)
    elif args.model == 'gat':
        model = GATModel(args.nfield, args.nfeat, args.nemb, args.k, args.h,
                         args.mlp_nlayer, args.mlp_nhid, args.dropout, 0.2, args.nattn_head)

    elif args.model == 'wd':
        model = WDModel(args.nfield, args.nfeat, args.nemb, args.mlp_nlayer, args.mlp_nhid, args.dropout)
    elif args.model == 'ipnn':
        model = IPNNModel(args.nfield, args.nfeat, args.nemb, args.mlp_nlayer, args.mlp_nhid, args.dropout)
    elif args.model == 'kpnn':
        model = KPNNModel(args.nfield, args.nfeat, args.nemb, args.mlp_nlayer, args.mlp_nhid, args.dropout)
    elif args.model == 'nfm':
        model = NFMModel(args.nfeat, args.nemb, args.mlp_nlayer, args.mlp_nhid, args.dropout)
    elif args.model == 'dfm':
        model = DeepFMModel(args.nfield, args.nfeat, args.nemb, args.mlp_nlayer, args.mlp_nhid, args.dropout)
    elif args.model == 'dcn+':
        model = DCNModel(args.nfield, args.nfeat, args.nemb, args.k, args.mlp_nlayer, args.mlp_nhid, args.dropout)
    elif args.model == 'xdfm':
        model = xDeepFMModel(args.nfield, args.nfeat, args.nemb, args.k, args.h,
                    args.mlp_nlayer, args.mlp_nhid, args.dropout)

    elif args.model == 'gc_arm':
        model = GC_ARMModel(args.nfield, args.nfeat, args.nemb, args.nattn_head, args.alpha, args.h, args.mlp_nlayer,
                            args.mlp_nhid, args.dropout, args.ensemble, args.dnn_nlayer, args.dnn_nhid)
    elif args.model == 'sa_glu':
        model = SA_GLUModel(args.nfield, args.nfeat, args.nemb, args.mlp_nlayer, args.mlp_nhid, args.dropout,
                            args.ensemble, args.dnn_nlayer, args.dnn_nhid)

    else:
        raise ValueError(f'unknown model {args.model}')

    if torch.cuda.is_available(): model = model.cuda()
    logger.info(f'{model}\nmodel parameters: {sum([p.data.nelement() for p in model.parameters()])}')
    return model