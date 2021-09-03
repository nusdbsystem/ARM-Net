import torch
import torch.nn as nn
from models.lr import LRModel, LR_CONFIG
from models.fm import FMModel, FM_CONFIG
from models.dnn import DNNModel, DNN_CONFIG
from models.snn import SNN, SNN_CONFIGS
from models.armnet import ARMNetModel, ARM_CONFIG
from models.perceiverio import PerceiverTab, PERCEIVER_CONFIG


def create_model(config, logger=None, verbose=False):
    if logger is None: logger = print
    if verbose: logger(f'=> creating model {config}')

    if config['model'] == 'lr':
        model = LRModel(config['nclass'], config['nfeat'])
    elif config['model'] == 'fm':
        model = FMModel(config['nclass'], config['nfeat'], config['nemb'])
    elif config['model'] == 'dnn':
        model = DNNModel(config['nclass'], config['nfeat'], config['nfeat'], config['nemb'],
                         config['mlp_layers'], config['mlp_hid'], config['dropout'])
    elif config['model'] == 'snn':
        model = SNN(config['nfeat'], config['nclass'], config)
    elif config['model'] == 'perceiver':
        model = PerceiverTab(config['nclass'], config['nfeat'], config['nfeat'], config['nemb'],
                             config['depth'], config['n_in_query'], config['n_attn_head'], config['hid_dim'])
    elif config['model'] == 'armnet':
        model = ARMNetModel(config['nclass'], config['nfeat'], config['nfeat'], config['nemb'],
                            config['alpha'], config['arm_hid'], config['mlp_layer'], config['mlp_hid'],
                            config['dropout'], config['ensemble'], config['mlp_layer'], config['mlp_hid'])
    else:
        raise ValueError(f'unknown model type {config["model"]}')

    if torch.cuda.is_available(): model = model.cuda()
    if verbose: logger(f'model parameters: {sum([p.data.nelement() for p in model.parameters()])}')
    return model


def get_config(model):
    if model == 'lr':
        return LR_CONFIG
    elif model == 'fm':
        return FM_CONFIG
    elif model == 'dnn':
        return DNN_CONFIG
    elif model == 'snn':
        return SNN_CONFIGS
    elif model == 'perceiver':
        return PERCEIVER_CONFIG
    elif model == 'armnet':
        return ARM_CONFIG
    else:
        raise ValueError(f'unknown model type {model}')


def to_device(model):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    return device