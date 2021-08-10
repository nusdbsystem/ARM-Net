import torch
import torch.nn as nn
from models.snn import SNN, SNN_CONFIGS
from models.armnet import ARMNetModel, ARM_CONFIG


def create_model(config, logger=None, verbose=False):
    if logger is None: logger = print
    if verbose: logger(f'=> creating model {config}')

    if config['model'] == 'snn':
        model = SNN(config['nfeat'], config['nclass'], config)
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
    if model == 'snn':
        return SNN_CONFIGS
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