import torch
from models.snn import SNN, SNN_CONFIGS
from models.armnet import ARMNetModel, ARM_CONFIG
from models.afn import AFNModel,AFN_CONFIG

def create_model(params, logger=None):
    if logger is None: logger = print
    logger(f'=> creating model {params}')

    if params['model'] == 'snn':
        model = SNN(params['nfeat'], params['nclass'], params)
    elif params['model'] == 'afn':
        model = AFNModel(params['nclass'], params['nfeat'], params['nfeat'], params['nemb'],
                         params['afn_hid'], params['mlp_layer'], params['mlp_hid'],
                         params['dropout'], params['ensemble'], params['mlp_layer'], params['mlp_hid'])
    elif params['model'] == 'armnet':
        model = ARMNetModel(params['nclass'], params['nfeat'], params['nfeat'], params['nemb'],
                    params['alpha'], params['arm_hid'], params['mlp_layer'], params['mlp_hid'],
                    params['dropout'], params['ensemble'], params['mlp_layer'], params['mlp_hid'])
    else:
        raise ValueError(f'unknown model type {params["model"]}')

    if torch.cuda.is_available(): model = model.cuda()
    logger(f'model parameters: {sum([p.data.nelement() for p in model.parameters()])}')
    return model

def get_hyperparams(model):
    if model == 'snn':
        return SNN_CONFIGS
    elif model == 'armnet':
        return ARM_CONFIG
    elif model == 'afn':
        return AFN_CONFIG
    else:
        raise ValueError(f'unknown model type {model}')
