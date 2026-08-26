import torch.optim as optim
from .muon import Muon
from .config import HYPERPARAMS

def get_optimizers(opt_name, model, dataset):
    hp = HYPERPARAMS[dataset][opt_name]
    
    if opt_name == 'AdamW':
        return [optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])]
    elif opt_name == 'SGD':
        return [optim.SGD(model.parameters(), lr=hp['lr'], momentum=hp['momentum'], weight_decay=hp['weight_decay'])]
    elif opt_name == 'Muon':
        muon_params = [p for p in model.parameters() if p.ndim >= 2]
        other_params = [p for p in model.parameters() if p.ndim < 2]
        return [
            Muon(muon_params, lr=hp['lr'], momentum=hp['momentum']), 
            optim.AdamW(other_params, lr=hp['adamw_lr'], weight_decay=hp['adamw_wd'])
        ]
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")