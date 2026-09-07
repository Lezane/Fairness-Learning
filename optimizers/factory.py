import torch.optim as optim
from .muon import Muon
import config  # Import the new unified config

def get_optimizers(opt_name, model):
    if opt_name == 'AdamW':
        return [optim.AdamW(
            model.parameters(), 
            lr=config.ADAM_LEARNING_RATE,
            betas=(config.ADAM_BETA1, config.ADAM_BETA2),
            eps=config.ADAM_EPS,
            weight_decay=config.ADAM_WEIGHT_DECAY
        )]
    elif opt_name == 'SGD':
        return [optim.SGD(
            model.parameters(), 
            lr=config.SGD_LEARNING_RATE, 
            momentum=config.SGD_MOMENTUM
        )]
    elif opt_name == 'Muon':
        muon_params = [p for p in model.parameters() if p.ndim >= 2]
        other_params = [p for p in model.parameters() if p.ndim < 2]
        return [
            Muon(muon_params, lr=config.MUON_LEARNING_RATE, momentum=config.MUON_MOMENTUM), 
            optim.AdamW(
                other_params, 
                lr=config.ADAM_LEARNING_RATE,
                betas=(config.ADAM_BETA1, config.ADAM_BETA2),
                eps=config.ADAM_EPS,
                weight_decay=config.ADAM_WEIGHT_DECAY
            )
        ]
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
