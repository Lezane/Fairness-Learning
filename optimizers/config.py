# Set A is for Covertype | Set B is for CIFAR-10
HYPERPARAMS = {
    'covertype': {
        'AdamW': {'lr': 1e-3, 'weight_decay': 1e-2},
        'SGD':   {'lr': 1e-2, 'momentum': 0.9, 'weight_decay': 5e-4},
        'Muon':  {'lr': 0.02, 'momentum': 0.95, 'adamw_lr': 1e-3, 'adamw_wd': 1e-2}
    },
    'cifar10': {
        'AdamW': {'lr': 5e-4, 'weight_decay': 1e-2},
        'SGD':   {'lr': 0.05, 'momentum': 0.9, 'weight_decay': 1e-4},
        'Muon':  {'lr': 0.02, 'momentum': 0.95, 'adamw_lr': 5e-4, 'adamw_wd': 1e-2}
    }
}