import torch
import torch.nn as nn
import config
from optimizers.muon import Muon

def get_optimizers(opt_name, model, is_cifar):
    """
    Returns a list of optimizers for the given model.
    For standard optimizers (AdamW, SGD), returns [opt].
    For Muon, splits parameters and returns [opt_muon, opt_adamw].
    """
    
    if opt_name == 'AdamW':
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=config.ADAM_LEARNING_RATE,
            betas=(config.ADAM_BETA1, config.ADAM_BETA2),
            eps=config.ADAM_EPS,
            weight_decay=config.ADAM_WEIGHT_DECAY
        )
        return [opt]

    elif opt_name == 'SGD':
        opt = torch.optim.SGD(
            model.parameters(),
            lr=config.SGD_LEARNING_RATE,
            momentum=config.SGD_MOMENTUM,
            weight_decay=config.SGD_WEIGHT_DECAY
        )
        return [opt]

    elif opt_name == 'Muon':
        muon_params = []
        adamw_params = []

        if is_cifar:
            # ==========================================
            # CIFAR-10 (CNNs like ResNet/VGG)
            # ==========================================
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    # Send 2D Convolution weights to Muon
                    if module.weight is not None:
                        muon_params.append(module.weight)
                    if module.bias is not None:
                        adamw_params.append(module.bias)
                elif isinstance(module, (nn.BatchNorm2d, nn.Linear)):
                    # Send BatchNorm and the final Linear classifier to AdamW
                    adamw_params.extend([p for p in module.parameters(recurse=False) if p is not None])
        else:
            # ==========================================
            # COVERTYPE (FT-Transformer / MLP)
            # ==========================================
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    # Send internal Linear weights to Muon
                    if module.weight is not None:
                        # Keep the final classification head on AdamW for stability
                        if 'head' in name or 'classifier' in name:
                            adamw_params.append(module.weight)
                        else:
                            muon_params.append(module.weight)
                    
                    # Biases always go to AdamW
                    if module.bias is not None:
                        adamw_params.append(module.bias)
                        
                elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.Embedding)):
                    # Send Tabular embeddings and normalizations to AdamW
                    adamw_params.extend([p for p in module.parameters(recurse=False) if p is not None])

        # Initialize both optimizers for the Muon setup
        opt_muon = Muon(
            muon_params, 
            lr=config.MUON_LEARNING_RATE, 
            momentum=config.MUON_MOMENTUM
        )
        
        opt_adamw = torch.optim.AdamW(
            adamw_params, 
            lr=config.MUON_ADAMW_LR, 
            weight_decay=config.MUON_ADAMW_WD
        )
        
        return [opt_muon, opt_adamw]

    else:
        raise ValueError(f"Optimizer {opt_name} is not supported.")
