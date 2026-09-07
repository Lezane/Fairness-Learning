import torch
import torch.nn as nn
from torchvision.models import vgg19_bn
import config  # <--- Added config import


def get_cifar10_model(arch, device):
    num_classes = 9
    if arch == 'resnet10':
        model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)
    elif arch == 'vgg19_bn':
        model = vgg19_bn(weights=None, num_classes=num_classes)
        model.avgpool = nn.Identity()
        model.classifier = nn.Linear(512, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # Proper Weight Initialization with Config `INIT_STD`
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or (isinstance(m, nn.Linear) and m.out_features != num_classes):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear) and m.out_features == num_classes:
            nn.init.normal_(m.weight, mean=0.0, std=config.INIT_STD) # <--- Applied Unified INIT_STD
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    for m in model.modules():
        if type(m).__name__ in ['Bottleneck', 'BasicBlock']:
            if hasattr(m, 'bn3'): nn.init.constant_(m.bn3.weight, 0)
            else: nn.init.constant_(m.bn2.weight, 0)
            
    return model.to(device)
