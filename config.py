import torch
import torch.nn as nn

# =========================================================
# 1. GENERAL & DATASET CONFIGURATION
# =========================================================
SEED = 42  # Seed to ensure optimizers start from the exact same weights
MINORITY_CLASS = 1         # 1 = 'car' in CIFAR-10
REMOVE_PERCENTAGE = 95     # Remove 95% of the car training dataset
CURRENT_ARCH = 'vgg19'  # Used across ALL optimizers for a fair comparison

# Transformer / Architecture Settings (Used by FT-Transformer)
INIT_STD = 0.02
DEPTH = 12
NUM_HEADS = 12
EMB_DIM = 768

# Training Settings
BATCH_SIZE = 256
NUM_EPOCHS = 200

# =========================================================
# 2. OPTIMIZER HYPERPARAMETERS
# =========================================================
# AdamW Parameters
ADAM_GRADIENT_ACC_STEPS = 16
ADAM_LEARNING_RATE = 5e-4
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 0.01

# SGD Parameters
SGD_GRADIENT_ACC_STEPS = 16
SGD_LEARNING_RATE = 0.05 
SGD_MOMENTUM = 0.9
SGD_WEIGHT_DECAY = 1e-4

# Muon Parameters
MUON_GRADIENT_ACC_STEPS = 16
MUON_LEARNING_RATE = 0.02
MUON_MOMENTUM = 0.95
MUON_ADAMW_LR = 5e-4
MUON_ADAMW_WD = 0.01

# =========================================================
# 3. HARDWARE & CACHE
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_CACHE_DIR = "./dataset_cache"

# =========================================================
# 4. MODEL INITIALIZATION
# =========================================================
def initialize_weights(model, num_classes=None):
    for m in model.modules():
        # 1. Convolutional Layers (CNNs like VGG/ResNet)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        # 2. Normalization Layers (1D for Tabular, 2D for CNNs, LayerNorm for Transformers)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        # 3. Linear Layers (MLP, FT-Transformer, and CNN Classification Heads)
        elif isinstance(m, nn.Linear):
            if num_classes is not None and m.out_features == num_classes:
                # Classification head
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            else:
                # Hidden linear layers (Kaiming works well for ReLU MLPs and generic projections)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        # 4. Embeddings (Categorical Features in FT-Transformers)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

    # 5. ResNet-specific Zero-initialization (Safely ignored by MLPs/Transformers)
    for m in model.modules():
        if type(m).__name__ == 'Bottleneck' and hasattr(m, 'bn3'):
            nn.init.constant_(m.bn3.weight, 0)
        elif type(m).__name__ == 'BasicBlock' and hasattr(m, 'bn2'):
            nn.init.constant_(m.bn2.weight, 0)
