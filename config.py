import torch

# General Settings
INIT_STD = 0.02
DEPTH = 12
NUM_HEADS = 12
EMB_DIM = 768
SEED = 42  # Seed to ensure AdamW and SGD start from the exact same weights

# Training Settings
TARGET_LENGTH = 1024
STEPS = 200
MAJOR_VOCAB_FRAC = 0.95  # Top 90% most common words

# AdamW Parameters
ADAM_TRAIN_BATCH_SIZE = 32
ADAM_GRADIENT_ACC_STEPS = 16
ADAM_LEARNING_RATE = 5e-4
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 0.01

# SGD Parameters
SGD_TRAIN_BATCH_SIZE = 32
SGD_GRADIENT_ACC_STEPS = 16
SGD_LEARNING_RATE = 0.02 
SGD_MOMENTUM = 0.9

# Muon Parameters
MUON_TRAIN_BATCH_SIZE = 32
MUON_GRADIENT_ACC_STEPS = 16
MUON_LEARNING_RATE = 0.02
MUON_MOMENTUM = 0.95

# Hardware Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache Directory (prevents conflicts with your python module folder names)
DATA_CACHE_DIR = "./dataset_cache"
