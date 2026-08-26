
```text
project_root/
│
├── requirements.txt      # Dependencies
├── main.py               # CLI entry point to run experiments
│
├── data/                 # Dataset loaders
│   ├── __init__.py
│   ├── cifar.py          # CIFAR-10 / CIFAR-100 loaders
│   └── covertype.py      # Forest Covertype tabular data loader
│
├── models/               # Neural network architectures
│   ├── __init__.py
│   ├── cifar.py          # ResNet10, VGG19-bn
│   └── tabular.py        # MLP, FT-Transformer
│
├── optimizers/           # Optimizer logic and hyperparameters
│   ├── __init__.py
│   ├── muon.py           # Custom Muon implementation
│   ├── config.py         # Hyperparameter sets A and B
│   └── factory.py        # Factory function to build optimizers
│
└── utils/                # Training loop, metrics, and plotting
    ├── __init__.py
    ├── seed.py           # Seed setting for reproducibility
    ├── metrics.py        # Accuracy tracking and loss calculation
    ├── trainer.py        # Main training loop
    └── logger.py         # Plotting and summary tables

Example of commands

For the Covertype Experience (Set A):
Runs mlp and fttransformer 3 times (k=3), tracking the variance with Seed 42, 43, 44.
python main.py --dataset covertype --archs mlp fttransformer --k_runs 3 --epochs 200

For the CIFAR-10 Experience (Set B):
Runs resnet10 and vgg19_bn 3 times. You can change --imbalance_factor freely between 5.0 and 30.0.
python main.py --dataset cifar10 --archs resnet10 vgg19_bn --k_runs 3 --epochs 200 --imbalance_factor 5.0
