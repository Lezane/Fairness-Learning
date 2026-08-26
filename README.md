

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
