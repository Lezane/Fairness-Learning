
```text
project_root/
│
├── requirements.txt      
├── main.py               # (UPDATED)
│
├── data/                 
│   ├── __init__.py
│   ├── cifar.py
│   └── covertype.py
│
├── models/               
│   ├── __init__.py
│   ├── cifar.py          
│   └── tabular.py        
│
├── optimizers/           
│   ├── __init__.py
│   ├── muon.py           
│   ├── config.py         
│   └── factory.py        
│
└── utils/                
    ├── __init__.py
    ├── seed.py           
    ├── metrics.py        
    ├── trainer.py        # (UPDATED)
    ├── logger.py         
    └── storage.py        # <--- NEW: Server-safe data storage

Example of commands

For the Covertype Experience (Set A):
Runs mlp and fttransformer 3 times (k=3), tracking the variance with Seed 42, 43, 44.
python main.py --dataset covertype --archs mlp fttransformer --k_runs 3 --epochs 200


For the CIFAR-10 Experience (Set B):
Runs resnet10 and vgg19_bn 3 times. You can change --imbalance_factor freely between 5.0 and 30.0.
python main.py --dataset cifar10 --archs resnet10 vgg19_bn --k_runs 3 --epochs 200 --imbalance_factor 5.0

To save the logs
nohup python main.py --dataset cifar10 --archs resnet10 vgg19_bn --k_runs 3 --epochs 200 --imbalance_factor 5.0 > cifar10_experiment.log 2>&1 &
nohup python main.py --dataset covertype --archs mlp fttransformer --k_runs 3 --epochs 200 > covertype_experiment.log 2>&1 &

To check the logs
tail -f cifar10_experiment.log
