import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

def get_cifar10_dataloaders(batch_size, imbalance_factor, random_seed):
    # Force Hugging Face to cache inside your project folder
    hf_dataset = load_dataset("uoft-cs/cifar10", cache_dir="./data")
    
    def create_dataset(train=True):
        raw_set = hf_dataset['train' if train else 'test']
        targets = np.array(raw_set['label'])
        data = np.stack([np.array(img) for img in raw_set['img']])
        
        plane_idx = np.where(targets == 0)[0]
        car_idx = np.where(targets == 1)[0]
        other_idx = np.where(targets > 1)[0]
        rng = np.random.RandomState(random_seed)
        
        if train:
            keep_planes = int(len(plane_idx) * 0.95)
            # Apply dynamic imbalance factor (e.g., 5.0 means 5%)
            keep_cars = int(len(car_idx) * (imbalance_factor / 100.0))
        else:
            keep_planes, keep_cars = int(len(plane_idx) * 0.50), int(len(car_idx) * 0.50)
            
        indices = np.concatenate([
            rng.choice(plane_idx, keep_planes, False), 
            rng.choice(car_idx, keep_cars, False), 
            other_idx
        ])
        rng.shuffle(indices)

        orig_labels = targets[indices]
        new_labels = np.where(orig_labels <= 1, 0, orig_labels - 1)
        is_original_car = np.where(orig_labels == 1, 1, 0)

        X = torch.tensor(data[indices]).permute(0, 3, 1, 2).float() / 255.0
        Y = torch.tensor(new_labels).long()
        is_car = torch.tensor(is_original_car).bool()
        
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1)
        
        return TensorDataset((X - mean) / std, Y, is_car)

    train_ds = create_dataset(train=True)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(train_ds, batch_size=batch_size, shuffle=False),
        DataLoader(create_dataset(train=False), batch_size=batch_size, shuffle=False)
    )
