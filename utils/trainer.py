import torch
import torch.nn as nn
from .metrics import get_split_acc

def train_and_track(model, optimizers, trainloader, evalloader, testloader, device, 
                    epochs, is_cifar, dataset_name, arch, opt_name, run_idx, seed, recorder):
    criterion = nn.CrossEntropyLoss()
    metrics = {'train_s0': [], 'train_s1': [], 'test_s0': [], 'test_s1': []}
    
    run_id = f"Run_{run_idx+1}"

    # Store Epoch 0 Base accuracy
    tr_s0, tr_s1 = get_split_acc(model, evalloader, device, is_cifar)
    te_s0, te_s1 = get_split_acc(model, testloader, device, is_cifar)
    
    metrics['train_s0'].append(tr_s0); metrics['train_s1'].append(tr_s1)
    metrics['test_s0'].append(te_s0);  metrics['test_s1'].append(te_s1)
    
    # Save Epoch 0 to CSV securely
    recorder.record_epoch(dataset_name, arch, opt_name, run_id, seed, 0, tr_s0, tr_s1, te_s0, te_s1)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in trainloader:
            if is_cifar:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(inputs)
            else:
                x_num, x_cat, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                outputs = model(x_num, x_cat)
            
            for opt in optimizers: opt.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            for opt in optimizers: opt.step()

        # Track Metrics
        tr_s0, tr_s1 = get_split_acc(model, evalloader, device, is_cifar)
        te_s0, te_s1 = get_split_acc(model, testloader, device, is_cifar)

        metrics['train_s0'].append(tr_s0); metrics['train_s1'].append(tr_s1)
        metrics['test_s0'].append(te_s0);  metrics['test_s1'].append(te_s1)
        
        # Save current epoch to CSV securely
        recorder.record_epoch(dataset_name, arch, opt_name, run_id, seed, epoch, tr_s0, tr_s1, te_s0, te_s1)
        
        if epoch % 10 == 0 or epoch == epochs:
            print(f"[{opt_name} - {run_id}] Epoch {epoch:03d}/{epochs} | Train (S0/S1): {tr_s0:5.1f}/{tr_s1:5.1f} | Test (S0/S1): {te_s0:5.1f}/{te_s1:5.1f}")
            
    return metrics
