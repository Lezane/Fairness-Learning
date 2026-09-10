import torch
import numpy as np

def get_split_acc(m, loader, device, is_cifar):
    m.eval()
    correct_s0, total_s0, correct_s1, total_s1 = 0, 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            if is_cifar:
                inputs, labels, s0_mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                outputs = m(inputs)
            else:
                x_num, x_cat, labels, s0_mask = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
                outputs = m(x_num, x_cat)
                
            _, predicted = torch.max(outputs, 1)
            mask_s0, mask_s1 = (s0_mask == True).cpu().numpy(), (s0_mask == False).cpu().numpy()
            labels_cpu, predicted_cpu = labels.cpu().numpy(), predicted.cpu().numpy()

            total_s0 += mask_s0.sum()
            total_s1 += mask_s1.sum()
            if mask_s0.sum() > 0: correct_s0 += (predicted_cpu[mask_s0] == labels_cpu[mask_s0]).sum()
            if mask_s1.sum() > 0: correct_s1 += (predicted_cpu[mask_s1] == labels_cpu[mask_s1]).sum()

    return (100 * correct_s0 / total_s0 if total_s0 > 0 else 0.0), (100 * correct_s1 / total_s1 if total_s1 > 0 else 0.0)



def get_quarter_averages(accuracy_history):
    """
    Module 1: Divides the total iterations into 4 chunks and averages the accuracy for each.
    """
    if not accuracy_history:
        return []
    
    # array_split handles indivisible lengths gracefully
    chunks = np.array_split(accuracy_history, 4)
    return [np.mean(chunk) if len(chunk) > 0 else 0.0 for chunk in chunks]

def get_first_iteration_above_k(accuracy_history, k=90.0):
    """
    Module 2: Returns the first iteration (1-based) where accuracy >= K%.
    Returns "N/A" if it never reaches K%.
    """
    for i, acc in enumerate(accuracy_history):
        if acc >= k:
            return i + 1
    return "N/A"
