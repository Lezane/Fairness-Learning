import torch

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