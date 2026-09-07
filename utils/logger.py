import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_k_runs_variance(arch, opt_name, k_runs, epochs, mean_metrics, std_metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{arch.upper()} | {opt_name} | k={k_runs} Runs Averaged", fontsize=16, fontweight='bold')
    
    # We dynamically use the length of the arrays instead of "epochs" to prevent dimension errors
    x = np.arange(len(mean_metrics['train_s0']))
    plots = [
        (0, 0, 'train_s0', 'Train S0 (Minority)', 'tab:blue'),
        (0, 1, 'train_s1', 'Train S1 (Majority)', 'tab:orange'),
        (1, 0, 'test_s0', 'Test S0 (Minority)', 'tab:green'),
        (1, 1, 'test_s1', 'Test S1 (Majority)', 'tab:red')
    ]
    
    for r, c, key, title, color in plots:
        axs[r,c].plot(x, mean_metrics[key], color=color, linewidth=2)
        if k_runs > 1:
            axs[r,c].fill_between(x, mean_metrics[key] - std_metrics[key], mean_metrics[key] + std_metrics[key], color=color, alpha=0.2)
        axs[r,c].set_title(f"{title}: {mean_metrics[key][-1]:.2f}%")
        axs[r,c].grid(True, linestyle='--', alpha=0.6)
        axs[r,c].set_ylim(0, 105)
        axs[r,c].set_xlabel("Epoch")
        axs[r,c].set_ylabel("Accuracy (%)")
        
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"{arch}_{opt_name}_variance_plot.png")
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    plt.close(fig)

def print_summary_table(dataset, k_runs, epochs, imbalance_factor, arch_results_dict, out_dir):
    rows = []
    for arch, arch_results in arch_results_dict.items():
        # Setup the rows we want to display for this architecture
        row_final = {'Metric': f'{arch.upper()} - Avg Final Test Acc (S0 / S1)'}
        
        row_ep_train_s0 = {'Metric': f'{arch.upper()} - 1st Ep > 80% Train S0 (Minority)'}
        row_ep_train_s1 = {'Metric': f'{arch.upper()} - 1st Ep > 80% Train S1 (Majority)'}
        row_ep_test_s0 = {'Metric': f'{arch.upper()} - 1st Ep > 80% Test S0 (Minority)'}
        row_ep_test_s1 = {'Metric': f'{arch.upper()} - 1st Ep > 80% Test S1 (Majority)'}
        
        for opt_name in ['AdamW', 'SGD', 'Muon']:
            res = arch_results[opt_name]
            
            # CHANGE 2: Calculate the average of the metrics across the `k_runs` at every epoch
            mean_metrics = {k: np.mean(v, axis=0) for k, v in res.items()}
            
            # Get the Final Test Accuracy (from the very last recorded epoch)
            final_test_s0 = mean_metrics['test_s0'][-1] if len(mean_metrics['test_s0']) > 0 else 0.0
            final_test_s1 = mean_metrics['test_s1'][-1] if len(mean_metrics['test_s1']) > 0 else 0.0
            
            row_final[opt_name] = f"{final_test_s0:.2f}% / {final_test_s1:.2f}%"
            
            # CHANGE 3: Helper function to find the exact epoch where accuracy bypasses 80%
            def get_first_ep_80(metric_array):
                # np.where returns an array of indices where the condition (> 80.0) is met
                idx = np.where(np.array(metric_array) > 80.0)[0]
                
                # Assume the index maps to the epoch number. (If your code stores Epoch 1 at 
                # index 0, you could optionally change this to idx[0] + 1)
                return f"Epoch {idx[0]}" if len(idx) > 0 else "Never"

            # Apply the function to our averaged Majority (S1) and Minority (S0) arrays
            row_ep_train_s0[opt_name] = get_first_ep_80(mean_metrics['train_s0'])
            row_ep_train_s1[opt_name] = get_first_ep_80(mean_metrics['train_s1'])
            row_ep_test_s0[opt_name] = get_first_ep_80(mean_metrics['test_s0'])
            row_ep_test_s1[opt_name] = get_first_ep_80(mean_metrics['test_s1'])
            
        # Add all rows to our table display in order
        rows.extend([
            row_final, 
            row_ep_train_s0, row_ep_train_s1, 
            row_ep_test_s0, row_ep_test_s1
        ])
        
    df = pd.DataFrame(rows).set_index('Metric')
    
    title = f"\n--- Summary Table for {dataset.upper()} (k={k_runs}, epochs={epochs}"
    if dataset == 'cifar10': title += f", imbalance={imbalance_factor}%"
    print(title + ") ---")
    
    try:
        from tabulate import tabulate
        print(df.to_markdown())
    except ImportError:
        print(df.to_string())
    
    # Save table to CSV
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{dataset}_summary.csv")
    df.to_csv(csv_path)
    print(f"\nSaved summary table to {csv_path}")
