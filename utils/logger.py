import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_k_runs_variance(arch, opt_name, k_runs, epochs, mean_metrics, std_metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{arch.upper()} | {opt_name} | k={k_runs} Runs Averaged", fontsize=16, fontweight='bold')
    
    x = np.arange(epochs + 1)
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
        row_early = {'Metric': f'{arch.upper()} - Early Train Accuracy (S0/S1)'}
        row_final = {'Metric': f'{arch.upper()} - Final Test Accuracy (S0/S1)'}
        
        for opt_name in ['AdamW', 'SGD', 'Muon']:
            res = arch_results[opt_name]
            mean_metrics = {k: np.mean(v, axis=0) for k, v in res.items()}
            
            # Early Accuracy Calculation (avg over epochs 1-50)
            early_end = min(epochs + 1, 51)
            early_s0 = np.mean(mean_metrics['train_s0'][1:early_end]) if epochs >= 1 else 0.0
            early_s1 = np.mean(mean_metrics['train_s1'][1:early_end]) if epochs >= 1 else 0.0
            
            # Final Test Accuracy (avg over last 20 epochs)
            final_start = max(1, epochs + 1 - 20)
            final_s0 = np.mean(mean_metrics['test_s0'][final_start:])
            final_s1 = np.mean(mean_metrics['test_s1'][final_start:])
            
            row_early[opt_name] = f"{early_s0:.2f}% / {early_s1:.2f}%"
            row_final[opt_name] = f"{final_s0:.2f}% / {final_s1:.2f}%"
            
        rows.extend([row_early, row_final])
        
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