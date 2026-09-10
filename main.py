import argparse
import copy
import torch
import numpy as np
import matplotlib
# Ensures matplotlib does not crash on headless SSH servers
matplotlib.use('Agg')

import config # New unified config

from utils.seed import set_seed
from utils.trainer import train_and_track
from utils.logger import plot_k_runs_variance, print_summary_table
from utils.storage import ExperimentRecorder
from optimizers.factory import get_optimizers

from data.cifar import get_cifar10_dataloaders
from data.covertype import get_covertype_dataloaders
from models.cifar import get_cifar10_model
from models.tabular import FTTransformerWrapper, MLPWrapper
# -------------------------------------------------------------

# --- Module Helpers ---
def get_quarter_averages(accuracy_history):
    """Module 1: Divides iterations into 4 chunks and averages the accuracy."""
    if len(accuracy_history) == 0:
        return []
    chunks = np.array_split(accuracy_history, 4)
    return [np.mean(chunk) if len(chunk) > 0 else 0.0 for chunk in chunks]

def get_first_iteration_above_k(accuracy_history, k):
    """Module 2: Returns the first iteration (1-based) where accuracy >= K%."""
    for i, acc in enumerate(accuracy_history):
        if acc >= k:
            return i + 1
    return "N/A"
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified Pipeline: Covertype & CIFAR-10")
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'covertype'])
    parser.add_argument('--archs', nargs='+', required=True, help="List of architectures to run")
    parser.add_argument('--k_runs', type=int, default=3, help="Run experience k times sequentially.")
    parser.add_argument('--epochs', type=int, default=config.STEPS, help="Epoch count.")
    parser.add_argument('--xi_remove_pct', type=float, default=95.0, help="Percentage of class 1 to remove.")
    parser.add_argument('--k_threshold', type=float, default=90.0, help="Threshold K% for Module 2.")
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    args = parser.parse_args()

    device = config.DEVICE
    print(f"Executing natively using device: {device}\n")
    
    is_cifar = (args.dataset == 'cifar10')
    arch_results_dict = {}
    recorder = ExperimentRecorder(args.output_dir, args.dataset)

    # Dictionary to store Module 2 data: arch -> run_idx -> opt_name -> {s0: iter, s1: iter}
    module2_stats = {} 

    for arch in args.archs:
        print(f"\n{'='*60}\n Starting Experiment: {args.dataset.upper()} | Model: {arch.upper()}\n{'='*60}")
        arch_results = {opt: {'train_s0': [], 'train_s1': [], 'test_s0': [], 'test_s1': []} for opt in ['AdamW', 'SGD', 'Muon']}
        module2_stats[arch] = {}
        
        for run_idx in range(args.k_runs):
            seed = config.SEED + run_idx
            set_seed(seed)
            print(f"\n>>> Executing Run {run_idx+1}/{args.k_runs} (Seed: {seed}) <<<")
            module2_stats[arch][run_idx] = {}
            
            # Setup Base Model
            if is_cifar:
                base_model = get_cifar10_model(arch, device)
            else:
                _, _, _, num_feat, cat_cards = get_covertype_dataloaders(256, seed)
                if arch.lower() == 'fttransformer':
                    base_model = FTTransformerWrapper(num_feat, cat_cards).to(device)
                else:
                    base_model = MLPWrapper(num_feat, cat_cards).to(device)
                    
            initial_state = copy.deepcopy(base_model.state_dict())
            
            for opt_name in ['AdamW', 'SGD', 'Muon']:
                print(f"\n -> Training with Optimizer: {opt_name}")
                model = copy.deepcopy(base_model)
                model.load_state_dict(initial_state)
                
                if opt_name == 'AdamW':
                    bs = config.ADAM_TRAIN_BATCH_SIZE
                elif opt_name == 'SGD':
                    bs = config.SGD_TRAIN_BATCH_SIZE
                else:
                    bs = config.MUON_TRAIN_BATCH_SIZE
                
                if args.batch_size is not None:
                    bs = args.batch_size
                
                if is_cifar:
                    train_ldr, eval_ldr, test_ldr = get_cifar10_dataloaders(bs, args.xi_remove_pct, seed)
                else:
                    train_ldr, eval_ldr, test_ldr, _, _ = get_covertype_dataloaders(bs, seed)
                
                optimizers = get_optimizers(opt_name, model)
                
                metrics = train_and_track(
                    model, optimizers, train_ldr, eval_ldr, test_ldr, device, 
                    args.epochs, is_cifar, args.dataset, arch, opt_name, run_idx, seed, recorder
                )
                
                # Module 2: Record first iteration passing K% for this run
                module2_stats[arch][run_idx][opt_name] = {
                    's0': get_first_iteration_above_k(metrics['train_s0'], args.k_threshold),
                    's1': get_first_iteration_above_k(metrics['train_s1'], args.k_threshold)
                }

                for k in metrics:
                    arch_results[opt_name][k].append(metrics[k])
                    
        arch_results_dict[arch] = arch_results
        
        for opt_name in ['AdamW', 'SGD', 'Muon']:
            res = arch_results[opt_name]
            mean_metrics = {k: np.mean(v, axis=0) for k, v in res.items()}
            std_metrics = {k: np.std(v, axis=0) for k, v in res.items()}
            plot_k_runs_variance(arch, opt_name, args.k_runs, args.epochs, mean_metrics, std_metrics, args.output_dir)

    # Standard Summary Table
    print_summary_table(args.dataset, args.k_runs, args.epochs, args.xi_remove_pct, arch_results_dict, args.output_dir)

    # --- Print Module 1 & 2 Custom Tables ---
    print("\n" + "="*80)
    print(" " * 25 + "MODULE 1 & 2 RESULTS")
    print("="*80)

    for arch in args.archs:
        print(f"\nArchitecture: {arch.upper()}")
        
        # --- MODULE 1 PRINT ---
        print(f"\n--- Module 1: Average Training Accuracy per Quarter (Averaged across {args.k_runs} runs) ---")
        print(f"{'Optimizer':<10} | {'Q1 (L0/L1)':<15} | {'Q2 (L0/L1)':<15} | {'Q3 (L0/L1)':<15} | {'Q4 (L0/L1)':<15}")
        print("-" * 80)
        
        for opt_name in ['AdamW', 'SGD', 'Muon']:
            res = arch_results_dict[arch][opt_name]
            mean_s0 = np.mean(res['train_s0'], axis=0)
            mean_s1 = np.mean(res['train_s1'], axis=0)
            
            q_s0 = get_quarter_averages(mean_s0)
            q_s1 = get_quarter_averages(mean_s1)
            
            if q_s0 and q_s1:
                q1 = f"{q_s0[0]:.1f}%/{q_s1[0]:.1f}%"
                q2 = f"{q_s0[1]:.1f}%/{q_s1[1]:.1f}%"
                q3 = f"{q_s0[2]:.1f}%/{q_s1[2]:.1f}%"
                q4 = f"{q_s0[3]:.1f}%/{q_s1[3]:.1f}%"
                print(f"{opt_name:<10} | {q1:<15} | {q2:<15} | {q3:<15} | {q4:<15}")
        
        # --- MODULE 2 PRINT ---
        print(f"\n--- Module 2: First Iteration to Pass {args.k_threshold}% Training Accuracy ---")
        print(f"{'Run':<5} | {'Optimizer':<10} | {'L0 (s0) Iter':<12} | {'L1 (s1) Iter':<12}")
        print("-" * 50)
        
        for run_idx in range(args.k_runs):
            for opt_name in ['AdamW', 'SGD', 'Muon']:
                stats = module2_stats[arch][run_idx][opt_name]
                print(f"{run_idx+1:<5} | {opt_name:<10} | {stats['s0']:<12} | {stats['s1']:<12}")

if __name__ == '__main__':
    main()
