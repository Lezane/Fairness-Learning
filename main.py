import argparse
import copy
import torch
import numpy as np

# Ensures matplotlib does not crash on headless SSH servers
import matplotlib
matplotlib.use('Agg')

# Imports from our modules
from utils.seed import set_seed
from utils.trainer import train_and_track
from utils.logger import plot_k_runs_variance, print_summary_table
from utils.storage import ExperimentRecorder
from optimizers.factory import get_optimizers

from data.cifar import get_cifar10_dataloaders
from data.covertype import get_covertype_dataloaders
from models.cifar import get_cifar10_model
from models.tabular import FTTransformerWrapper, MLPWrapper

def main():
    parser = argparse.ArgumentParser(description="Unified Pipeline: Covertype & CIFAR-10")
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'covertype'])
    parser.add_argument('--archs', nargs='+', required=True, help="List of architectures to run")
    parser.add_argument('--k_runs', type=int, default=3, help="Run experience k times sequentially.")
    parser.add_argument('--epochs', type=int, default=200, help="Epoch count.")
    parser.add_argument('--imbalance_factor', type=float, default=5.0, help="CIFAR-10 Minor class ratio.")
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = 256 if args.dataset == 'cifar10' else 2048

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing natively using device: {device}\n")
    
    is_cifar = (args.dataset == 'cifar10')
    arch_results_dict = {}

    # Initialize the robust SSH-safe CSV recorder
    recorder = ExperimentRecorder(args.output_dir, args.dataset)

    for arch in args.archs:
        print(f"\n{'='*60}\n Starting Experiment: {args.dataset.upper()} | Model: {arch.upper()}\n{'='*60}")
        arch_results = {opt: {'train_s0': [], 'train_s1': [], 'test_s0': [], 'test_s1': []} for opt in ['AdamW', 'SGD', 'Muon']}
        
        for run_idx in range(args.k_runs):
            seed = 42 + run_idx
            set_seed(seed)
            print(f"\n>>> Executing Run {run_idx+1}/{args.k_runs} (Seed: {seed}) <<<")
            
            # Setup Data & Base Model
            if is_cifar:
                train_ldr, eval_ldr, test_ldr = get_cifar10_dataloaders(args.batch_size, args.imbalance_factor, seed)
                base_model = get_cifar10_model(arch, device)
            else:
                train_ldr, eval_ldr, test_ldr, num_feat, cat_cards = get_covertype_dataloaders(args.batch_size, seed)
                if arch.lower() == 'fttransformer':
                    base_model = FTTransformerWrapper(num_feat, cat_cards).to(device)
                else:
                    base_model = MLPWrapper(num_feat, cat_cards).to(device)
                    
            initial_state = copy.deepcopy(base_model.state_dict())
            
            # Sequence: AdamW, SGD, Muon
            for opt_name in ['AdamW', 'SGD', 'Muon']:
                print(f"\n -> Training with Optimizer: {opt_name}")
                model = copy.deepcopy(base_model)
                model.load_state_dict(initial_state)
                
                optimizers = get_optimizers(opt_name, model, args.dataset)
                
                # Pass the recorder to log data dynamically
                metrics = train_and_track(
                    model, optimizers, train_ldr, eval_ldr, test_ldr, device, 
                    args.epochs, is_cifar, args.dataset, arch, opt_name, run_idx, seed, recorder
                )
                
                for k in metrics:
                    arch_results[opt_name][k].append(metrics[k])
                    
        arch_results_dict[arch] = arch_results
        
        # Plotting (Works on headless servers thanks to `matplotlib.use('Agg')`)
        for opt_name in ['AdamW', 'SGD', 'Muon']:
            res = arch_results[opt_name]
            mean_metrics = {k: np.mean(v, axis=0) for k, v in res.items()}
            std_metrics = {k: np.std(v, axis=0) for k, v in res.items()}
            plot_k_runs_variance(arch, opt_name, args.k_runs, args.epochs, mean_metrics, std_metrics, args.output_dir)

    print_summary_table(args.dataset, args.k_runs, args.epochs, args.imbalance_factor, arch_results_dict, args.output_dir)

if __name__ == '__main__':
    main()
