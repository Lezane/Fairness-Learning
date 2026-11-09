"""Additional Covertype MLP experiments; add this file beside the existing main.py.

Runs at most five LR configurations per optimizer, each with exactly two seeds.
No existing repository files or config values are modified. No automatic selection
on test results: summary.csv reports every configuration for comparison.
"""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from data.covertype import get_covertype_dataloaders
from optimizers.muon import Muon
from utils.seed import set_seed
from utils.storage import ExperimentRecorder
from utils.trainer import train_and_track

DEFAULT_LRS = {
    'AdamW': [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
    'SGD': [0.005, 0.01, 0.02, 0.05, 0.1],
    'Muon': [0.001, 0.002, 0.005, 0.01, 0.02],
}


def make_optimizers(name, model, lr, aux_lr):
    """Hard-coded zero history, regardless of momentum settings in config.py."""
    if name == 'AdamW':
        optimizers = [torch.optim.AdamW(
            model.parameters(), lr=lr, betas=(0.0, 0.0), eps=config.ADAM_EPS,
            weight_decay=config.ADAM_WEIGHT_DECAY)]
    elif name == 'SGD':
        optimizers = [torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.0, nesterov=False,
            weight_decay=config.SGD_WEIGHT_DECAY)]
    elif name == 'Muon':
        # Match the repository MLP grouping: hidden weights only on Muon.
        hidden = [m.weight for m in model.mlp.modules() if isinstance(m, torch.nn.Linear)]
        hidden_ids = {id(p) for p in hidden}
        auxiliary = [p for p in model.parameters() if id(p) not in hidden_ids]
        optimizers = [
            Muon(hidden, lr=lr, momentum=0.0, nesterov=False),
            torch.optim.AdamW(auxiliary, lr=aux_lr, betas=(0.0, 0.0),
                              eps=config.ADAM_EPS, weight_decay=config.MUON_ADAMW_WD),
        ]
    else:
        raise ValueError(name)
    assigned = []
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            if group.get('momentum', 0.0) != 0.0 or group.get('betas', (0.0, 0.0)) != (0.0, 0.0):
                raise RuntimeError('Momentum must be disabled in every parameter group')
            assigned.extend(id(p) for p in group['params'])
    if len(assigned) != len(set(assigned)) or set(assigned) != {id(p) for p in model.parameters()}:
        raise RuntimeError('Each model parameter must have exactly one optimizer')
    return optimizers


def optimizer_details(optimizers):
    return [{'optimizer': type(opt).__name__, 'groups': [
        {key: value for key, value in group.items() if key != 'params'}
        for group in opt.param_groups]} for opt in optimizers]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--seeds', type=int, nargs=2, default=[42, 43], metavar=('SEED1', 'SEED2'))
    parser.add_argument('--optimizers', nargs='+', choices=list(DEFAULT_LRS), default=list(DEFAULT_LRS))
    parser.add_argument('--adam_lrs', type=float, nargs='+', default=DEFAULT_LRS['AdamW'])
    parser.add_argument('--sgd_lrs', type=float, nargs='+', default=DEFAULT_LRS['SGD'])
    parser.add_argument('--muon_lrs', type=float, nargs='+', default=DEFAULT_LRS['Muon'])
    parser.add_argument('--muon_aux_ratio', type=float, default=0.05,
                        help='Auxiliary AdamW LR = ratio * Muon LR (default: 0.05).')
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()
    rates = {'AdamW': args.adam_lrs, 'SGD': args.sgd_lrs, 'Muon': args.muon_lrs}
    if args.epochs < 1 or args.batch_size < 1 or len(set(args.seeds)) != 2:
        parser.error('Use positive epochs/batch size and exactly two distinct seeds')
    if len(set(args.optimizers)) != len(args.optimizers):
        parser.error('Do not repeat optimizers')
    for name in args.optimizers:
        values = rates[name]
        if not 1 <= len(values) <= 5 or len(set(values)) != len(values):
            parser.error(f'{name}: provide 1 to 5 distinct learning rates')
        if any(not math.isfinite(v) or v <= 0 for v in values):
            parser.error('Learning rates must be finite and positive')
    if not math.isfinite(args.muon_aux_ratio) or args.muon_aux_ratio <= 0:
        parser.error('Muon auxiliary ratio must be finite and positive')

    # Reuse the actual repository model without altering its architecture.
    from models.tabular import MLPWrapper

    out = Path(args.output_dir or ('./outputs/additional_covertype_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    if out.exists() and any(out.iterdir()):
        parser.error('Choose a new or empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'experiment_settings.json').write_text(json.dumps({
        **vars(args), 'output_dir': str(out), 'device': str(config.DEVICE),
        'no_momentum': True, 'schedule': 'cosine, T_max=epochs, eta_min=0',
        'group_definition': 'S0: other samples; S1: designated minority cover type',
        'muon_aux_rule': 'auxiliary_lr = muon_lr * muon_aux_ratio',
        'total_runs': 2 * sum(len(rates[name]) for name in args.optimizers),
    }, indent=2))
    print(f'Output directory: {out.resolve()}', flush=True)
    print('All momentum and both AdamW betas are zero. Exactly two seeds per configuration.', flush=True)
    # Cache each seed's loaders once; shuffle and dropout RNG are reset for every run.
    datasets = {seed: get_covertype_dataloaders(args.batch_size, seed) for seed in args.seeds}
    if any(len(data[0]) == 0 for data in datasets.values()):
        raise ValueError('Batch size exceeds training sample count')
    summary_path = out / 'summary.csv'
    with summary_path.open('w', newline='') as summary_file:
        fields = ['optimizer', 'lr', 'aux_lr', 'seeds', 'epochs', 'batch_size']
        for metric in ['train_s0', 'train_s1', 'test_s0', 'test_s1']:
            fields += [metric + '_mean', metric + '_std']
        writer = csv.DictWriter(summary_file, fieldnames=fields)
        writer.writeheader()
        for name in args.optimizers:
            for trial, lr in enumerate(rates[name], 1):
                aux_lr = lr * args.muon_aux_ratio if name == 'Muon' else None
                trial_dir = out / f'{name}_{trial}_lr_{lr:g}'
                results = []
                for run_index, seed in enumerate(args.seeds):
                    train, train_eval, test, num_features, cards = datasets[seed]
                    set_seed(seed)
                    model = MLPWrapper(num_features, cards).to(config.DEVICE)
                    config.initialize_weights(model, num_classes=2)
                    optimizers = make_optimizers(name, model, lr, aux_lr)
                    schedulers = [CosineAnnealingLR(opt, T_max=args.epochs) for opt in optimizers]
                    run_dir = trial_dir / f'seed_{seed}'
                    recorder = ExperimentRecorder(str(run_dir), 'covertype')
                    details = optimizer_details(optimizers)
                    (run_dir / 'optimizer_settings.json').write_text(json.dumps(details, indent=2))
                    print(f'\n{name} LR={lr:g}, auxiliary LR={aux_lr}, seed={seed}', flush=True)
                    print(json.dumps(details), flush=True)
                    history = train_and_track(
                        model, optimizers, schedulers, train, train_eval, test, config.DEVICE,
                        args.epochs, False, 'covertype', 'mlp', name, run_index, seed, recorder)
                    results.append({key: history[key][-1] for key in history})
                    del model, optimizers, schedulers
                row = dict(optimizer=name, lr=lr, aux_lr=aux_lr, seeds=','.join(map(str, args.seeds)),
                           epochs=args.epochs, batch_size=args.batch_size)
                for metric in ['train_s0', 'train_s1', 'test_s0', 'test_s1']:
                    values = [r[metric] for r in results]
                    row[metric + '_mean'] = statistics.mean(values)
                    row[metric + '_std'] = statistics.pstdev(values)
                writer.writerow(row)
                summary_file.flush()
                print('Final two-seed summary:', json.dumps(row), flush=True)
    print(f'Finished. All configurations: {summary_path.resolve()}', flush=True)


if __name__ == '__main__':
    main()
