"""CIFAR-10 hidden-minority experiment with gradient geometry diagnostics.

Place this file beside main.py in https://github.com/Lezane/Fairness-Learning.
Default: 50 epochs x seeds 42,43,44 x AdamW/SGD/Muon x resnet10/vgg19.
Run: python additional-cifar10-experience.py
Tests (no dataset download): python additional-cifar10-experience.py --self_test

MATHEMATICAL CONVENTIONS
  S0 = original cars; S1 = all other retained training examples. As in data/cifar.py,
  cars and planes share target 0; the classifier has NINE outputs. Default removal
  is 95% of cars (250 cars, 4750 planes, 40000 other training examples).
  f(theta) is mean unregularized cross-entropy. Gradient diagnostics exclude
  weight decay, optimizer state, and orthogonalized updates.
  Each update logs the RAW pre-update training-mode minibatch gradient g_B.
  Epoch 0 and every epoch end additionally log g_full = grad mean_train CE in
  eval mode (fixed BatchNorm buffers, dropout disabled). This is a deterministic
  empirical objective conditional on the current buffers, not an unbiased claim
  about training-mode BatchNorm gradients. Full-gradient accumulation weights
  every example equally, including the short last batch. Test data never enters it.
  Nonlinear support/rank of an averaged gradient is NOT averaged support/rank.

  Table 1 and Eq. (16) of the supplied paper motivate sqrt(|supp(g)|) for sign
  descent and sqrt(rank(G)) for spectral descent. supp(|g|) = supp(g). We report:
    support_nonzero: count of exactly nonzero stored floating-point entries;
    support_thresholded: count |g_i| > atol + rtol*max_j|g_j|, globally;
    support_participation: ||g||_1^2 / ||g||_2^2 (0 at g=0).
  The first is not a certificate of exact real-arithmetic support; the others
  measure concentration, not the paper's exact support.

  For each Muon convolution weight, G = grad.reshape(out_channels, -1), exactly
  as in optimizers/muon.py. SVD is done in float64 on CPU/CUDA. Numerical rank
  counts sigma_i > rank_atol + rank_rtol*sigma_max, where default rank_rtol is
  max(G.shape)*eps(original_gradient_dtype). We also report ranks at 0.1x/10x
  that threshold, stable rank ||G||_F^2/||G||_2^2 and nuclear participation rank
  ||G||_*^2/||G||_F^2. Both concentration measures are 0 for a zero matrix.
  Each layer is logged separately; sum(rank(G_l)) is the numerical rank of the
  block diagonal collection, using the stated PER-BLOCK thresholds. Its square
  root is the Frobenius norm of the corresponding truncated polar directions.
  This is a blockwise geometric diagnostic, not a theorem for the hybrid network.
  Sum nuclear norms / sqrt(sum ranks) additionally records the ideal block-polar
  alignment scale using singular values ABOVE those same thresholds.

  AdamW has betas=(0,0): its data direction is g/(|g|+eps), an approximate sign.
  Muon has zero momentum and five approximate Newton-Schulz steps, with an
  auxiliary AdamW group. Neither is exactly the paper's ideal flow. Prefix means
  of sqrt-support/rank are over observed training updates, NOT over an identified
  pre-entry interval. LR-weighted prefix means use the main optimizer LR as dt.
  We do not estimate psi, D, theoretical entry times, or claim causal validation.

REPRODUCTION NOTES
  Reuses repository data, architecture and Muon implementation; fixes inverted
  S0/S1 labels in utils/metrics.py and main.py's num_classes=10 initialization
  call (actual task has 9 classes). Uses supported vgg19, not README's vgg19_bn.
  Zero-history hyperparameters and cosine schedule match repository config/main
  at b27cd8e6e6b9e6366547e2f0eb00f09c3b1037b7. Muon auxiliary decay defaults to
  repository 1e-3; paper Table 8 says 1e-2: pass --muon_aux_wd 0.01 to use that.
  Paired optimizer runs share split, initialization, shuffle order and dropout RNG.
  All three seeds vary both subsampling and initialization, as in the repository.

OUTPUTS
  Each arch/optimizer/seed: steps.csv, epochs.csv, matrix_layers.csv,
  optimizer_settings.json, final_checkpoint.pt. Root: settings.json,
  epoch_summary.csv (mean, sample SD and n across independent seeds), plots.
  Accuracy is in percent. Epoch metrics include the complete retained train set.
  steps.csv has minibatch S0/S1 accuracy at every update; absent groups are blank,
  not zero. Full train/test S0/S1 accuracy is measured at every epoch boundary.
  SVD every update is costly, especially for VGG19. --diagnostic_every K samples
  geometry every K updates (plus first/final); epoch geometry remains exhaustive.
  Default K=1 implements every-update diagnostics. No full training is needed for
  --self_test; --smoke_test exercises the real model and all optimizer paths on
  synthetic data, explicitly NOT CIFAR results.
"""

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SOURCE_COMMIT = 'b27cd8e6e6b9e6366547e2f0eb00f09c3b1037b7'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class CSVLog:
    """Stream records to disk; fixed schema, including blank unavailable values."""
    def __init__(self, path):
        self.file = path.open('w', newline='')
        self.writer = None

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def divide(x, y):
    return x / y if y else 0.0


def vector_stats(gradients, atol, rtol, eps):
    """Global support, including zero tensors for unused trainable parameters."""
    max_abs = max((float(g.detach().abs().max()) for g in gradients), default=0.0)
    threshold = atol + rtol * max_abs
    count = nonzero = selected = 0
    l1 = l2sq = sign_eps_sq = 0.0
    for raw in gradients:
        g = raw.detach().to(dtype=torch.float64)
        if not bool(torch.isfinite(g).all()):
            raise FloatingPointError('Non-finite raw gradient')
        a = g.abs()
        count += g.numel()
        nonzero += int(torch.count_nonzero(g))
        selected += int((a > threshold).sum())
        l1 += float(a.sum())
        l2sq += float(g.square().sum())
        sign_eps_sq += float((a / (a + eps)).square().sum())
    return dict(
        parameter_count=count, support_nonzero=nonzero,
        support_fraction=divide(nonzero, count), support_threshold=threshold,
        support_thresholded=selected, support_thresholded_fraction=divide(selected, count),
        sqrt_support_nonzero=math.sqrt(nonzero),
        sqrt_support_thresholded=math.sqrt(selected),
        support_participation=divide(l1*l1, l2sq), grad_l1=l1,
        grad_l2=math.sqrt(l2sq), grad_linf=max_abs,
        sign_alignment_scale=divide(l1, math.sqrt(nonzero)),
        adam_eps_sign_direction_l2=math.sqrt(sign_eps_sq),
    )


def matrix_stats(raw, args):
    """Threshold the singular values of the raw gradient, never G.T@G."""
    original_eps = torch.finfo(raw.dtype).eps
    target = args.svd_device
    if target == 'auto':
        target = str(raw.device) if raw.device.type in ('cpu', 'cuda') else 'cpu'
    g = raw.detach().reshape(raw.shape[0], -1).to(device=target, dtype=torch.float64)
    if not bool(torch.isfinite(g).all()):
        raise FloatingPointError('Non-finite matrix gradient')
    s = torch.linalg.svdvals(g)
    spectral = float(s[0]) if s.numel() else 0.0
    rtol = args.rank_rtol if args.rank_rtol is not None else max(g.shape)*original_eps
    tol = args.rank_atol + rtol*spectral
    fro2 = float(s.square().sum())
    nuc = float(s.sum())
    retained = s[s > tol]
    return dict(
        rows=g.shape[0], cols=g.shape[1], max_rank=min(g.shape),
        numerical_rank=int(retained.numel()), rank_tol=tol, rank_rtol=rtol,
        rank_tol_times_0_1=int((s > 0.1*tol).sum()),
        rank_tol_times_10=int((s > 10*tol).sum()),
        stable_rank=divide(fro2, spectral*spectral),
        nuclear_participation_rank=divide(nuc*nuc, fro2),
        grad_nuclear=nuc, grad_spectral=spectral, grad_frobenius=math.sqrt(fro2),
        retained_nuclear=float(retained.sum()),
    )


def geometry(model, matrix_ids, args):
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    gradients = [p.grad if p.grad is not None else torch.zeros_like(p) for _, p in named]
    result = vector_stats(gradients, args.support_atol, args.support_rtol, args.adam_eps)
    result['unused_parameter_tensors'] = sum(p.grad is None for _, p in named)
    layers = []
    for (name, p), g in zip(named, gradients):
        if id(p) in matrix_ids:
            layers.append({'layer': name, **matrix_stats(g, args)})
    if layers:
        rank = sum(v['numerical_rank'] for v in layers)
        capacity = sum(v['max_rank'] for v in layers)
        fro2 = sum(v['grad_frobenius']**2 for v in layers)
        nuc = sum(v['grad_nuclear'] for v in layers)
        spectral = max(v['grad_spectral'] for v in layers)
        result.update(
            matrix_block_count=len(layers), matrix_rank_sum=rank,
            matrix_max_rank_sum=capacity, matrix_rank_fraction=divide(rank, capacity),
            sqrt_matrix_rank_sum=math.sqrt(rank),
            matrix_rank_sum_tol_times_0_1=sum(v['rank_tol_times_0_1'] for v in layers),
            matrix_rank_sum_tol_times_10=sum(v['rank_tol_times_10'] for v in layers),
            matrix_stable_rank_sum=sum(v['stable_rank'] for v in layers),
            matrix_nuclear_participation_rank_sum=sum(v['nuclear_participation_rank'] for v in layers),
            block_diagonal_stable_rank=divide(fro2, spectral*spectral),
            block_diagonal_nuclear_participation_rank=divide(nuc*nuc, fro2),
            matrix_nuclear_sum=nuc, matrix_frobenius=math.sqrt(fro2),
            truncated_block_polar_alignment_scale=divide(
                sum(v['retained_nuclear'] for v in layers), math.sqrt(rank)),
        )
    return result, layers


def group_counts(logits, labels, minority):
    correct = logits.detach().argmax(dim=1).eq(labels)
    return [int((correct & minority).sum()), int(minority.sum()),
            int((correct & ~minority).sum()), int((~minority).sum())]


def accuracy_fields(counts):
    c0, n0, c1, n1 = counts
    return dict(s0_accuracy=100*c0/n0 if n0 else None,
                s1_accuracy=100*c1/n1 if n1 else None, s0_count=n0, s1_count=n1)


def evaluate(model, loader, device, gradient=False):
    """Exact sample-weighted full empirical gradient, holding BN buffers fixed."""
    was_training = model.training
    model.eval()
    if gradient:
        model.zero_grad(set_to_none=True)
    total_loss = 0.0
    counts = [0, 0, 0, 0]
    n = len(loader.dataset)
    seen = 0
    try:
        with torch.set_grad_enabled(gradient):
            for x, y, minority in loader:
                x, y, minority = x.to(device), y.to(device), minority.to(device)
                logits = model(x)
                loss_sum = nn.functional.cross_entropy(logits, y, reduction='sum')
                if not bool(torch.isfinite(loss_sum)):
                    raise FloatingPointError('Non-finite full-dataset loss')
                if gradient:
                    (loss_sum / n).backward()
                total_loss += float(loss_sum.detach())
                counts = [a+b for a, b in zip(counts, group_counts(logits, y, minority))]
                seen += y.numel()
    finally:
        model.train(was_training)
    if seen != n or n == 0:
        raise ValueError('Evaluation must cover the full, nonempty dataset exactly once')
    return dict(loss=total_loss/n, **accuracy_fields(counts))


def make_optimizers(name, model, args):
    from optimizers.muon import Muon
    params = [p for p in model.parameters() if p.requires_grad]
    # Match the repository: ONLY convolution weights use Muon, not classifier.
    matrix_ids = {id(m.weight) for m in model.modules() if isinstance(m, nn.Conv2d)}
    if name == 'AdamW':
        opts = [torch.optim.AdamW(params, lr=args.adam_lr, betas=(0., 0.),
                                 eps=args.adam_eps, weight_decay=args.adam_wd)]
    elif name == 'SGD':
        opts = [torch.optim.SGD(params, lr=args.sgd_lr, momentum=0., weight_decay=args.sgd_wd)]
    else:
        matrices = [p for p in params if id(p) in matrix_ids]
        auxiliary = [p for p in params if id(p) not in matrix_ids]
        opts = [Muon(matrices, lr=args.muon_lr, momentum=0., nesterov=True),
                torch.optim.AdamW(auxiliary, lr=args.muon_aux_lr, betas=(0., 0.),
                                  eps=args.adam_eps, weight_decay=args.muon_aux_wd)]
    assigned = [id(p) for opt in opts for group in opt.param_groups for p in group['params']]
    if len(assigned) != len(set(assigned)) or set(assigned) != {id(p) for p in params}:
        raise RuntimeError('Every trainable parameter must belong to exactly one optimizer')
    # Compare support on every trajectory; expensive matrix ranks only on Muon.
    return opts, matrix_ids if name == 'Muon' else set()


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for t in tensors:
        a = t.detach().cpu().contiguous().numpy()
        digest.update(str((a.shape, a.dtype)).encode())
        digest.update(memoryview(a).cast('B'))
    return digest.hexdigest()


def fit(model, train_dataset, test_dataset, name, args, run_dir):
    device = next(model.parameters()).device
    seed_all(args.run_seed)  # Paired dropout draws; evaluation consumes a separate RNG.
    train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                       drop_last=True, generator=torch.Generator().manual_seed(args.run_seed))
    evaluation_rng = torch.Generator().manual_seed(args.run_seed + 1)
    train_eval = DataLoader(train_dataset, batch_size=args.eval_batch_size,
                            shuffle=False, generator=evaluation_rng)
    test_eval = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                           shuffle=False, generator=evaluation_rng)
    if not len(train):
        raise ValueError('Batch size exceeds retained training set')
    opts, matrix_ids = make_optimizers(name, model, args)
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=args.epochs) for o in opts]
    dump_json(run_dir / 'optimizer_settings.json', [
        {'optimizer': type(o).__name__, 'groups': [
            {k: v for k, v in g.items() if k != 'params'} for g in o.param_groups]} for o in opts])
    logs = {key: CSVLog(run_dir / (key + '.csv')) for key in ['steps', 'epochs', 'matrix_layers']}
    history, prefix, step = [], {}, 0
    total_steps = args.epochs * len(train)

    def record_geometry(scope, epoch):
        values, layers = geometry(model, matrix_ids, args)
        for row in layers:
            logs['matrix_layers'].write(dict(scope=scope, epoch=epoch, step=step, **row))
        return values

    def record_epoch(epoch):
        tr = evaluate(model, train_eval, device, gradient=True)
        values = record_geometry('full_train_eval_mode', epoch)
        model.zero_grad(set_to_none=True)
        te = evaluate(model, test_eval, device)
        row = dict(epoch=epoch, step=step, **{'train_'+k: v for k, v in tr.items()},
                   **{'test_'+k: v for k, v in te.items()}, **values)
        logs['epochs'].write(row)
        history.append(row)
        print(f'{args.run_arch} {name} seed={args.run_seed} epoch={epoch}/{args.epochs} '
              f'train S0/S1={tr["s0_accuracy"]:.2f}/{tr["s1_accuracy"]:.2f}% '
              f'support={values["support_nonzero"]}', flush=True)

    try:
        record_epoch(0)
        for epoch in range(1, args.epochs+1):
            model.train()
            for batch_index, (x, y, minority) in enumerate(train, 1):
                step += 1
                x, y, minority = x.to(device), y.to(device), minority.to(device)
                model.zero_grad(set_to_none=True)
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f'Non-finite loss at step {step}')
                loss.backward()
                lr = opts[0].param_groups[0]['lr']
                sampled = step == 1 or step % args.diagnostic_every == 0 or step == total_steps
                if sampled:
                    values = record_geometry('minibatch_train_mode_pre_update', epoch)
                    for key in ['sqrt_support_nonzero', 'sqrt_support_thresholded', 'sqrt_matrix_rank_sum']:
                        if key in values:
                            accum = prefix.setdefault(key, [0., 0., 0., 0])
                            accum[0] += values[key]
                            accum[1] += lr*values[key]
                            accum[2] += lr
                            accum[3] += 1
                            values[key+'_sampled_prefix_mean'] = accum[0]/accum[3]
                            values[key+'_sampled_lr_weighted_prefix_mean'] = divide(accum[1], accum[2])
                    last_keys = list(values)
                else:
                    values = {k: None for k in last_keys}
                # Accuracy and loss refer to this same minibatch BEFORE the update.
                logs['steps'].write(dict(epoch=epoch, batch=batch_index, step=step, lr=lr,
                    auxiliary_lr=opts[1].param_groups[0]['lr'] if len(opts) > 1 else None,
                    batch_loss=float(loss.detach()), **accuracy_fields(group_counts(logits, y, minority)),
                    geometry_sampled=sampled, **values))
                for opt in opts:
                    opt.step()
            record_epoch(epoch)
            for scheduler in schedulers:
                scheduler.step()
        torch.save({'model': model.state_dict(), 'optimizers': [o.state_dict() for o in opts],
                    'schedulers': [s.state_dict() for s in schedulers],
                    'epoch': args.epochs, 'step': step, 'args': vars(args)},
                   run_dir / 'final_checkpoint.pt')
    finally:
        for log in logs.values():
            log.close()
    return history


def summarize(root, histories, plots=True):
    """The independent unit is a seed, never a batch or a layer."""
    log = CSVLog(root / 'epoch_summary.csv')
    summaries = {}
    try:
        for (arch, name), runs in histories.items():
            rows = []
            for epoch in range(len(runs[0])):
                row = dict(arch=arch, optimizer=name, epoch=epoch, n_runs=len(runs))
                for key in runs[0][epoch]:
                    if key in ('epoch', 'step'):
                        continue
                    vals = [run[epoch][key] for run in runs if run[epoch][key] is not None]
                    row[key+'_mean'] = statistics.mean(vals) if vals else None
                    row[key+'_std'] = statistics.stdev(vals) if len(vals) > 1 else None
                rows.append(row)
            summaries[arch, name] = rows
        # Union schema: Muon has additional fields; they remain blank for others.
        fields = list(dict.fromkeys(k for rows in summaries.values() for row in rows for k in row))
        for rows in summaries.values():
            for row in rows:
                log.write({key: row.get(key) for key in fields})
    finally:
        log.close()
    if not plots:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for arch in dict.fromkeys(k[0] for k in summaries):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        panels = [('train_s0_accuracy', 'S0: minority cars, train accuracy (%)'),
                  ('train_s1_accuracy', 'S1: remaining examples, train accuracy (%)'),
                  ('sqrt_support_nonzero', 'Full-gradient sqrt(nonzero support)'),
                  ('sqrt_matrix_rank_sum', 'Muon: sqrt(sum of numerical block ranks)')]
        for ax, (metric, label) in zip(axes.flat, panels):
            drawn = False
            for (a, name), rows in summaries.items():
                if a != arch or metric+'_mean' not in rows[0]:
                    continue
                x = np.array([r['epoch'] for r in rows])
                y = np.array([r[metric+'_mean'] for r in rows])
                sd = np.array([r[metric+'_std'] or 0. for r in rows])
                ax.plot(x, y, label=name)
                ax.fill_between(x, y-sd, y+sd, alpha=.15)
                drawn = True
            ax.set(xlabel='Epoch', ylabel=label)
            ax.grid(alpha=.2)
            if drawn:
                ax.legend()
        fig.suptitle(f'{arch}: mean +/- sample SD across seeds; geometry in eval mode')
        fig.savefig(root / f'{arch}_epoch_curves.png', dpi=160)
        plt.close(fig)


def self_test():
    """Analytic diagnostics, zero cases, accumulation, BN isolation, no downloads."""
    from types import SimpleNamespace
    a = SimpleNamespace(svd_device='cpu', rank_atol=0., rank_rtol=1e-10)
    v = vector_stats([torch.tensor([0., -3., 4.])], 0., 0., 1e-8)
    assert v['support_nonzero'] == 2 and v['grad_l1'] == 7
    assert math.isclose(v['support_participation'], 49/25)
    m = matrix_stats(torch.diag(torch.tensor([3., 4., 0.])), a)
    assert m['numerical_rank'] == 2
    assert math.isclose(m['stable_rank'], 25/16)
    assert math.isclose(m['nuclear_participation_rank'], 49/25)
    z = matrix_stats(torch.zeros(2, 3), a)
    assert z['numerical_rank'] == z['stable_rank'] == z['nuclear_participation_rank'] == 0
    zv = vector_stats([torch.zeros(4)], 0., 0., 1e-8)
    assert zv['support_nonzero'] == zv['support_participation'] == 0
    assert matrix_stats(torch.ones(2, 3, 2, 2), a)['numerical_rank'] == 1
    threshold = vector_stats([torch.tensor([1., 1e-8]), torch.tensor([0.])], 0., 1e-6, 1e-8)
    assert threshold['support_nonzero'] == 2 and threshold['support_thresholded'] == 1
    # Sign cancellation: stats(mean g) is not mean(stats(g)).
    assert vector_stats([torch.tensor([1.])+torch.tensor([-1.])], 0., 0., 1e-8)['support_nonzero'] == 0
    seed_all(5)
    net = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 2)).double()
    ds = TensorDataset(torch.randn(5, 3, dtype=torch.float64),
                       torch.tensor([0, 1, 0, 1, 0]), torch.tensor([True, False, False, False, True]))
    buffers = {k: v.clone() for k, v in net.named_buffers()}
    net.train()
    result = evaluate(net, DataLoader(ds, batch_size=2), 'cpu', gradient=True)
    assert net.training and result['s0_count'] == 2 and result['s1_count'] == 3
    assert all(torch.equal(v, buffers[k]) for k, v in net.named_buffers())
    accumulated = [p.grad.clone() for p in net.parameters()]
    net.eval()
    net.zero_grad()
    nn.functional.cross_entropy(net(ds.tensors[0]), ds.tensors[1]).backward()
    for actual, p in zip(accumulated, net.parameters()):
        torch.testing.assert_close(actual, p.grad, rtol=1e-10, atol=1e-12)
    assert accuracy_fields([0, 0, 1, 2])['s0_accuracy'] is None
    print('Self-tests passed: support/rank identities, zero/threshold cases, group counts, '
          'full-gradient sample weighting, and BatchNorm buffer preservation.')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--repo_root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--seeds', type=int, nargs=3, default=[42, 43, 44])
    p.add_argument('--archs', nargs='+', choices=['resnet10', 'vgg19'], default=['resnet10', 'vgg19'])
    p.add_argument('--optimizers', nargs='+', choices=['AdamW', 'SGD', 'Muon'], default=['AdamW', 'SGD', 'Muon'])
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--eval_batch_size', type=int, default=128)
    p.add_argument('--remove_cars_pct', type=float, default=95.)
    p.add_argument('--device', default='auto', help='auto selects CUDA if available, otherwise CPU')
    p.add_argument('--svd_device', default='auto', help='auto, cpu, or cuda[:index]; SVD uses float64')
    p.add_argument('--diagnostic_every', type=int, default=1)
    p.add_argument('--support_atol', type=float, default=0.)
    p.add_argument('--support_rtol', type=float, default=1e-6)
    p.add_argument('--rank_atol', type=float, default=0.)
    p.add_argument('--rank_rtol', type=float, default=None)
    for key, default in dict(adam_lr=1e-3, adam_wd=1e-3, adam_eps=1e-8,
                             sgd_lr=.05, sgd_wd=1e-3, muon_lr=.02,
                             muon_aux_lr=1e-3, muon_aux_wd=1e-3).items():
        p.add_argument('--'+key, type=float, default=default)
    p.add_argument('--output_dir', type=Path)
    p.add_argument('--no_plots', action='store_true')
    p.add_argument('--self_test', action='store_true')
    p.add_argument('--smoke_test', action='store_true', help='1 epoch, 3 seeds, tiny synthetic data')
    a = p.parse_args()
    for key in ['epochs', 'batch_size', 'eval_batch_size', 'diagnostic_every']:
        if getattr(a, key) < 1:
            p.error(key+' must be positive')
    if len(set(a.seeds)) != 3 or any(s < 0 or s >= 2**32-1 for s in a.seeds):
        p.error('Provide exactly three distinct seeds in [0, 2**32-2]')
    if any(len(set(v)) != len(v) for v in [a.archs, a.optimizers]):
        p.error('Do not repeat architectures or optimizers')
    for key, value in vars(a).items():
        if isinstance(value, float) and (not math.isfinite(value) or value < 0):
            p.error(key+' must be finite and nonnegative')
    for key in ['adam_lr', 'sgd_lr', 'muon_lr', 'muon_aux_lr', 'adam_eps']:
        if getattr(a, key) <= 0:
            p.error(key+' must be positive')
    if not 0 <= a.remove_cars_pct <= 99.98:
        p.error('Removal must leave at least one car: use [0,99.98] percent')
    if a.device != 'auto' and torch.device(a.device).type not in ('cpu', 'cuda'):
        p.error('Use CPU or CUDA for reproducible float64 gradient diagnostics')
    if a.svd_device != 'auto' and torch.device(a.svd_device).type not in ('cpu', 'cuda'):
        p.error('SVD device must be CPU or CUDA')
    return a


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    repo = args.repo_root.resolve()
    if not all((repo / f).is_file() for f in ['config.py', 'data/cifar.py', 'models/cifar.py', 'optimizers/muon.py']):
        raise SystemExit('Put this script beside main.py or pass --repo_root /path/to/Fairness-Learning')
    sys.path.insert(0, str(repo))
    import config
    from models.cifar import get_cifar10_model
    if args.smoke_test:
        args.epochs, args.batch_size, args.eval_batch_size = 1, 4, 3
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    root = args.output_dir or Path('outputs') / ('additional_cifar10_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise SystemExit('Choose a new or empty output directory; existing results will not be overwritten')
    root.mkdir(parents=True, exist_ok=True)
    args.repo_root, args.output_dir = str(repo), str(root)
    source_files = ['config.py', 'data/cifar.py', 'models/cifar.py', 'optimizers/muon.py']
    settings = dict(arguments=vars(args).copy(), reference_commit=SOURCE_COMMIT,
        created_utc=datetime.now(timezone.utc).isoformat(), torch_version=torch.__version__,
        numpy_version=np.__version__, python_version=sys.version, device=str(device),
        gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        source_sha256={f: hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in source_files},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol=__doc__, synthetic_smoke_test=args.smoke_test,
        total_runs=len(args.archs)*len(args.optimizers)*3)
    dump_json(root / 'settings.json', settings)
    print(f'Output: {root}\nS0 = minority cars; S1 = other samples; 3 paired seeds.', flush=True)
    if args.smoke_test:
        print('SYNTHETIC SMOKE TEST ONLY: these are not CIFAR-10 results.', flush=True)
    histories = {}
    for seed in args.seeds:
        seed_all(seed)
        if args.smoke_test:
            train_ds = TensorDataset(torch.randn(8, 3, 32, 32), torch.arange(8) % 9,
                                     torch.tensor([True, False]*4))
            test_ds = train_ds
        else:
            from data.cifar import get_cifar10_dataloaders
            train_loader, _, test_loader = get_cifar10_dataloaders(args.batch_size, args.remove_cars_pct, seed)
            train_ds, test_ds = train_loader.dataset, test_loader.dataset
            del train_loader, test_loader
        counts = {'train_s0': int(train_ds.tensors[2].sum()),
                  'train_s1': int((~train_ds.tensors[2]).sum()),
                  'test_s0': int(test_ds.tensors[2].sum()),
                  'test_s1': int((~test_ds.tensors[2]).sum())}
        if min(counts.values()) <= 0:
            raise ValueError(f'Both groups must be nonempty: {counts}')
        dump_json(root / f'data_seed_{seed}.json', dict(**counts,
            train_sha256=tensor_digest(train_ds.tensors), test_sha256=tensor_digest(test_ds.tensors)))
        for arch in args.archs:
            seed_all(seed)
            base = get_cifar10_model(arch, torch.device('cpu'))
            config.initialize_weights(base, num_classes=9)
            state = copy.deepcopy(base.state_dict())
            args.run_arch, args.run_seed = arch, seed
            for name in args.optimizers:
                run_dir = root / arch / name / f'seed_{seed}'
                run_dir.mkdir(parents=True)
                model = copy.deepcopy(base)
                model.load_state_dict(state)
                model.to(device)
                dump_json(run_dir / 'initial_state.json', {'sha256': tensor_digest(model.state_dict().values())})
                history = fit(model, train_ds, test_ds, name, args, run_dir)
                histories.setdefault((arch, name), []).append(history)
                del model
            del base, state
        del train_ds, test_ds
    summarize(root, histories, plots=not args.no_plots)
    print(f'Completed all runs. Summary: {root / "epoch_summary.csv"}', flush=True)


if __name__ == '__main__':
    main()
