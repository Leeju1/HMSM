"""Optional graph-level hyperparameter search and repeated-CV epoch analysis.

Apply selected settings to the configuration before running the next stage
or training; this script does not update settings or train a final model.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
import json
from typing import Any, Dict, List, Tuple, Sequence
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from embedding.model import (DiGAEModel, build_tilde_A, normalize_hatA,
    row_normalize_target, valid_rows_from_target, kl_rowsum)
from embedding.train import set_seed, _preprocess, _subset_data, _train_loop, GraphData, validate_training_config
from embedding.graph_io import (load_config, resolve_path, load_posterior_mean_graphs,
    prepare_output, save_json, configure_device, provenance, configured_names)


def run_tuning(cfg, stage, graphs_dir=None, out_dir=None, device=None):
    validate_training_config(cfg)
    device = configure_device(cfg, device)
    pm_dir = resolve_path(graphs_dir or cfg['pm_dir'])
    graphs, names, nodes = load_posterior_mean_graphs(
        pm_dir, cfg['N'], device, cfg['pm_filename'], cfg['expected_graphs'], configured_names(cfg))
    settings = cfg['grid'] if stage == 'grid' else cfg['epoch_cv']
    if stage == 'grid':
        dims = settings['d_grid']
        if not dims or len(set(dims)) != len(dims) or any(not isinstance(d, int) or d < 1 for d in dims):
            raise ValueError('d_grid must contain unique positive integers')
        for key in ('alpha_grid', 'beta_grid'):
            if not settings[key] or len(set(settings[key])) != len(settings[key]) or any(not np.isfinite(v) or v < 0 for v in settings[key]):
                raise ValueError(f'Invalid {key}')
        for d in dims:
            item = settings['per_d_config'][str(d)]
            if any(not isinstance(item[k], int) or item[k] < 1 for k in ('hidden_dim', 'epochs')) or not np.isfinite(item['lr']) or item['lr'] <= 0:
                raise ValueError(f'Invalid grid settings for d={d}')
        if not 2 <= settings['k_folds'] <= len(graphs) or settings['eval_interval'] < 1:
            raise ValueError('Invalid grid fold count or evaluation interval')
    else:
        if any(not isinstance(settings[k], int) or settings[k] < 1
               for k in ('budget_epochs', 'cv_repeats', 'eval_every')):
            raise ValueError('Epoch CV budgets, repeats and intervals must be positive integers')
        if not 2 <= settings['cv_folds'] <= len(graphs):
            raise ValueError('Invalid epoch CV fold count')
    out = prepare_output(resolve_path(out_dir or str(Path(cfg['tuning_dir']) / stage)))
    metadata = provenance(dict(cfg, stage=stage, pm_dir=str(pm_dir), out_dir=str(out)),
                          [pm_dir / n / cfg['pm_filename'] for n in names])
    save_json(out / 'run_metadata.json', dict(metadata, status='running'))
    if stage == 'grid':
        grid_search_with_kfold(graphs, names, settings['alpha_grid'], settings['beta_grid'],
            settings['d_grid'], {int(k): v for k, v in settings['per_d_config'].items()},
            cfg['weight_decay'], settings['k_folds'], settings['eval_interval'],
            device, str(out), cfg['seed'], dropout=cfg['dropout'])
    else:
        data = _preprocess(graphs, cfg['alpha'], cfg['beta'], device)
        runs = run_repeated_kfold_cv(data, names, cfg['N'], cfg['hidden_dim'], cfg['out_dim'],
            cfg['dropout'], cfg['lr'], cfg['weight_decay'], settings['budget_epochs'],
            settings['eval_every'], settings['cv_folds'], settings['cv_repeats'], cfg['seed'], device)
        aggregate = aggregate_cv_histories(runs)
        one_se = select_epoch_one_se(aggregate, runs, settings['budget_epochs'])
        selection = dict(one_se, one_se_epoch=one_se['selected_epoch'],
                         selected_epoch=one_se['argmin_epoch'],
                         rule='Minimum mean held-out KL across repeated graph-level CV runs.',
                         configured_final_epochs=cfg['final_epochs'])
        save_json(out / 'epoch_cv_history.json', runs)
        save_json(out / 'epoch_selection.json', selection)
        pd.DataFrame(aggregate).to_csv(out / 'epoch_cv_summary.csv', index=False)
        plot_stage1_aggregate(aggregate, selection, str(out), settings['cv_folds'], settings['cv_repeats'])
        if cfg.get('diagnostic_plots', False):
            plot_stage1_individual_runs(runs, aggregate, str(out))
        print(f'Mean-loss argmin: {selection["selected_epoch"]}; one-SE: {selection["one_se_epoch"]}')
    save_json(out / 'run_metadata.json', dict(metadata, status='complete'))
    print('Tuning complete. Review the results and explicitly set the final training configuration.')

def _preprocess_graphs(
    graphs: List[torch.Tensor],
    alpha: float,
    beta: float,
    device: torch.device,
):
    N = graphs[0].size(0)
    X_list, A_hat_list, A_tgt_list = [], [], []
    valid_nodes_list, pm_mask_list = [], []
    for A in graphs:
        X_list.append(torch.eye(N, dtype=torch.float32, device=device))
        A_tilde = build_tilde_A(A)
        A_hat_list.append(normalize_hatA(A_tilde, alpha=alpha, beta=beta))
        A_tgt = row_normalize_target(A, zero_diag=True)
        A_tgt_list.append(A_tgt)
        valid = valid_rows_from_target(A_tgt)
        if not valid.any():
            raise ValueError("Graph has no outgoing target mass")
        valid_nodes_list.append(valid)
        pm_mask_list.append(A > 0)
    return X_list, A_hat_list, A_tgt_list, valid_nodes_list, pm_mask_list


def _mean_kl(model, X_list, A_hat_list, A_tgt_list,
             valid_nodes_list, pm_mask_list) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(len(X_list)):
            A_bar, _, _ = model(X_list[i], A_hat_list[i],
                                pm_mask=pm_mask_list[i])
            kl = kl_rowsum(A_tgt_list[i], A_bar,
                           valid_nodes=valid_nodes_list[i])
            total += float(kl.item())
    value = total / len(X_list)
    if not np.isfinite(value):
        raise ValueError("Evaluation produced non-finite loss")
    return value


def train_with_val_tracking(
    train_graphs: List[torch.Tensor],
    val_graphs: List[torch.Tensor],
    hidden_dim: int,
    out_dim: int,
    alpha: float,
    beta: float,
    lr: float,
    weight_decay: float,
    budget_epochs: int,
    eval_interval: int,
    device: torch.device,
    dropout: float = 0.0,
) -> Dict:
    N = train_graphs[0].size(0)

    (Xtr, Ahat_tr, Atgt_tr,
     vn_tr, pm_tr) = _preprocess_graphs(train_graphs, alpha, beta, device)
    (Xva, Ahat_va, Atgt_va,
     vn_va, pm_va) = _preprocess_graphs(val_graphs, alpha, beta, device)

    model = DiGAEModel(
        in_dim=N,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        dropout=dropout,
        prevent_self_loop=True
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    history = {'epoch': [], 'train_kl': [], 'val_kl': []}
    best_val = float('inf')
    best_epoch = -1
    train_at_best = float('nan')

    for epoch in range(1, budget_epochs + 1):
        model.train()
        opt.zero_grad()

        total_loss = 0.0
        for i in range(len(train_graphs)):
            A_bar, _, _ = model(Xtr[i], Ahat_tr[i], pm_mask=pm_tr[i])
            loss = kl_rowsum(Atgt_tr[i], A_bar, valid_nodes=vn_tr[i])
            total_loss = total_loss + loss

        avg_loss = total_loss / len(train_graphs)
        if not torch.isfinite(avg_loss):
            raise ValueError(f"Non-finite training loss at epoch {epoch}")
        avg_loss.backward()
        opt.step()

        if epoch == 1 or epoch % eval_interval == 0 or epoch == budget_epochs:
            train_kl = _mean_kl(model, Xtr, Ahat_tr, Atgt_tr, vn_tr, pm_tr)
            val_kl = _mean_kl(model, Xva, Ahat_va, Atgt_va, vn_va, pm_va)
            history['epoch'].append(epoch)
            history['train_kl'].append(train_kl)
            history['val_kl'].append(val_kl)
            if val_kl < best_val:
                best_val = val_kl
                best_epoch = epoch
                train_at_best = train_kl

    return {
        'best_val': best_val,
        'best_epoch': best_epoch,
        'train_at_best': train_at_best,
        'final_val': history['val_kl'][-1],
        'final_train': history['train_kl'][-1],
        'history': history,
    }


def select_one_se(results: Dict, k_folds: int) -> Tuple:
    best_key = min(results.keys(), key=lambda k: results[k]['mean_val'])
    best = results[best_key]
    threshold = best['mean_val'] + best['std_val'] / np.sqrt(k_folds)

    within = [k for k, v in results.items() if v['mean_val'] <= threshold]
    min_d = min(k[2] for k in within)
    candidates = [k for k in within if k[2] == min_d]
    one_se_key = min(candidates, key=lambda k: results[k]['mean_val'])

    return best_key, one_se_key, float(threshold)


def grid_search_with_kfold(
    graphs: List[torch.Tensor],
    param_names: List[str],
    alpha_grid: List[float],
    beta_grid: List[float],
    d_grid: List[int],
    per_d_config: Dict[int, Dict],
    weight_decay: float,
    k_folds: int,
    eval_interval: int,
    device: torch.device,
    out_dir: str,
    seed: int = 42,
    dropout: float = 0.0,
) -> Tuple[Dict, Tuple, Tuple]:
    set_seed(seed)

    n_graphs = len(graphs)
    if n_graphs != 28:
        print(f"WARNING:  Warning: Expected 28 graphs, got {n_graphs}")

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    indices = list(range(n_graphs))
    folds = list(kf.split(indices))

    os.makedirs(out_dir, exist_ok=True)
    grid_histories = {}

    results = {}
    total_combinations = len(alpha_grid) * len(beta_grid) * len(d_grid)
    current = 0

    print(f"\n{'='*70}")
    print(f"Grid Search Configuration")
    print(f"{'='*70}")
    print(f"  Total graphs      : {n_graphs}")
    print(f"  K-folds           : {k_folds}")
    print(f"  Eval interval     : every {eval_interval} epochs")
    print(f"  Alpha grid        : {alpha_grid}")
    print(f"  Beta grid         : {beta_grid}")
    print(f"  D grid            : {d_grid}")
    for d in d_grid:
        c = per_d_config[d]
        print(f"    d={d:<4d} -> hidden={c['hidden_dim']}, "
              f"lr={c['lr']}, budget={c['epochs']} epochs")
    print(f"  Fold score        : min val KL within budget "
          f"(pre-specified rule; not final-epoch KL)")
    print(f"  Total combinations: {total_combinations}")
    print(f"{'='*70}\n")

    for d in d_grid:
        cfg = per_d_config[d]
        budget = cfg['epochs']
        for alpha in alpha_grid:
            for beta in beta_grid:
                current += 1
                param_key = (alpha, beta, d)

                print(f"\n{'='*70}")
                print(f"[{current}/{total_combinations}] "
                      f"Testing alpha={alpha:.2f}, beta={beta:.2f}, d={d} "
                      f"(hidden={cfg['hidden_dim']}, lr={cfg['lr']}, "
                      f"budget={budget})")
                print(f"{'='*70}")

                fold_scores, fold_train_scores = [], []
                fold_best_epochs = []
                fold_histories = {}

                for fold_idx, (train_idx, val_idx) in enumerate(folds):
                    print(f"\n   Fold {fold_idx + 1}/{k_folds}")
                    print(f"     Train graphs: {len(train_idx)}")
                    print(f"     Val graphs  : {len(val_idx)} "
                          f"({[param_names[i] for i in val_idx]})")

                    train_graphs = [graphs[i] for i in train_idx]
                    val_graphs = [graphs[i] for i in val_idx]

                    out = train_with_val_tracking(
                        train_graphs=train_graphs,
                        val_graphs=val_graphs,
                        hidden_dim=cfg['hidden_dim'],
                        out_dim=d,
                        alpha=alpha,
                        beta=beta,
                        lr=cfg['lr'],
                        weight_decay=weight_decay,
                        budget_epochs=budget,
                        eval_interval=eval_interval,
                        device=device,
                        dropout=dropout,
                    )

                    fold_scores.append(out['best_val'])
                    fold_train_scores.append(out['train_at_best'])
                    fold_best_epochs.append(out['best_epoch'])
                    fold_histories[f"fold_{fold_idx + 1}"] = out['history']

                    print(f"     Best val KL : {out['best_val']:.6f} "
                          f"@ epoch {out['best_epoch']} "
                          f"(final-epoch val KL: {out['final_val']:.6f})")
                    print(f"     Train KL @ best epoch: "
                          f"{out['train_at_best']:.6f}")

                near_budget = [e >= 0.9 * budget for e in fold_best_epochs]
                frac_at_budget = float(np.mean(near_budget))

                mean_val = float(np.mean(fold_scores))
                std_val = float(np.std(fold_scores, ddof=1))
                mean_train = float(np.mean(fold_train_scores))

                results[param_key] = {
                    'mean_val': mean_val,
                    'std_val': std_val,
                    'mean_train': mean_train,
                    'folds_val': [float(s) for s in fold_scores],
                    'folds_train': [float(s) for s in fold_train_scores],
                    'folds_best_epoch': [int(e) for e in fold_best_epochs],
                    'median_best_epoch': float(np.median(fold_best_epochs)),
                    'frac_folds_at_budget': frac_at_budget,
                    'hidden_dim': cfg['hidden_dim'],
                    'lr': cfg['lr'],
                    'budget_epochs': budget,
                }

                grid_histories[f"alpha{alpha}_beta{beta}_d{d}"] = {
                    'alpha': alpha, 'beta': beta, 'd': d,
                    'budget_epochs': budget,
                    'eval_interval': eval_interval,
                    'histories': fold_histories,
                    'folds_best_epoch': [int(e) for e in fold_best_epochs],
                }

                print(f"\n   Summary for alpha={alpha:.2f}, "
                      f"beta={beta:.2f}, d={d}:")
                print(f"     Mean Train KL (@best): {mean_train:.6f}")
                print(f"     Mean Val KL (min)    : {mean_val:.6f} "
                      f"+/- {std_val:.6f}")
                print(f"     Best epochs by fold  : {fold_best_epochs}")
                if frac_at_budget > 0.4:
                    print(f"     WARNING:  {frac_at_budget:.0%} of folds hit the "
                          f"last 10% of the budget - budget may be "
                          f"insufficient for this configuration")

    best_params, best_params_1se, threshold = select_one_se(results, k_folds)

    print(f"\n{'='*70}")
    print(f" Grid Search Complete!")
    print(f"{'='*70}")
    print(f"  [Overall best by mean val KL]")
    print(f"    alpha={best_params[0]:.2f}, beta={best_params[1]:.2f}, "
          f"d={best_params[2]}")
    print(f"    Val KL: {results[best_params]['mean_val']:.6f} "
          f"+/- {results[best_params]['std_val']:.6f}")
    print(f"  [1-SE rule selection]")
    print(f"    threshold = best mean + SE = {threshold:.6f}")
    print(f"    alpha={best_params_1se[0]:.2f}, beta={best_params_1se[1]:.2f}, "
          f"d={best_params_1se[2]}")
    print(f"    Val KL: {results[best_params_1se]['mean_val']:.6f} "
          f"+/- {results[best_params_1se]['std_val']:.6f}")
    if best_params_1se != best_params:
        print(f"    -> smaller d selected within one SE of the best")
    else:
        print(f"    -> coincides with the overall best")

    saturated = [(k, v['frac_folds_at_budget']) for k, v in results.items()
                 if v['frac_folds_at_budget'] > 0.4]
    if saturated:
        print(f"\n  WARNING:  Budget saturation flagged for "
              f"{len(saturated)}/{len(results)} configurations "
              f"(best epoch in last 10% of budget for >40% of folds).")
        print(f"      Consider increasing grid.per_d_config[d].epochs for the affected d "
              f"and re-running.")
    print(f"{'='*70}")

    results_json = {
        f"alpha{k[0]}_beta{k[1]}_d{k[2]}": v
        for k, v in results.items()
    }

    results_path = os.path.join(out_dir, "grid_search_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'results': results_json,
            'best_params': {
                'alpha': best_params[0],
                'beta': best_params[1],
                'd': best_params[2]
            },
            'best_scores': results[best_params],
            'best_params_1se': {
                'alpha': best_params_1se[0],
                'beta': best_params_1se[1],
                'd': best_params_1se[2]
            },
            'best_scores_1se': results[best_params_1se],
            'one_se_threshold': threshold,
            'selection_rule': (
                "Pre-specified two-part rule. (i) Checkpoint within each "
                "fold: each configuration is trained for a fixed epoch "
                "budget, the held-out KL is evaluated every eval_interval "
                "epochs, and the minimum observed held-out KL (not the "
                "final-epoch KL) is taken as the fold score, so that "
                "configurations with different convergence speeds are "
                "compared at their respective best checkpoints within their "
                "configured budgets. (ii) Configuration selection: 1-SE rule "
                "- among configurations whose mean fold score is within "
                "one standard error (std_val/sqrt(K), std with ddof=1) of "
                "the overall best, select the smallest embedding dimension "
                "d, then the minimum mean fold score within that d."
            ),
            'config': {
                'n_graphs': n_graphs,
                'k_folds': k_folds,
                'eval_interval': eval_interval,
                'alpha_grid': alpha_grid,
                'beta_grid': beta_grid,
                'd_grid': d_grid,
                'per_d_config': {str(d): per_d_config[d] for d in d_grid},
                'seed': seed,
                'dropout': dropout
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n Results saved to: {results_path}")
    history_path = os.path.join(out_dir, 'grid_cv_history.json')
    save_json(history_path, grid_histories)
    print(f" Training histories saved to: {history_path}")

    summary_data = []
    for params, scores in results.items():
        row = {
            'alpha': params[0],
            'beta': params[1],
            'd': params[2],
            'hidden_dim': scores['hidden_dim'],
            'lr': scores['lr'],
            'budget_epochs': scores['budget_epochs'],
            'median_best_epoch': scores['median_best_epoch'],
            'frac_folds_at_budget': scores['frac_folds_at_budget'],
            'mean_train_kl': scores['mean_train'],
            'mean_val_kl': scores['mean_val'],
            'std_val_kl': scores['std_val'],
        }
        for i, val_kl in enumerate(scores['folds_val']):
            row[f'fold_{i+1}_val'] = val_kl
        for i, be in enumerate(scores['folds_best_epoch']):
            row[f'fold_{i+1}_best_epoch'] = be
        summary_data.append(row)

    df_summary = pd.DataFrame(summary_data).sort_values('mean_val_kl')

    summary_path = os.path.join(out_dir, "grid_search_summary.csv")
    df_summary.to_csv(summary_path, index=False, float_format='%.6f')
    print(f" Summary table saved to: {summary_path}")

    print(f"\n Top 5 Configurations:")
    print(df_summary[['alpha', 'beta', 'd', 'budget_epochs',
                      'median_best_epoch', 'mean_val_kl', 'std_val_kl']]
          .head(5).to_string(index=False))

    return results, best_params, best_params_1se

def make_repeated_kfold_splits(
    n_items: int,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if n_splits < 2:
        raise ValueError("cv_folds must be >= 2")
    if n_splits > n_items:
        raise ValueError("cv_folds cannot exceed the number of graphs")
    if n_repeats < 1:
        raise ValueError("cv_repeats must be >= 1")

    splits: List[Dict[str, Any]] = []
    all_indices = np.arange(n_items, dtype=int)

    for repeat_idx in range(n_repeats):
        split_seed = seed + repeat_idx
        rng = np.random.RandomState(split_seed)
        permuted = rng.permutation(all_indices)
        folds = [
            np.asarray(x, dtype=int)
            for x in np.array_split(permuted, n_splits)
        ]

        for fold_idx, val_idx in enumerate(folds):
            train_idx = np.concatenate(
                [folds[j] for j in range(n_splits) if j != fold_idx]
            )
            splits.append(
                {
                    "repeat": repeat_idx + 1,
                    "fold": fold_idx + 1,
                    "split_seed": split_seed,
                    "train_idx": train_idx.tolist(),
                    "val_idx": val_idx.tolist(),
                }
            )

    return splits


def run_repeated_kfold_cv(
    all_data: Sequence[GraphData],
    param_names: Sequence[str],
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    budget_epochs: int,
    eval_every: int,
    cv_folds: int,
    cv_repeats: int,
    seed: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    splits = make_repeated_kfold_splits(
        n_items=len(all_data),
        n_splits=cv_folds,
        n_repeats=cv_repeats,
        seed=seed,
    )

    runs: List[Dict[str, Any]] = []
    n_runs = len(splits)

    for run_idx, split in enumerate(splits, start=1):
        repeat = int(split["repeat"])
        fold = int(split["fold"])
        train_idx = [int(i) for i in split["train_idx"]]
        val_idx = [int(i) for i in split["val_idx"]]

        model_seed = seed + 10_000 + (repeat - 1) * cv_folds + (fold - 1)
        set_seed(model_seed)

        train_data = _subset_data(all_data, train_idx)
        val_data = _subset_data(all_data, val_idx)

        print(
            f"\n CV run {run_idx:02d}/{n_runs} "
            f"(repeat {repeat}/{cv_repeats}, fold {fold}/{cv_folds}, "
            f"train={len(train_idx)}, val={len(val_idx)}, "
            f"model_seed={model_seed})"
        )
        print(f"   Val graphs: {[param_names[i] for i in val_idx]}")

        model = DiGAEModel(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
            prevent_self_loop=True,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        history = _train_loop(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            epochs=budget_epochs,
            eval_every=eval_every,
            val_data=val_data,
            log_prefix=f"   [R{repeat:02d} F{fold:02d}] ",
        )

        runs.append(
            {
                "run": run_idx,
                "repeat": repeat,
                "fold": fold,
                "split_seed": int(split["split_seed"]),
                "model_seed": model_seed,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "train_idx": train_idx,
                "val_idx": val_idx,
                "train_names": [param_names[i] for i in train_idx],
                "val_names": [param_names[i] for i in val_idx],
                "history": history,
            }
        )

        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return runs


def aggregate_cv_histories(
    cv_runs: Sequence[Dict[str, Any]],
) -> List[Dict[str, float]]:
    if not cv_runs:
        raise ValueError("cv_runs must not be empty")

    reference_epochs = [int(h["epoch"]) for h in cv_runs[0]["history"]]
    if not reference_epochs:
        raise ValueError("CV histories are empty")

    train_rows: List[List[float]] = []
    val_rows: List[List[float]] = []

    for run in cv_runs:
        history = run["history"]
        epochs = [int(h["epoch"]) for h in history]
        if epochs != reference_epochs:
            raise ValueError(
                "CV histories use different evaluation grids. Keep "
                "budget_epochs and eval_every identical across runs."
            )
        train_rows.append([float(h["train_loss"]) for h in history])
        val_rows.append([float(h["val_loss"]) for h in history])

    train_matrix = np.asarray(train_rows, dtype=float)
    val_matrix = np.asarray(val_rows, dtype=float)
    n_runs = train_matrix.shape[0]
    ddof = 1 if n_runs > 1 else 0

    train_mean = train_matrix.mean(axis=0)
    train_sd = train_matrix.std(axis=0, ddof=ddof)
    train_se = train_sd / np.sqrt(n_runs)
    val_mean = val_matrix.mean(axis=0)
    val_sd = val_matrix.std(axis=0, ddof=ddof)
    val_se = val_sd / np.sqrt(n_runs)

    aggregated: List[Dict[str, float]] = []
    for i, epoch in enumerate(reference_epochs):
        aggregated.append(
            {
                "epoch": int(epoch),
                "n_cv_runs": int(n_runs),
                "train_mean": float(train_mean[i]),
                "train_sd": float(train_sd[i]),
                "train_se": float(train_se[i]),
                "val_mean": float(val_mean[i]),
                "val_sd": float(val_sd[i]),
                "val_se": float(val_se[i]),
            }
        )

    return aggregated


def select_epoch_one_se(
    aggregate_history: Sequence[Dict[str, float]],
    cv_runs: Sequence[Dict[str, Any]],
    budget_epochs: int,
) -> Dict[str, Any]:
    if not aggregate_history:
        raise ValueError("aggregate_history must not be empty")

    epochs = np.asarray([h["epoch"] for h in aggregate_history], dtype=int)
    val_mean = np.asarray([h["val_mean"] for h in aggregate_history], dtype=float)
    val_se = np.asarray([h["val_se"] for h in aggregate_history], dtype=float)

    argmin_idx = int(np.argmin(val_mean))
    argmin_epoch = int(epochs[argmin_idx])
    min_mean_val_loss = float(val_mean[argmin_idx])
    se_at_min = float(val_se[argmin_idx])
    threshold = min_mean_val_loss + se_at_min

    eligible = np.flatnonzero(val_mean <= threshold)
    if eligible.size == 0:

        raise RuntimeError("No epoch satisfies the 1-SE threshold.")
    selected_epoch = int(epochs[int(eligible[0])])

    per_run_argmin_epochs: List[int] = []
    for run in cv_runs:
        history = run["history"]
        run_val = np.asarray([h["val_loss"] for h in history], dtype=float)
        run_epochs = np.asarray([h["epoch"] for h in history], dtype=int)
        per_run_argmin_epochs.append(int(run_epochs[int(np.argmin(run_val))]))

    late_cutoff = 0.9 * budget_epochs
    late_run_fraction = float(
        np.mean(np.asarray(per_run_argmin_epochs, dtype=float) >= late_cutoff)
    )
    aggregate_saturation = bool(argmin_epoch >= late_cutoff)

    return {
        "selected_epoch": selected_epoch,
        "argmin_epoch": argmin_epoch,
        "min_mean_val_loss": min_mean_val_loss,
        "se_at_min": se_at_min,
        "one_se_threshold": float(threshold),
        "n_cv_runs": int(len(cv_runs)),
        "per_run_argmin_epochs": per_run_argmin_epochs,
        "per_run_argmin_median": float(np.median(per_run_argmin_epochs)),
        "per_run_argmin_min": int(np.min(per_run_argmin_epochs)),
        "per_run_argmin_max": int(np.max(per_run_argmin_epochs)),
        "late_run_fraction": late_run_fraction,
        "saturation_warning": aggregate_saturation,
        "rule": (
            "Repeated graph-level K-fold one-standard-error rule. At each "
            "evaluation epoch, validation KL losses are averaged across all "
            "fold/repeat runs. Let e_min minimize that mean curve. The final "
            "epoch count is the earliest evaluated epoch whose mean "
            "validation loss is no greater than mean(e_min) plus the "
            "empirical standard error across CV runs at e_min."
        ),
    }


def plot_stage1_aggregate(
    aggregate_history: Sequence[Dict[str, float]],
    selection: Dict[str, Any],
    out_dir: str,
    cv_folds: int,
    cv_repeats: int,
) -> str:
    epochs = np.asarray([h["epoch"] for h in aggregate_history])
    train_mean = np.asarray([h["train_mean"] for h in aggregate_history])
    train_se = np.asarray([h["train_se"] for h in aggregate_history])
    val_mean = np.asarray([h["val_mean"] for h in aggregate_history])
    val_se = np.asarray([h["val_se"] for h in aggregate_history])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_mean, lw=2, label="CV train mean")
    ax.fill_between(
        epochs,
        train_mean - train_se,
        train_mean + train_se,
        alpha=0.18,
        label="Train +/- 1 SE",
    )
    ax.plot(epochs, val_mean, lw=2, label="CV validation mean")
    ax.fill_between(
        epochs,
        val_mean - val_se,
        val_mean + val_se,
        alpha=0.18,
        label="Validation +/- 1 SE",
    )
    ax.axhline(
        selection["one_se_threshold"],
        linestyle=":",
        linewidth=1.4,
        label="1-SE threshold at validation minimum",
    )
    ax.axvline(
        selection["selected_epoch"],
        linestyle="--",
        linewidth=1.6,
        label=f"Selected epoch E={selection['selected_epoch']}",
    )
    ax.axvline(
        selection["one_se_epoch"],
        linestyle=":",
        linewidth=1.2,
        alpha=0.75,
        label=f"One-SE alternative={selection['one_se_epoch']}",
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("KL divergence loss", fontsize=12)
    ax.set_title(
        f"Epoch analysis: repeated {cv_folds}-fold CV "
        f"({cv_repeats} repeats, mean +/- SE)",
        fontsize=13,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, "epoch_cv_epoch_selection.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {path}")
    return path


def plot_stage1_individual_runs(
    cv_runs: Sequence[Dict[str, Any]],
    aggregate_history: Sequence[Dict[str, float]],
    out_dir: str,
) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))

    for run in cv_runs:
        epochs = [h["epoch"] for h in run["history"]]
        val_loss = [h["val_loss"] for h in run["history"]]
        ax.plot(epochs, val_loss, linewidth=1.0, alpha=0.25)

    mean_epochs = [h["epoch"] for h in aggregate_history]
    mean_val = [h["val_mean"] for h in aggregate_history]
    ax.plot(
        mean_epochs,
        mean_val,
        linewidth=2.5,
        label="Mean validation loss",
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation KL divergence loss", fontsize=12)
    ax.set_title("Epoch analysis: individual CV validation curves", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, "epoch_cv_individual_val_curves.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['cdtally', 'lamp_return'])
    parser.add_argument('--stage', required=True, choices=['grid', 'epochs'])
    parser.add_argument('--config', help='Optional JSON overrides')
    parser.add_argument('--graphs-dir')
    parser.add_argument('--out-dir')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'])
    args = parser.parse_args()
    run_tuning(load_config('embedding', args.dataset, args.config), args.stage,
               args.graphs_dir, args.out_dir, args.device)


if __name__ == '__main__':
    main()
