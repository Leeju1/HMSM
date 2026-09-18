"""Reconstruction diagnostics for posterior draws and mean training graphs.

These are reconstruction diagnostics, not held-out predictive performance.
AUROC uses raw logits before support masking; KL/JS/NDCG use the masked decoder.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
from typing import List, Dict
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from embedding.model import DiGAEModel, build_tilde_A, normalize_hatA
from embedding.graph_io import (load_config, resolve_path, load_model, read_graph,
    graph_names, sample_paths, load_embedding_record, load_embedding_summary, prepare_output, save_json,
    configure_device, provenance, configured_names, sha256)

EPS = 1e-12


def validate_draws(cfg, model_dir=None, graphs_dir=None, emb_dir=None, pm_dir=None,
                   out_dir=None, max_draws=None, device=None):
    device = configure_device(cfg, device)
    model_dir = resolve_path(model_dir or cfg['model_dir'])
    graphs_dir = resolve_path(graphs_dir or cfg['graphs_dir'])
    emb_dir = resolve_path(emb_dir or cfg['emb_dir'])
    pm_dir = resolve_path(pm_dir or cfg['pm_dir'])
    model, model_cfg = load_model(model_dir, device)
    N, nodes = model_cfg['N'], model_cfg['node_names']
    if N != cfg['N']:
        raise ValueError('Model and dataset node counts differ')
    names = graph_names(graphs_dir, configured_names(cfg), cfg['expected_graphs'])
    if names != model_cfg['param_names']:
        raise ValueError('Model and graph types differ')
    graph_names(pm_dir, names, len(names))
    graph_names(emb_dir, names, len(names))
    ids, graph_files = sample_paths(graphs_dir, names, expected_draws=cfg['expected_draws'], max_draws=max_draws)
    emb_ids, emb_files = sample_paths(emb_dir, names, kind='embedding',
        expected_draws=None if max_draws else cfg['expected_draws'], max_draws=max_draws)
    if ids != emb_ids:
        raise ValueError('Graph and embedding draw IDs differ')
    summary = load_embedding_summary(emb_dir, names, ids, cfg['dataset'], N)
    model_hash = sha256(model_dir / 'model.pt')
    if summary.get('model_sha256', model_hash) != model_hash:
        raise ValueError('Embedding summary checkpoint differs from the validation model')
    ks = cfg['ndcg_k']
    if not ks or any(not isinstance(k, int) or k < 1 or k > N for k in ks):
        raise ValueError('ndcg_k must contain positive integers no greater than N')
    out = prepare_output(resolve_path(out_dir or cfg['validation_dir']))
    metadata = provenance(dict(cfg, model_dir=str(model_dir), graphs_dir=str(graphs_dir),
        emb_dir=str(emb_dir), pm_dir=str(pm_dir), out_dir=str(out), max_draws=max_draws),
        [model_dir / 'model.pt', model_dir / 'train_config.json'] +
        [pm_dir / n / cfg['pm_filename'] for n in names])
    save_json(out / 'run_metadata.json', dict(metadata, status='running'))
    records = []
    for name in names:
        pm, _ = read_graph(pm_dir / name / cfg['pm_filename'], N, nodes)
        uniform, mask = build_uniform_target(pm), pm > 0
        for draw in ids:
            graph_path = graph_files[name][draw]
            a, _ = read_graph(graph_path, N, nodes)
            emb = load_embedding_record(emb_files[name][draw], nodes, model_cfg['out_dim'],
                graph=name, draw_id=draw, model_hash=summary.get('model_sha256'))
            if 'model_sha256' in emb and emb['model_sha256'] != model_hash:
                raise ValueError(f'Embedding checkpoint mismatch: {name}, draw {draw}')
            if 'graph_sha256' in emb and emb['graph_sha256'] != sha256(graph_path):
                raise ValueError(f'Embedding input graph mismatch: {name}, draw {draw}')
            if np.any((a > 0) & ~mask):
                raise ValueError(f'Draw support exceeds mean support: {name}, draw {draw}')
            isolated = (a.sum(1) == 0) & (a.sum(0) == 0)
            if not np.array_equal(isolated, emb['is_isolated'].numpy()):
                raise ValueError(f'Isolation mask mismatch: {name}, draw {draw}')
            metrics = compute_metrics_single(a, emb['Zs'].numpy().astype(float),
                emb['Zt'].numpy().astype(float), uniform, ks, mask)
            records.append(dict(metrics, graph=name, sample=f'sample_{draw:04d}'))
        print(f'Validated {name}: {len(ids)} draws')
    metric_cols = ['recon_kl', 'recon_js', 'uniform_kl', 'gain_unif'] + [f'ndcg@{k}' for k in ks] + ['auroc']
    df = pd.DataFrame(records)[['graph', 'sample', 'n_valid'] + metric_cols]
    per_sample = out / 'per_sample_metrics.csv'
    df.to_csv(per_sample, index=False)
    overall = pd.DataFrame({'mean': df[metric_cols].mean(), 'std': df[metric_cols].std()})
    overall.to_csv(out / 'validation_overall.csv', index_label='metric')
    pm_ref = compute_pm_reference(model, model_cfg, str(pm_dir), ks, device, cfg['pm_filename'])
    comparison = build_comparison(pm_ref, str(per_sample))
    consolidate_validation(df, pm_ref, comparison, metric_cols).to_csv(
        out / 'validation_by_graph.csv', index=False)
    plot_drawlevel_vs_pm(pm_ref, str(per_sample), str(out / 'validation_comparison.png'))
    save_json(out / 'run_metadata.json', dict(metadata, status='complete',
        draw_ids=ids, param_names=names, n_records=len(df)))
    return df


def consolidate_validation(draws, pm_ref, comparison, metric_cols):
    """Combine draw summaries and posterior-mean diagnostics in one row per graph."""
    rows = []
    for name in pm_ref['graph']:
        sub = draws.loc[draws.graph == name, metric_cols]
        if sub.empty:
            raise ValueError(f'Missing validation draws: {name}')
        row = {'graph': name, 'n_draws': len(sub)}
        for metric in metric_cols:
            row.update({f'draw_{metric}_mean': sub[metric].mean(),
                        f'draw_{metric}_std': sub[metric].std(),
                        f'draw_{metric}_median': sub[metric].median()})
        rows.append(row)
    pm = pm_ref.rename(columns={c: f'pm_{c}' for c in pm_ref if c != 'graph'})
    # The mean-graph KL, draw count and draw KL median are already included above.
    extra = comparison[['graph', 'draw_kl_q05', 'draw_kl_q95', 'draw_kl_max', 'rel_diff_median']]
    extra = extra.rename(columns={'draw_kl_q05': 'draw_recon_kl_q05',
                                  'draw_kl_q95': 'draw_recon_kl_q95',
                                  'draw_kl_max': 'draw_recon_kl_max',
                                  'rel_diff_median': 'recon_kl_relative_median_difference'})
    return (pd.DataFrame(rows).merge(pm, on='graph', validate='one_to_one')
            .merge(extra, on='graph', validate='one_to_one'))

def masked_row_softmax(S: np.ndarray) -> np.ndarray:
    maxv = np.max(S, axis=1, keepdims=True)
    all_ninf = np.isneginf(maxv).squeeze(1)
    safe_maxv = np.where(all_ninf[:, None], 0.0, maxv)
    shifted = S - safe_maxv
    exps = np.exp(shifted)
    exps = np.where(np.isneginf(S), 0.0, exps)
    denom = exps.sum(axis=1, keepdims=True)
    return np.where(denom > 0, exps / np.maximum(denom, EPS), 0.0)


def compute_A_bar(
    Zs: np.ndarray,
    Zt: np.ndarray,
    pm_mask: np.ndarray = None
) -> np.ndarray:
    N = Zs.shape[0]
    S = Zs @ Zt.T
    if not np.isfinite(S).all():
        raise ValueError('Decoder scores contain non-finite values')
    if pm_mask is not None:
        S[~pm_mask] = -np.inf
    S[np.arange(N), np.arange(N)] = -np.inf
    return masked_row_softmax(S)


def kl_divergence_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p_mask = p > 0
    q_safe = np.where(p_mask, np.maximum(q, EPS), 1.0)
    log_ratio = np.where(p_mask, np.log(np.where(p_mask, p, 1.0)) - np.log(q_safe), 0.0)
    kl = np.sum(np.where(p_mask, p * log_ratio, 0.0), axis=1)
    return np.where(p.sum(axis=1) > 0, kl, 0.0)


def js_divergence_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    js = 0.5 * kl_divergence_rows(p, m) + 0.5 * kl_divergence_rows(q, m)
    return np.where(p.sum(axis=1) > 0, js, 0.0)


def build_uniform_target(pm_matrix: np.ndarray) -> np.ndarray:
    pm_nodiag = pm_matrix.copy()
    np.fill_diagonal(pm_nodiag, 0.0)
    S_mask = (pm_nodiag > 0).astype(float)
    row_count = S_mask.sum(axis=1, keepdims=True)
    return np.where(row_count > 0, S_mask / np.maximum(row_count, EPS), 0.0)


def ndcg_at_k(p: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    """NDCG@k per row. p: relevance (A_target), q: predicted (A_bar)."""
    N = p.shape[0]
    ndcg = np.zeros(N)
    for i in range(N):
        pi, qi = p[i], q[i]
        if pi.sum() < EPS:
            continue
        top_k_idx = np.argsort(qi)[::-1][:k]
        top_k_ideal = np.argsort(pi)[::-1][:k]
        dcg = sum(pi[j] / np.log2(r + 2) for r, j in enumerate(top_k_idx))
        idcg = sum(pi[j] / np.log2(r + 2) for r, j in enumerate(top_k_ideal))
        if idcg > 0:
            ndcg[i] = dcg / idcg
    return ndcg


def compute_auroc(
    Zs: np.ndarray,
    Zt: np.ndarray,
    pm_mask: np.ndarray,
    valid_rows: np.ndarray,
) -> float:
    N = Zs.shape[0]
    S = Zs @ Zt.T
    off_diag = ~np.eye(N, dtype=bool)
    row_mask = np.zeros((N, N), dtype=bool)
    row_mask[valid_rows, :] = True
    eval_mask = off_diag & row_mask

    y_true = pm_mask[eval_mask].astype(int)
    y_score = S[eval_mask]
    if y_true.sum() == 0 or (1 - y_true).sum() == 0:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def compute_metrics_single(
    A: np.ndarray,
    Zs: np.ndarray,
    Zt: np.ndarray,
    Q_unif: np.ndarray,
    ndcg_ks: List[int],
    pm_mask: np.ndarray = None,
) -> Dict:
    A_nodiag = A.copy()
    np.fill_diagonal(A_nodiag, 0.0)
    rs = A_nodiag.sum(axis=1, keepdims=True)
    A_norm = np.where(rs > 0, A_nodiag / np.maximum(rs, EPS), 0.0)

    valid_mask = A_norm.sum(axis=1) > EPS
    n_valid = int(valid_mask.sum())

    nan_result = {"n_valid": 0, "recon_kl": np.nan, "recon_js": np.nan,
                  "uniform_kl": np.nan, "gain_unif": np.nan, "auroc": np.nan}
    for k in ndcg_ks:
        nan_result[f"ndcg@{k}"] = np.nan
    if n_valid == 0:
        return nan_result

    A_bar = compute_A_bar(Zs, Zt, pm_mask=pm_mask)

    kl_rows = kl_divergence_rows(A_norm, A_bar)
    js_rows = js_divergence_rows(A_norm, A_bar)
    ukl_rows = kl_divergence_rows(A_norm, Q_unif)

    recon_kl = float(kl_rows[valid_mask].mean())
    recon_js = float(js_rows[valid_mask].mean())
    uniform_kl = float(ukl_rows[valid_mask].mean())
    gain_unif = (float(1.0 - recon_kl / uniform_kl)
                 if uniform_kl > EPS else np.nan)

    result = {
        "n_valid": int(n_valid),
        "recon_kl": recon_kl,
        "recon_js": recon_js,
        "uniform_kl": uniform_kl,
        "gain_unif": gain_unif,
    }
    for k in ndcg_ks:
        ndcg_rows = ndcg_at_k(A_norm, A_bar, k)
        result[f"ndcg@{k}"] = float(ndcg_rows[valid_mask].mean())

    result["auroc"] = compute_auroc(Zs, Zt, pm_mask=pm_mask,
                                    valid_rows=valid_mask)
    return result


def compute_pm_reference(model: DiGAEModel, config: dict, pm_dir: str, ndcg_ks: List[int], device: torch.device, pm_filename: str='posterior_mean.csv') -> pd.DataFrame:
    N = config['N']
    alpha, beta = config['alpha'], config['beta']
    param_names = graph_names(pm_dir, config['param_names'], len(config['param_names']))
    print(f'\n[report / Part 1] Posterior-mean reference: {len(param_names)} graphs')
    X = torch.eye(N, dtype=torch.float32, device=device)
    records = []
    for g_name in param_names:
        pm_path = os.path.join(pm_dir, g_name, pm_filename)
        pm_np = read_graph(pm_path, N, config['node_names'])[0]
        A = torch.tensor(pm_np, dtype=torch.float32, device=device)
        A_hat = normalize_hatA(build_tilde_A(A), alpha=alpha, beta=beta)
        pm_mask_np = pm_np > 0
        pm_mask = torch.tensor(pm_mask_np, dtype=torch.bool, device=device)
        model.eval()
        with torch.no_grad():
            _, Zs, Zt = model(X, A_hat, pm_mask=pm_mask)
        Zs_np = Zs.cpu().numpy().astype(np.float64)
        Zt_np = Zt.cpu().numpy().astype(np.float64)
        Q_unif = build_uniform_target(pm_np)
        metrics = compute_metrics_single(A=pm_np, Zs=Zs_np, Zt=Zt_np, Q_unif=Q_unif, ndcg_ks=ndcg_ks, pm_mask=pm_mask_np)
        metrics['graph'] = g_name
        records.append(metrics)
        print(f'  {g_name:20s}  recon_kl={metrics['recon_kl']:.6f}  gain_unif={metrics['gain_unif']:.4f}  auroc={metrics['auroc']:.4f}')
    cols = ['graph', 'n_valid', 'recon_kl', 'recon_js', 'uniform_kl', 'gain_unif'] + [f'ndcg@{k}' for k in ndcg_ks] + ['auroc']
    return pd.DataFrame(records)[cols]


def build_comparison(pm_ref: pd.DataFrame, per_sample_csv: str) -> pd.DataFrame:
    print(f'\n[report / Part 2] Loading draw-level metrics: {per_sample_csv}')
    ps = pd.read_csv(per_sample_csv)
    if 'recon_kl' not in ps.columns:
        raise ValueError("per_sample_metrics.csv must contain 'recon_kl'")
    rows = []
    for _, ref in pm_ref.iterrows():
        g = ref['graph']
        sub = ps.loc[ps['graph'] == g, 'recon_kl'].dropna()
        if len(sub) == 0:
            raise ValueError(f'Missing validation records: {g}')
        pm_kl = ref['recon_kl']
        med = sub.median()
        rows.append({'graph': g, 'n_draws': int(len(sub)), 'pm_recon_kl': pm_kl, 'draw_kl_median': med, 'draw_kl_q05': sub.quantile(0.05), 'draw_kl_q95': sub.quantile(0.95), 'draw_kl_max': sub.max(), 'rel_diff_median': (med - pm_kl) / pm_kl if pm_kl > 0 else np.nan})
    return pd.DataFrame(rows)


def plot_drawlevel_vs_pm(pm_ref: pd.DataFrame, per_sample_csv: str,
                         out_path: str):
    ps = pd.read_csv(per_sample_csv)
    graphs = pm_ref["graph"].tolist()

    data, pm_vals, labels = [], [], []
    for g in graphs:
        sub = ps.loc[ps["graph"] == g, "recon_kl"].dropna().values
        if len(sub) == 0:
            continue
        data.append(sub)
        pm_vals.append(
            float(pm_ref.loc[pm_ref["graph"] == g, "recon_kl"].iloc[0]))
        labels.append(g)

    fig, ax = plt.subplots(figsize=(max(12, 0.45 * len(labels)), 6))
    bp = ax.boxplot(
        data, positions=np.arange(len(labels)),
        widths=0.55, whis=(5, 95), showfliers=False, patch_artist=True,
        boxprops=dict(facecolor="lightsteelblue", alpha=0.8),
        medianprops=dict(color="navy", linewidth=1.5),
        whiskerprops=dict(color="gray"), capprops=dict(color="gray"),
    )
    ax.scatter(np.arange(len(labels)), pm_vals,
               marker="D", s=45, color="firebrick", zorder=5)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("Row-averaged KL reconstruction loss", fontsize=11)
    ax.set_xlabel("Country-group combination", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    handles = [ax.scatter([], [], marker="D", s=45, color="firebrick"),
               bp["boxes"][0]]
    ax.legend(handles,
              ["Posterior mean (training input)",
               "Posterior draws (whiskers: 5-95%)"],
              fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Figure saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['cdtally', 'lamp_return'])
    parser.add_argument('--config')
    parser.add_argument('--model-dir')
    parser.add_argument('--graphs-dir')
    parser.add_argument('--emb-dir')
    parser.add_argument('--pm-dir')
    parser.add_argument('--out-dir')
    parser.add_argument('--max-draws', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'])
    args = parser.parse_args()
    validate_draws(load_config('embedding', args.dataset, args.config), args.model_dir,
        args.graphs_dir, args.emb_dir, args.pm_dir, args.out_dir, args.max_draws, args.device)


if __name__ == '__main__':
    main()
