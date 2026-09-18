"""Exact posterior Wasserstein distances on concatenated source/target embeddings.

The paper uses W1 with Euclidean ground cost. Nonisolated nodes have uniform
mass and fully isolated nodes have zero mass. Draw IDs are paired across graphs.
HPD outputs use the shortest empirical interval containing ceil(mass * S) draws.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import ot
from embedding.graph_io import (load_config, resolve_path, graph_names, sample_paths,
    load_embedding_record, load_embedding_summary, prepare_output, save_json, provenance, configured_names)


def distance_matrix(files, names, draw, p=1, node_names=None, out_dim=None, model_hash=None):
    clouds, weights = [], []
    for name in names:
        record = load_embedding_record(files[name][draw], node_names, out_dim,
            graph=name, draw_id=draw, model_hash=model_hash)
        if node_names is None:
            node_names = record['node_names']
            out_dim = record['Zs'].shape[1]
        clouds.append(np.hstack([record['Zs'].numpy().astype(float), record['Zt'].numpy().astype(float)]))
        weights.append((~record['is_isolated'].numpy()).astype(float))
    matrix = np.zeros((len(names), len(names)), dtype=float)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            matrix[i, j] = matrix[j, i] = wasserstein_distance_pot(
                clouds[i], clouds[j], weights[i], weights[j], p)
    assert_finite_array(matrix, f'distance matrix for draw {draw}')
    if np.any(matrix < 0):
        raise ValueError('Negative optimal-transport cost')
    return matrix


def compute_distances(cfg, emb_dir=None, out_dir=None, max_draws=None):
    if cfg['p'] not in (1, 2) or not 0 < cfg['hpd_mass'] <= 1:
        raise ValueError('p must be 1 or 2 and hpd_mass must be in (0, 1]')
    if not isinstance(cfg['progress_every'], int) or cfg['progress_every'] < 1:
        raise ValueError('progress_every must be a positive integer')
    emb_dir = resolve_path(emb_dir or cfg['emb_dir'])
    names = graph_names(emb_dir, configured_names(cfg), cfg['expected_graphs'])
    ids, files = sample_paths(emb_dir, names, kind='embedding',
        expected_draws=None if max_draws else cfg['expected_draws'], max_draws=max_draws)
    summary_path = emb_dir / 'embedding_summary.json'
    summary = load_embedding_summary(emb_dir, names, ids, cfg['dataset'], cfg['expected_nodes'])
    model_cfg = summary['config']
    out = prepare_output(resolve_path(out_dir or cfg['out_dir']))
    metadata = provenance(dict(cfg, emb_dir=str(emb_dir), out_dir=str(out), max_draws=max_draws), [summary_path])
    save_json(out / 'run_metadata.json', dict(metadata, status='running'))
    individual = out / 'distances'
    if cfg['save_individual']:
        individual.mkdir()
    # About 63 MB for 10,000 x 28 x 28 float64 values.
    stack = np.empty((len(ids), len(names), len(names)), dtype=np.float64)
    started = time.time()
    for pos, draw in enumerate(ids):
        stack[pos] = distance_matrix(files, names, draw, cfg['p'], model_cfg['node_names'],
                                     model_cfg['out_dim'], summary.get('model_sha256'))
        if cfg['save_individual']:
            pd.DataFrame(stack[pos], index=names, columns=names).to_csv(
                individual / f'iteration_{draw:04d}.csv', float_format='%.8f')
        if pos == 0 or (pos + 1) % cfg['progress_every'] == 0 or pos == len(ids) - 1:
            print(f'Distances: {pos + 1}/{len(ids)} paired draws')
    mean, lower, upper = compute_posterior_statistics(stack, names, cfg['hpd_mass'])
    for filename, matrix in [('posterior_mean', mean), ('posterior_hpd_lower', lower), ('posterior_hpd_upper', upper)]:
        matrix.to_csv(out / f'{filename}.csv', float_format='%.8f')
    save_json(out / 'summary.json', {'status': 'complete', 'emb_dir': str(emb_dir),
        'n_iterations': len(ids), 'iteration_ids_first': ids[0], 'iteration_ids_last': ids[-1],
        'draw_ids': ids, 'n_parameters': len(names), 'param_names': names,
        'wasserstein_p': cfg['p'], 'method': 'POT emd2', 'hpd_mass': cfg['hpd_mass'],
        'save_individual': cfg['save_individual'], 'model_sha256': summary.get('model_sha256'),
        'interval_method': 'Shortest interval containing ceil(hpd_mass * n_draws) samples',
        'total_time_seconds': time.time() - started})
    save_json(out / 'run_metadata.json', dict(metadata, status='complete'))
    return mean, lower, upper

def _normalize_weights(weights: Optional[np.ndarray], n: int, name: str) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=float) / n

    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {weights.shape}")
    if not np.isfinite(weights).all():
        raise ValueError(f"{name} contains non-finite values")
    if np.any(weights < 0):
        raise ValueError(f"{name} contains negative weights")

    total = float(weights.sum())
    if total <= 0:
        raise ValueError(f"{name} has no positive mass")
    return weights / total


def wasserstein_distance_pot(
    X: np.ndarray,
    Y: np.ndarray,
    weights_X: Optional[np.ndarray] = None,
    weights_Y: Optional[np.ndarray] = None,
    p: int = 1,
) -> float:
    n, d = X.shape
    m, d2 = Y.shape
    if d != d2:
        raise ValueError(f"Dimension mismatch: {d} vs {d2}")
    if p not in (1, 2):
        raise ValueError("Only p=1 or p=2 supported")

    a = _normalize_weights(weights_X, n, "weights_X")
    b = _normalize_weights(weights_Y, m, "weights_Y")

    C = ot.dist(X, Y, metric="euclidean")
    if p == 2:
        C = C ** 2

    W, solver = ot.emd2(a, b, C, numItermax=1000000, log=True)
    if solver['warning'] is not None:
        raise RuntimeError(f"Optimal transport did not converge: {solver['warning']}")
    if not np.isfinite(W) or W < 0:
        raise ValueError('Optimal transport returned an invalid cost')
    return float(W if p == 1 else np.sqrt(W))


def assert_finite_array(x: np.ndarray, name: str) -> None:
    finite = np.isfinite(x)
    if finite.all():
        return
    n_bad = int((~finite).sum())
    bad_pos = np.argwhere(~finite)
    preview = bad_pos[:5].tolist()
    raise ValueError(
        f"Non-finite values found in {name}: {n_bad} entries. First positions: {preview}"
    )


def compute_hpd_interval(samples: np.ndarray, mass: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    if samples.ndim < 1:
        raise ValueError("samples must have at least one dimension")
    if not (0.0 < mass <= 1.0):
        raise ValueError("mass must be in (0, 1]")

    sorted_samples = np.sort(samples, axis=0)
    n = sorted_samples.shape[0]
    if n == 0:
        raise ValueError("No posterior samples provided")
    if n == 1 or mass == 1.0:
        return sorted_samples[0], sorted_samples[-1]

    k = int(np.ceil(mass * n)) - 1
    k = max(0, min(k, n - 1))
    widths = sorted_samples[k:] - sorted_samples[: n - k]
    min_idx = np.argmin(widths, axis=0)

    lower = np.take_along_axis(
        sorted_samples[: n - k], np.expand_dims(min_idx, axis=0), axis=0
    )[0]
    upper = np.take_along_axis(
        sorted_samples[k:], np.expand_dims(min_idx, axis=0), axis=0
    )[0]
    return lower, upper


def compute_posterior_statistics(
    distance_matrices: List[np.ndarray],
    param_names: List[str],
    hpd_mass: float = 0.95,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    D_stack = np.array(distance_matrices, dtype=float)
    if D_stack.ndim != 3:
        raise ValueError(f"Expected distance stack [S,G,G], got shape {D_stack.shape}")
    assert_finite_array(D_stack, "distance_matrices")

    mean_matrix = np.mean(D_stack, axis=0)
    lower_matrix, upper_matrix = compute_hpd_interval(D_stack, mass=hpd_mass)

    posterior_mean = pd.DataFrame(mean_matrix, index=param_names, columns=param_names)
    hpd_lower = pd.DataFrame(lower_matrix, index=param_names, columns=param_names)
    hpd_upper = pd.DataFrame(upper_matrix, index=param_names, columns=param_names)
    return posterior_mean, hpd_lower, hpd_upper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['cdtally', 'lamp_return'])
    parser.add_argument('--config')
    parser.add_argument('--emb-dir')
    parser.add_argument('--out-dir')
    parser.add_argument('--max-draws', type=int, help='Pilot: use first N paired draws')
    parser.add_argument('--no-save-individual', action='store_true')
    args = parser.parse_args()
    cfg = load_config('distance', args.dataset, args.config)
    if args.no_save_individual:
        cfg['save_individual'] = False
    compute_distances(cfg, args.emb_dir, args.out_dir, args.max_draws)


if __name__ == '__main__':
    main()
