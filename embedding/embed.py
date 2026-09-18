"""Apply one fixed DiGAE checkpoint to paired posterior graph draws."""
from __future__ import annotations

import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time
import torch
from embedding.model import build_tilde_A, normalize_hatA
from embedding.graph_io import (load_config, resolve_path, load_model, read_graph,
    graph_names, configured_names, sample_paths, prepare_output, save_json,
    configure_device, provenance, sha256)


def embed_single_graph(model, A, X, alpha, beta, pm_mask=None):
    A_hat = normalize_hatA(build_tilde_A(A), alpha, beta)
    with torch.no_grad():
        return model(X, A_hat, pm_mask=pm_mask)


def embed_draws(cfg, model_dir=None, graphs_dir=None, pm_dir=None, out_dir=None,
                max_draws=None, device=None):
    device = configure_device(cfg, device)
    model_dir = resolve_path(model_dir or cfg['model_dir'])
    graphs_dir = resolve_path(graphs_dir or cfg['graphs_dir'])
    pm_dir = resolve_path(pm_dir or cfg['pm_dir'])
    model, model_cfg = load_model(model_dir, device)
    if model_cfg['N'] != cfg['N']:
        raise ValueError('Model and dataset node counts differ')
    names = graph_names(graphs_dir, configured_names(cfg), cfg['expected_graphs'])
    if names != model_cfg['param_names']:
        raise ValueError('Model and draw graph types differ')
    graph_names(pm_dir, names, len(names))
    ids, paths = sample_paths(graphs_dir, names, expected_draws=cfg['expected_draws'], max_draws=max_draws)
    nodes, N = model_cfg['node_names'], model_cfg['N']
    masks = {}
    for name in names:
        pm, _ = read_graph(pm_dir / name / cfg['pm_filename'], N, nodes)
        masks[name] = torch.tensor(pm > 0, dtype=torch.bool, device=device)
    out = prepare_output(resolve_path(out_dir or cfg['emb_dir']))
    model_hash = sha256(model_dir / 'model.pt')
    metadata = provenance(dict(cfg, model_dir=str(model_dir), graphs_dir=str(graphs_dir),
        pm_dir=str(pm_dir), out_dir=str(out), max_draws=max_draws),
        [model_dir / 'model.pt', model_dir / 'train_config.json'] +
        [pm_dir / name / cfg['pm_filename'] for name in names])
    save_json(out / 'run_metadata.json', dict(metadata, status='running'))
    X = torch.eye(N, dtype=torch.float32, device=device)
    started, count = time.time(), 0
    for name in names:
        dest = out / name
        dest.mkdir()
        for draw in ids:
            graph_path = paths[name][draw]
            a, _ = read_graph(graph_path, N, nodes)
            A = torch.tensor(a, dtype=torch.float32, device=device)
            if torch.any((A > 0) & ~masks[name]):
                raise ValueError(f'Draw support exceeds posterior-mean support: {graph_path}')
            isolated = (A.sum(1) == 0) & (A.sum(0) == 0)
            if isolated.all():
                raise ValueError(f'All nodes are isolated: {graph_path}')
            _, zs, zt = embed_single_graph(model, A, X,
                                         model_cfg['alpha'], model_cfg['beta'], masks[name])
            if not torch.isfinite(zs).all() or not torch.isfinite(zt).all():
                raise ValueError(f'Non-finite embedding: {graph_path}')
            torch.save({'Zs': zs.cpu(), 'Zt': zt.cpu(), 'node_names': nodes,
                        'is_isolated': isolated.cpu(), 'draw_id': draw, 'graph': name,
                        'model_sha256': model_hash, 'graph_sha256': sha256(graph_path)},
                       dest / f'sample_{draw:04d}_embeddings.pt')
            count += 1
        print(f'Embedded {name}: {len(ids)} draws ({count} total)')
    summary = {'status': 'complete', 'model_dir': str(model_dir), 'model_sha256': model_hash,
        'graphs_dir': str(graphs_dir), 'pm_dir': str(pm_dir), 'total_processed': count,
        'total_errors': 0, 'param_names': names, 'draw_ids': ids,
        'elapsed_seconds': time.time() - started, 'config': model_cfg}
    save_json(out / 'embedding_summary.json', summary)
    save_json(out / 'run_metadata.json', dict(metadata, status='complete'))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['cdtally', 'lamp_return'])
    parser.add_argument('--config')
    parser.add_argument('--model-dir')
    parser.add_argument('--graphs-dir')
    parser.add_argument('--pm-dir')
    parser.add_argument('--out-dir')
    parser.add_argument('--max-draws', type=int, help='Pilot: first N paired draws of every graph')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'])
    args = parser.parse_args()
    embed_draws(load_config('embedding', args.dataset, args.config), args.model_dir,
                args.graphs_dir, args.pm_dir, args.out_dir, args.max_draws, args.device)


if __name__ == '__main__':
    main()
