"""Configuration, labeled graph IO and reproducibility metadata.

Relative paths resolve from the repository root. Inputs must preserve the
model's action order and use matching draw IDs across graph types.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("cdtally", "lamp_return")


def resolve_path(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _merge(base, updates):
    result = dict(base)
    for key, value in updates.items():
        result[key] = (_merge(result[key], value)
                       if isinstance(value, dict) and isinstance(result.get(key), dict)
                       else value)
    return result


def load_config(section, dataset, config_path=None):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    paths = [ROOT / 'config' / section / 'common.json',
             ROOT / 'config' / section / f'{dataset}.json']
    if config_path:
        paths.append(resolve_path(config_path))
    cfg = {}
    for path in paths:
        with path.open(encoding='utf-8-sig') as handle:
            cfg = _merge(cfg, json.load(handle))
    if cfg['dataset'] != dataset:
        raise ValueError('Config dataset does not match --dataset')
    cfg['config_files'] = [str(p) for p in paths]
    return cfg


def graph_names(directory, expected_names=None, expected_count=28):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'Graph directory not found: {directory}')
    names = sorted(p.name for p in directory.iterdir() if p.is_dir())
    if len(names) != expected_count or any(not re.fullmatch(r'prob[01]_[A-Z]{2}', n) for n in names):
        raise ValueError(f'Expected {expected_count} prob0/prob1 country folders in {directory}, got {names}')
    if expected_names is not None and names != sorted(expected_names):
        raise ValueError(f'Graph labels do not match the configured/model graph labels: {directory}')
    return names


def configured_names(cfg):
    return sorted(f'prob{response}_{country}' for response in (0, 1)
                  for country in cfg['countries'])


def read_graph(path, N, node_names=None):
    df = pd.read_csv(path, index_col=0)
    rows = [str(x) for x in df.index]
    cols = [str(x) for x in df.columns]
    if df.shape != (N, N) or len(set(rows)) != N or len(set(cols)) != N:
        raise ValueError(f'Invalid graph shape or duplicate labels: {path}')
    if rows != cols or (node_names is not None and rows != list(node_names)):
        raise ValueError(f'Action row/column order mismatch: {path}')
    a = df.to_numpy(dtype=np.float64)
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError(f'Graph contains negative or non-finite entries: {path}')
    if np.any(np.diag(a) != 0):
        raise ValueError(f'Transition graphs must have zero diagonal: {path}')
    sums = a.sum(axis=1)
    if not np.all((sums == 0) | np.isclose(sums, 1, atol=1e-6, rtol=0)):
        raise ValueError(f'Graph rows must sum to zero or one: {path}')
    return a, rows


def load_posterior_mean_graphs(base_dir, N, device, filename='posterior_mean.csv',
                               expected_graphs=28, expected_names=None):
    names = graph_names(base_dir, expected_names, expected_graphs)
    graphs, node_names = [], None
    for name in names:
        a, node_names = read_graph(Path(base_dir) / name / filename, N, node_names)
        graphs.append(torch.tensor(a, dtype=torch.float32, device=device))
    return graphs, names, node_names


def sample_paths(directory, names, kind='graph', expected_draws=10000, max_draws=None):
    """Return paired ID-to-path maps; a pilot selects the same IDs in every graph."""
    suffix = '.csv' if kind == 'graph' else '_embeddings.pt'
    pattern = re.compile(r'^sample_(\d+)' + re.escape(suffix) + '$')
    maps, reference = {}, None
    for name in names:
        mapping = {}
        for path in (Path(directory) / name).iterdir():
            match = pattern.fullmatch(path.name)
            if path.is_file() and match:
                draw = int(match.group(1))
                if draw in mapping:
                    raise ValueError(f'Duplicate draw ID {draw}: {path.parent}')
                mapping[draw] = path
        ids = sorted(mapping)
        if not ids or (reference is not None and ids != reference):
            raise ValueError(f'Missing or inconsistent draw IDs: {name}')
        if expected_draws is not None and ids != list(range(expected_draws)):
            raise ValueError(f'Expected draw IDs 0..{expected_draws - 1}: {name}')
        reference = ids
        maps[name] = mapping
    if max_draws is not None:
        if max_draws < 1 or max_draws > len(reference):
            raise ValueError('--max-draws must be between 1 and the available draw count')
        reference = reference[:max_draws]
    return reference, {n: {i: maps[n][i] for i in reference} for n in names}


def load_model(model_dir, device, model_name='model.pt'):
    from embedding.model import DiGAEModel
    model_dir = Path(model_dir)
    with (model_dir / 'train_config.json').open(encoding='utf-8-sig') as f:
        cfg = json.load(f)
    model = DiGAEModel(cfg['N'], cfg['hidden_dim'], cfg['out_dim'],
                       cfg.get('dropout', 0.0), prevent_self_loop=True).to(device)
    model.load_state_dict(torch.load(model_dir / model_name, map_location=device, weights_only=True))
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError(f'Model contains non-finite weights: {model_dir}')
    model.eval()
    return model, cfg


def load_embedding_summary(directory, names, draw_ids, dataset=None, N=None):
    path = Path(directory) / 'embedding_summary.json'
    with path.open(encoding='utf-8-sig') as handle:
        summary = json.load(handle)
    if summary.get('total_errors', 0) or summary.get('status', 'complete') != 'complete':
        raise ValueError('Embedding run was not completed without errors')
    cfg = summary['config']
    if names != cfg['param_names']:
        raise ValueError('Embedding summary graph labels differ')
    if N is not None and cfg['N'] != N:
        raise ValueError('Embedding and dataset node counts differ')
    if dataset is not None and cfg.get('dataset', dataset) != dataset:
        raise ValueError('Embedding and requested dataset differ')
    if 'draw_ids' in summary and not set(draw_ids) <= set(summary['draw_ids']):
        raise ValueError('Selected draw IDs are absent from the embedding summary')
    return summary


def load_embedding_record(path, node_names=None, out_dim=None, *, graph=None,
                          draw_id=None, model_hash=None):
    record = torch.load(path, map_location='cpu', weights_only=True)
    if not {'Zs', 'Zt', 'node_names', 'is_isolated'} <= record.keys():
        raise ValueError(f'Incomplete embedding record: {path}')
    names = record['node_names']
    zs, zt = record['Zs'], record['Zt']
    isolated = record['is_isolated']
    if (zs.ndim != 2 or zt.shape != zs.shape or len(names) != zs.shape[0]
            or len(set(names)) != len(names) or isolated.shape != (zs.shape[0],)
            or isolated.dtype != torch.bool or not torch.isfinite(zs).all()
            or not torch.isfinite(zt).all()):
        raise ValueError(f'Invalid embedding record: {path}')
    if node_names is not None and list(names) != list(node_names):
        raise ValueError(f'Embedding action order mismatch: {path}')
    if out_dim is not None and zs.shape[1] != out_dim:
        raise ValueError(f'Embedding dimension mismatch: {path}')
    if isolated.all():
        raise ValueError(f'All nodes are isolated: {path}')
    # Legacy records omit provenance; modern summaries require a matching hash.
    if model_hash is not None and record.get('model_sha256') != model_hash:
        raise ValueError(f'Embedding checkpoint mismatch: {path}')
    for key, expected in [('graph', graph), ('draw_id', draw_id)]:
        if expected is not None and record.get(key, expected) != expected:
            raise ValueError(f'Embedding {key} mismatch: {path}')
    return record


def prepare_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'Output must be new or empty; use a separate --out-dir: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, value):
    def convert(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)
    with Path(path).open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=True, default=convert, allow_nan=False)
        f.write('\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def provenance(cfg, input_paths=()):
    versions = {}
    for package in ('numpy', 'pandas', 'torch', 'matplotlib', 'scikit-learn', 'POT', 'scipy'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    sources = list((ROOT / 'embedding').glob('*.py')) + list((ROOT / 'distance').glob('*.py'))
    paths = [Path(p) for p in cfg.get('config_files', [])] + list(map(Path, input_paths)) + sources
    return {'settings': cfg, 'python': platform.python_version(), 'platform': platform.platform(),
            'packages': versions, 'torch_cuda': torch.version.cuda,
            'files_sha256': {str(p): sha256(p) for p in paths}}


def configure_device(cfg, override=None):
    threads = cfg.get('num_threads', 1)
    if not isinstance(threads, int) or threads < 1:
        raise ValueError('num_threads must be a positive integer')
    torch.set_num_threads(threads)
    name = override or cfg.get('device', 'cpu')
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name not in ('cpu', 'cuda') or (name == 'cuda' and not torch.cuda.is_available()):
        raise ValueError(f'Device unavailable: {name}')
    cfg['device'] = name
    return torch.device(name)
