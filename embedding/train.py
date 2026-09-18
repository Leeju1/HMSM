"""Train from scratch for an explicitly configured number of epochs."""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
import random
from typing import Dict, List, Sequence, Optional, Tuple
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from embedding.model import (DiGAEModel, build_tilde_A, normalize_hatA,
    row_normalize_target, valid_rows_from_target, kl_rowsum)
from embedding.graph_io import (load_config, resolve_path, load_posterior_mean_graphs,
    prepare_output, save_json, configure_device, provenance, configured_names)

GraphData = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def validate_training_config(cfg):
    for key in ('N', 'hidden_dim', 'out_dim', 'final_epochs', 'eval_every'):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('alpha', 'beta', 'lr', 'weight_decay', 'dropout'):
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if cfg['lr'] == 0 or cfg['dropout'] >= 1:
        raise ValueError('lr must be positive; dropout must be less than one')


def train_final(cfg, graphs_dir=None, out_dir=None, device=None):
    validate_training_config(cfg)
    device = configure_device(cfg, device)
    pm_dir = resolve_path(graphs_dir or cfg['pm_dir'])
    graphs, names, nodes = load_posterior_mean_graphs(
        pm_dir, cfg['N'], device, cfg['pm_filename'], cfg['expected_graphs'],
        configured_names(cfg))
    data = _preprocess(graphs, cfg['alpha'], cfg['beta'], device)
    out_dir = prepare_output(resolve_path(out_dir or cfg['model_dir']))
    cfg = dict(cfg, pm_dir=str(pm_dir), model_dir=str(out_dir))
    metadata = provenance(cfg, [pm_dir / name / cfg['pm_filename'] for name in names])
    save_json(out_dir / 'run_metadata.json', dict(metadata, status='running'))
    set_seed(cfg['seed'])
    model = DiGAEModel(cfg['N'], cfg['hidden_dim'], cfg['out_dim'],
                       cfg['dropout'], prevent_self_loop=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    history = _train_loop(model, optimizer, data, cfg['final_epochs'], cfg['eval_every'])
    torch.save(model.state_dict(), out_dir / 'model.pt')
    saved_cfg = dict(cfg, param_names=names, node_names=nodes, n_loaded_graphs=len(graphs),
                     final_seed=cfg['seed'], final_loss=history[-1]['train_loss'],
                     procedure='Reinitialize from seed and train on all mean graphs for the fixed final_epochs.')
    save_json(out_dir / 'train_config.json', saved_cfg)
    save_json(out_dir / 'train_history.json', history)
    plot_stage2(history, str(out_dir))
    save_json(out_dir / 'run_metadata.json', dict(metadata, status='complete',
              final_loss=history[-1]['train_loss']))
    print(f'Saved fixed-epoch model ({cfg["final_epochs"]} epochs): {out_dir}')
    return model, saved_cfg, history

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _preprocess(
    graphs: Sequence[torch.Tensor],
    alpha: float,
    beta: float,
    device: torch.device,
) -> List[GraphData]:
    if not graphs:
        raise ValueError("graphs must not be empty")

    N = graphs[0].size(0)
    X = torch.eye(N, dtype=torch.float32, device=device)
    out: List[GraphData] = []

    for graph_idx, A in enumerate(graphs):
        if A.dim() != 2 or A.shape != (N, N):
            raise ValueError(
                f"Graph {graph_idx} has shape {tuple(A.shape)}; expected {(N, N)}"
            )

        A_tilde = build_tilde_A(A)
        A_hat = normalize_hatA(A_tilde, alpha=alpha, beta=beta)
        A_tgt = row_normalize_target(A, zero_diag=True)
        valid = valid_rows_from_target(A_tgt)
        if not valid.any():
            raise ValueError(f"Graph {graph_idx} has no outgoing target mass")
        pm_mask = A > 0
        out.append((X, A_hat, A_tgt, valid, pm_mask))

    return out


def _subset_data(data: Sequence[GraphData], indices: Sequence[int]) -> List[GraphData]:
    subset = [data[int(i)] for i in indices]
    if not subset:
        raise ValueError("A train/validation subset is empty.")
    return subset


def _mean_kl_eval(model: DiGAEModel, data: Sequence[GraphData]) -> float:
    if not data:
        raise ValueError("Evaluation data must not be empty.")

    model.eval()
    total = 0.0
    with torch.no_grad():
        for X, A_hat, A_tgt, valid, pm_mask in data:
            A_bar, _, _ = model(X, A_hat, pm_mask=pm_mask)
            total += float(
                kl_rowsum(A_tgt, A_bar, valid_nodes=valid).item()
            )
    value = total / len(data)
    if not np.isfinite(value):
        raise ValueError("Evaluation produced non-finite loss")
    return value


def _train_loop(
    model: DiGAEModel,
    optimizer: torch.optim.Optimizer,
    train_data: Sequence[GraphData],
    epochs: int,
    eval_every: int,
    val_data: Optional[Sequence[GraphData]] = None,
    log_prefix: str = "  ",
    log_every_evals: int = 10,
) -> List[Dict[str, float]]:
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if eval_every < 1:
        raise ValueError("eval_every must be >= 1")
    if not train_data:
        raise ValueError("train_data must not be empty")

    history: List[Dict[str, float]] = []
    eval_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss: Optional[torch.Tensor] = None
        for X, A_hat, A_tgt, valid, pm_mask in train_data:
            A_bar, _, _ = model(X, A_hat, pm_mask=pm_mask)
            graph_loss = kl_rowsum(A_tgt, A_bar, valid_nodes=valid)
            total_loss = (
                graph_loss if total_loss is None else total_loss + graph_loss
            )

        assert total_loss is not None
        mean_loss = total_loss / len(train_data)
        if not torch.isfinite(mean_loss):
            raise ValueError(f"Non-finite training loss at epoch {epoch}")
        mean_loss.backward()
        optimizer.step()

        should_eval = (
            epoch == 1 or epoch % eval_every == 0 or epoch == epochs
        )
        if not should_eval:
            continue

        eval_count += 1
        rec: Dict[str, float] = {
            "epoch": int(epoch),
            "train_loss": _mean_kl_eval(model, train_data),
        }
        if val_data is not None:
            rec["val_loss"] = _mean_kl_eval(model, val_data)
        history.append(rec)

        should_log = (
            epoch == 1
            or epoch == epochs
            or eval_count % max(1, log_every_evals) == 0
        )
        if should_log:
            msg = (
                f"{log_prefix}Epoch {epoch:4d}/{epochs}: "
                f"train={rec['train_loss']:.6f}"
            )
            if val_data is not None:
                msg += f", val={rec['val_loss']:.6f}"
            print(msg)

    return history


def plot_stage2(history: Sequence[Dict[str, float]], out_dir: str) -> str:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, linewidth=2, label="Train loss (all graphs)")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("KL divergence loss", fontsize=12)
    ax.set_title("Final training on all graphs", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, "train_loss.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['cdtally', 'lamp_return'])
    parser.add_argument('--config', help='Optional JSON overrides, merged after the paper settings')
    parser.add_argument('--epochs', type=int, help='Override final_epochs; no CV or early stopping')
    parser.add_argument('--graphs-dir', help='Posterior-mean graph directory')
    parser.add_argument('--out-dir')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'])
    args = parser.parse_args()
    cfg = load_config('embedding', args.dataset, args.config)
    if args.epochs is not None:
        cfg['final_epochs'] = args.epochs
        cfg['epoch_selection'] = 'Fixed epoch count supplied by --epochs'
    train_final(cfg, args.graphs_dir, args.out_dir, args.device)


if __name__ == '__main__':
    main()
