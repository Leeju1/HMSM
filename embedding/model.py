"""Shared DiGAE encoder, support-masked decoder and row-averaged KL loss.

Sink rows receive an encoder self-loop; fully isolated rows remain zero.
The decoder excludes diagonal and off-support entries; loss uses only rows
with outgoing target mass.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_tilde_A(A: torch.Tensor) -> torch.Tensor:
    assert A.dim() == 2 and A.size(0) == A.size(1), "A must be square [N,N] tensor"
    N = A.size(0)
    device = A.device
    dtype = A.dtype

    outdeg = A.sum(dim=1)
    indeg = A.sum(dim=0)

    is_sink = (outdeg == 0) & (indeg > 0)

    I = torch.eye(N, device=device, dtype=dtype)

    A_tilde = A.clone()

    if is_sink.any():
        rows = is_sink.nonzero(as_tuple=True)[0]
        A_tilde[rows, :] = I[rows, :]

    return A_tilde


def normalize_hatA(
    A_tilde: torch.Tensor,
    alpha: float = 0.0,
    beta: float = 0.0,
) -> torch.Tensor:
    d_plus = A_tilde.sum(dim=1)
    d_minus = A_tilde.sum(dim=0)

    if beta != 0.0:
        inv_dplus_beta = torch.zeros_like(d_plus)
        mask = d_plus > 0
        inv_dplus_beta[mask] = d_plus[mask].pow(-beta)
    else:
        inv_dplus_beta = torch.ones_like(d_plus)

    if alpha != 0.0:
        inv_dminus_alpha = torch.zeros_like(d_minus)
        mask = d_minus > 0
        inv_dminus_alpha[mask] = d_minus[mask].pow(-alpha)
    else:
        inv_dminus_alpha = torch.ones_like(d_minus)

    Dp = torch.diag(inv_dplus_beta)
    Dm = torch.diag(inv_dminus_alpha)

    A_hat = Dp @ A_tilde @ Dm
    return A_hat


class RowSoftmaxDecoder(nn.Module):
    def __init__(self, prevent_self_loop: bool = True):
        super().__init__()
        self.prevent_self_loop = prevent_self_loop

    @staticmethod
    def _masked_row_softmax(scores: torch.Tensor) -> torch.Tensor:
        maxv, _ = torch.max(scores, dim=1, keepdim=True)
        is_all_ninf = torch.isneginf(maxv).squeeze(1)
        safe_maxv = torch.where(is_all_ninf.unsqueeze(1),
                                torch.zeros_like(maxv),
                                maxv)

        shifted = scores - safe_maxv
        exps = torch.exp(shifted)
        exps = torch.where(torch.isneginf(scores), torch.zeros_like(exps), exps)

        denom = exps.sum(dim=1, keepdim=True)
        probs = torch.where(denom > 0,
                            exps / denom.clamp_min(1e-12),
                            torch.zeros_like(exps))
        return probs

    def forward(
        self,
        Zs: torch.Tensor,
        Zt: torch.Tensor,
        pm_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert Zs.dim() == 2 and Zt.dim() == 2 and Zs.size(0) == Zt.size(0),\
            "Zs, Zt must be [N, d] with same N"

        N = Zs.size(0)
        S = Zs @ Zt.t()
        if not torch.isfinite(S).all():
            raise ValueError('Decoder scores contain non-finite values')

        if pm_mask is not None:
            assert pm_mask.shape == (N, N), "pm_mask must be [N, N]"
            S = S.masked_fill(~pm_mask, float("-inf"))

        if self.prevent_self_loop:
            diag_mask = torch.eye(N, dtype=torch.bool, device=S.device)
            S = S.masked_fill(diag_mask, float("-inf"))

        return self._masked_row_softmax(S)


def row_normalize_target(A: torch.Tensor, zero_diag: bool = True) -> torch.Tensor:
    assert A.dim() == 2 and A.size(0) == A.size(1)
    A_out = A.clone()
    if zero_diag:
        idx = torch.arange(A.size(0), device=A.device)
        A_out[idx, idx] = 0.0
    row_sum = A_out.sum(dim=1, keepdim=True)
    A_norm = torch.where(row_sum > 0.0, A_out / row_sum.clamp_min(1e-12), torch.zeros_like(A_out))
    return A_norm


def valid_rows_from_target(A_target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    assert A_target.dim() == 2 and A_target.size(0) == A_target.size(1),\
        "A_target must be square [N,N]"
    return A_target.sum(dim=1) > eps


def kl_rowsum(
    p: torch.Tensor,
    q: torch.Tensor,
    valid_nodes: Optional[torch.Tensor] = None,
    eps: float = 1e-12
) -> torch.Tensor:
    assert p.shape == q.shape and p.dim() == 2 and p.size(0) == p.size(1), "p, q must be square [N,N]"

    support_mask = p > 0
    q_safe = q.clamp_min(eps)

    kl_per_row = torch.sum(
        torch.where(
            support_mask,
            p * (torch.log(p.clamp_min(eps)) - torch.log(q_safe)),
            torch.zeros_like(p)
        ),
        dim=1
    )

    if valid_nodes is not None:
        assert valid_nodes.dim() == 1 and valid_nodes.size(0) == p.size(0), "valid_nodes must be [N] bool"
        kl_per_row = kl_per_row * valid_nodes.to(kl_per_row.dtype)
        n_valid = valid_nodes.sum().float()
        return kl_per_row.sum() / (n_valid + eps)

    return kl_per_row.mean()


class DiGAEEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()

        self.Ws0 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.Ws1 = nn.Linear(hidden_dim, out_dim, bias=False)

        self.Wt0 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.Wt1 = nn.Linear(hidden_dim, out_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.Ws0, self.Ws1, self.Wt0, self.Wt1]:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A_hat_T = A_hat.transpose(0, 1)

        Hs = F.relu(self.Ws0(A_hat_T @ X))
        Hs = self.dropout(Hs)
        Zs = self.Wt1(A_hat @ Hs)

        Ht = F.relu(self.Wt0(A_hat @ X))
        Ht = self.dropout(Ht)
        Zt = self.Ws1(A_hat_T @ Ht)

        return Zs, Zt


class DiGAEModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0, prevent_self_loop: bool = True):
        super().__init__()
        self.encoder = DiGAEEncoder(in_dim, hidden_dim, out_dim, dropout=dropout)
        self.decoder = RowSoftmaxDecoder(prevent_self_loop=prevent_self_loop)

    def forward(
        self,
        X: torch.Tensor,
        A_hat: torch.Tensor,
        pm_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Zs, Zt = self.encoder(X, A_hat)
        A_bar = self.decoder(Zs, Zt, pm_mask=pm_mask)
        return A_bar, Zs, Zt
