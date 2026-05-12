import sys
import os
from typing import Tuple

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "kernels"))

from compile_wrapper import BaseKernel


def _shared_mask_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """
    Shared-mask sparse GEMM prototype.

    x:      [B, 1, D] during decode
    weight: [N, D]
    output: [B, 1, N]

    Computes:
      mask = max(abs(x), dim=batch) > threshold
      y = x[:, mask] @ weight[:, mask].T

    This is not a custom Triton kernel yet, but it is a true batched GEMM over
    active hidden dimensions instead of running sparse GEMV separately per row.
    """
    B, S, D = x.shape
    assert S == 1, "decode-only shared-mask GEMM expects seq_len == 1"
    assert weight.shape[1] == D

    x2 = x.reshape(B, D)

    # One mask shared by all batch rows.
    shared_score = x2.abs().amax(dim=0)
    mask = shared_score > threshold

    # Safety fallback: avoid empty active set.
    if not bool(mask.any()):
        return torch.zeros(B, 1, weight.shape[0], device=x.device, dtype=x.dtype)

    active_idx = torch.nonzero(mask, as_tuple=False).flatten()

    x_active = x2.index_select(dim=1, index=active_idx)
    w_active = weight.index_select(dim=1, index=active_idx)

    y = torch.matmul(x_active, w_active.T)

    return y.reshape(B, 1, weight.shape[0])


def _shared_mask_qkv_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold_q: float,
    threshold_k: float,
    threshold_v: float,
    kv_size: int,
) -> torch.Tensor:
    """
    Shared-mask sparse GEMM for fused QKV.

    x:      [B, 1, D]
    weight: [N_total, D]
      where N_total = N_q + N_k + N_v

    For Q, K, V we use separate thresholds, but each threshold creates one
    mask shared across batch.
    """
    B, S, D = x.shape
    assert S == 1, "decode-only shared-mask QKV GEMM expects seq_len == 1"
    assert weight.shape[1] == D

    N_total = weight.shape[0]
    N_q = N_total - 2 * kv_size
    N_k = kv_size
    N_v = kv_size

    x2 = x.reshape(B, D)
    shared_score = x2.abs().amax(dim=0)

    def project(start: int, end: int, threshold: float) -> torch.Tensor:
        mask = shared_score > threshold

        if not bool(mask.any()):
            return torch.zeros(B, end - start, device=x.device, dtype=x.dtype)

        active_idx = torch.nonzero(mask, as_tuple=False).flatten()
        x_active = x2.index_select(dim=1, index=active_idx)
        w_active = weight[start:end].index_select(dim=1, index=active_idx)

        return torch.matmul(x_active, w_active.T)

    q = project(0, N_q, threshold_q)
    k = project(N_q, N_q + N_k, threshold_k)
    v = project(N_q + N_k, N_q + N_k + N_v, threshold_v)

    y = torch.cat([q, k, v], dim=-1)

    return y.reshape(B, 1, N_total)


class SharedMaskGEMM(BaseKernel):
    def meta(
        self,
        hidden_states: torch.Tensor,
        weights: torch.Tensor,
        threshold: float,
        sparsity_bin: int,
    ) -> torch.Tensor:
        return hidden_states.new_empty(
            hidden_states.size(0),
            hidden_states.size(1),
            weights.size(0),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        weights: torch.Tensor,
        threshold: float,
        sparsity_bin: int,
    ) -> torch.Tensor:
        if hidden_states.shape[1] == 1:
            return _shared_mask_gemm(hidden_states, weights, threshold)

        return torch.matmul(hidden_states, weights.T)


class SharedMaskQKVGEMM(BaseKernel):
    def meta(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        threshold_q: float,
        threshold_k: float,
        threshold_v: float,
        sparsity_bin: int,
        kv_size: int,
    ) -> torch.Tensor:
        return x.new_empty(x.shape[0], x.shape[1], weight.shape[0])

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        threshold_q: float,
        threshold_k: float,
        threshold_v: float,
        sparsity_bin: int,
        kv_size: int,
    ) -> torch.Tensor:
        if x.shape[1] == 1:
            return _shared_mask_qkv_gemm(
                x,
                weight,
                threshold_q,
                threshold_k,
                threshold_v,
                kv_size,
            )

        return torch.matmul(x, weight.T)