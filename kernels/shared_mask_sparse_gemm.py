import sys
import os

import torch
import triton
import triton.language as tl

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "kernels"))

from compile_wrapper import BaseKernel


@triton.jit
def shared_mask_sparse_gemm_kernel(
    Y,
    W,
    X,
    threshold,
    N: tl.constexpr,
    M: tl.constexpr,
    BATCHSIZE: tl.constexpr,
    MIN_ACTIVE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Computes Y[B, N] = sparse_shared(X[B, M]) @ W[N, M].T

    X layout: [B, 1, M], contiguous/flattened as [B, M]
    W layout: [N, M], but TEAL stores W in column-major style via W.T.contiguous().T,
              so W[n, d] lives at offset d * N + n.
    Y layout: [B, 1, N], contiguous/flattened as [B, N]

    Grid:
      pid_n: output feature block
      pid_d: hidden-dimension block
      pid_b: batch block

    This supports BATCHSIZE > BLOCK_B by splitting the batch into batch blocks.
    """
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    x = tl.load(
        X + offs_b[:, None] * M + offs_d[None, :],
        mask=(offs_b[:, None] < BATCHSIZE) & (offs_d[None, :] < M),
        other=0.0,
    )

    shared_score = tl.max(tl.abs(x), axis=0)  # [BLOCK_D]
    active = shared_score > threshold         # [BLOCK_D]
    active_count = tl.sum(active.to(tl.int32), axis=0)

    # Work skip: avoid loading W and avoid tl.dot for mostly inactive D-blocks.
    # MIN_ACTIVE=1 means skip only fully inactive blocks.
    if active_count < MIN_ACTIVE:
        return

    # W[n, d] offset in column-major style: d * N + n
    w = tl.load(
        W + offs_d[:, None] * N + offs_n[None, :],
        mask=(offs_d[:, None] < M) & (offs_n[None, :] < N),
        other=0.0,
    )

    w = tl.where(active[:, None], w, 0.0).to(tl.float16)
    acc = tl.dot(x.to(tl.float16), w)

    tl.atomic_add(
        Y + offs_b[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_b[:, None] < BATCHSIZE) & (offs_n[None, :] < N),
    )


@triton.jit
def shared_mask_sparse_qkv_gemm_kernel(
    Y,
    W,
    X,
    threshold_q,
    threshold_k,
    threshold_v,
    N: tl.constexpr,
    N_Q: tl.constexpr,
    N_KV: tl.constexpr,
    M: tl.constexpr,
    BATCHSIZE: tl.constexpr,
    MIN_ACTIVE: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused QKV version.

    W rows:
      [0:N_Q]              = Q
      [N_Q:N_Q+N_KV]       = K
      [N_Q+N_KV:N]         = V

    Grid:
      pid_n: output feature block
      pid_d: hidden-dimension block
      pid_b: batch block

    This supports BATCHSIZE > BLOCK_B by splitting the batch into batch blocks.
    """
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    is_q = offs_n < N_Q
    is_v = offs_n >= (N_Q + N_KV)
    threshold_n = tl.where(is_q, threshold_q, tl.where(is_v, threshold_v, threshold_k))

    x = tl.load(
        X + offs_b[:, None] * M + offs_d[None, :],
        mask=(offs_b[:, None] < BATCHSIZE) & (offs_d[None, :] < M),
        other=0.0,
    )

    shared_score = tl.max(tl.abs(x), axis=0)  # [BLOCK_D]

    # Conservative block-skip:
    # Q/K/V thresholds can differ across output columns. Use the minimum threshold
    # across this output tile to decide whether the D-block might matter at all.
    threshold_min = tl.min(threshold_n, axis=0)
    block_active = shared_score > threshold_min
    active_count = tl.sum(block_active.to(tl.int32), axis=0)

    if active_count < MIN_ACTIVE:
        return

    # Fine-grained Q/K/V masking after the block passes coarse skip.
    active = shared_score[:, None] > threshold_n[None, :]  # [BLOCK_D, BLOCK_N]

    w = tl.load(
        W + offs_d[:, None] * N + offs_n[None, :],
        mask=(offs_d[:, None] < M) & (offs_n[None, :] < N),
        other=0.0,
    )

    w = tl.where(active, w, 0.0).to(tl.float16)
    acc = tl.dot(x.to(tl.float16), w)

    tl.atomic_add(
        Y + offs_b[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_b[:, None] < BATCHSIZE) & (offs_n[None, :] < N),
    )


def _get_block_config():
    """
    Env-var controlled kernel config.

    Suggested starting points:
      TEAL_BLOCK_B=16
      TEAL_BLOCK_N=128
      TEAL_BLOCK_D=64
      TEAL_BLOCK_MIN_ACTIVE=1

    For B=32, this patched file now works correctly with TEAL_BLOCK_B=16
    because the grid has a batch-block dimension.
    """
    block_b = int(os.environ.get("TEAL_BLOCK_B", "16"))
    block_n = int(os.environ.get("TEAL_BLOCK_N", "128"))
    block_d = int(os.environ.get("TEAL_BLOCK_D", "64"))
    min_active = int(os.environ.get("TEAL_BLOCK_MIN_ACTIVE", "1"))

    return block_b, block_n, block_d, min_active


def shared_mask_sparse_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
    sparsity_bin: int,
) -> torch.Tensor:
    """
    x:      [B, 1, M]
    weight: [N, M]
    output: [B, 1, N]

    Fused Triton shared-mask sparse GEMM with block skipping.
    """
    B, S, M = x.shape
    N, M_w = weight.shape

    assert S == 1, "shared_mask_sparse_gemm is decode-only"
    assert M == M_w
    assert weight.stride(1) > 1, "weight should be column-major style"

    x = x.contiguous()

    output = torch.empty(B, S, N, device=x.device, dtype=torch.float16)
    output.zero_()

    BLOCK_B, BLOCK_N, BLOCK_D, min_active = _get_block_config()

    grid = (
        triton.cdiv(N, BLOCK_N),
        triton.cdiv(M, BLOCK_D),
        triton.cdiv(B, BLOCK_B),
    )

    shared_mask_sparse_gemm_kernel[grid](
        output,
        weight,
        x,
        threshold,
        N,
        M,
        B,
        min_active,
        BLOCK_B,
        BLOCK_N,
        BLOCK_D,
    )

    if x.dtype is not output.dtype:
        return output.to(dtype=x.dtype)

    return output


def shared_mask_sparse_qkv_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold_q: float,
    threshold_k: float,
    threshold_v: float,
    sparsity_bin: int,
    kv_size: int,
) -> torch.Tensor:
    """
    x:      [B, 1, M]
    weight: [N_total, M]
    output: [B, 1, N_total]
    """
    B, S, M = x.shape
    N, M_w = weight.shape

    assert S == 1, "shared_mask_sparse_qkv_gemm is decode-only"
    assert M == M_w
    assert weight.stride(1) > 1, "weight should be column-major style"

    x = x.contiguous()

    N_Q = N - 2 * kv_size
    N_KV = kv_size

    output = torch.empty(B, S, N, device=x.device, dtype=torch.float16)
    output.zero_()

    BLOCK_B, BLOCK_N, BLOCK_D, min_active = _get_block_config()

    grid = (
        triton.cdiv(N, BLOCK_N),
        triton.cdiv(M, BLOCK_D),
        triton.cdiv(B, BLOCK_B),
    )

    shared_mask_sparse_qkv_gemm_kernel[grid](
        output,
        weight,
        x,
        threshold_q,
        threshold_k,
        threshold_v,
        N,
        N_Q,
        N_KV,
        M,
        B,
        min_active,
        BLOCK_B,
        BLOCK_N,
        BLOCK_D,
    )

    if x.dtype is not output.dtype:
        return output.to(dtype=x.dtype)

    return output


class SharedMaskSparseGEMM(BaseKernel):
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
            return shared_mask_sparse_gemm(
                hidden_states,
                weights,
                threshold,
                sparsity_bin,
            )

        return torch.matmul(hidden_states, weights.T)


class SharedMaskSparseQKVGEMM(BaseKernel):
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
            return shared_mask_sparse_qkv_gemm(
                x,
                weight,
                threshold_q,
                threshold_k,
                threshold_v,
                sparsity_bin,
                kv_size,
            )

        return torch.matmul(x, weight.T)