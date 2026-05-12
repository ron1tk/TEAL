# adapted from deja vu

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def init_to_zero(*names):
    def init_func(nargs):
        for name in names:
            nargs[name].zero_()
    return init_func


# NOTE: will need to warm up kernels each time, triton autotune caching isn't a thing right now
configs = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=2, pre_hook=init_to_zero("Y")),

    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=2, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 16}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=4, pre_hook=init_to_zero("Y")),

    triton.Config({"BLOCK_M": 128, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4, pre_hook=init_to_zero("Y")),

    # Llama 3 variants can use BLOCK_N >= 1024
    # triton.Config({"BLOCK_M": 128, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
    # triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
    # triton.Config({"BLOCK_M": 64, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
    # triton.Config({"BLOCK_M": 32, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
    # triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=4, pre_hook=init_to_zero("Y")),
]


@triton.autotune(
    configs=configs,
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE", "SPARSITY_BIN"],
)
@triton.jit
def splitk_sparse_gemv_kernel(
    Y,  # output
    A,
    X,
    threshold,
    # Matrix dimensions
    N,
    M,
    CACHE_KEY_N,
    CACHE_KEY_M,
    # Meta-parameters
    BATCHSIZE: tl.constexpr,
    SPARSITY_BIN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    start_n = tl.program_id(0)
    start_m = tl.program_id(1)
    batch_id = tl.program_id(2)

    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    A_ptr = A + (rm[:, None] * N + rn[None, :])

    # x shape is [B, 1, M], contiguous.
    # Since seq_len == 1 during decode, offset is batch_id * M + rm.
    X_ptr = X + batch_id * M + rm

    # output shape is [B, 1, N], contiguous.
    # Since seq_len == 1 during decode, offset is batch_id * N + rn.
    Y_ptr = Y + batch_id * N + rn

    if BATCHSIZE == 1:
        x0 = tl.load(X_ptr, mask=rm < M, other=0.0, eviction_policy="evict_last")
        idx = tl.abs(x0) > threshold

        a = tl.load(A_ptr, mask=idx[:, None], other=0.0, eviction_policy="evict_first")
        acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
    else:
        # Shared batch mask prototype:
        # Compute one mask for all batch rows using max(abs(x_b)) across batch.
        # This is NOT a full batched sparse GEMM. It still computes one output row
        # per batch_id, but all rows share the same active hidden dimensions.
        b = tl.arange(0, BATCHSIZE)

        X_all_ptr = X + b[:, None] * M + rm[None, :]
        x_all = tl.load(X_all_ptr, mask=rm[None, :] < M, other=0.0)

        shared_score = tl.max(tl.abs(x_all), axis=0)
        idx = shared_score > threshold

        x0 = tl.load(X_ptr, mask=rm < M, other=0.0, eviction_policy="evict_last")
        a = tl.load(A_ptr, mask=idx[:, None], other=0.0, eviction_policy="evict_first")
        acc0 = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)

    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    tl.atomic_add(Y_ptr, acc0, mask=rn < N)


# NOTE: assumes that weight is column major
def splitk_sparse_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold: float,
    sparsity_bin: int,
) -> torch.Tensor:
    """
    Compute y = sparse(X) @ weight.

    x:      [B, 1, Z]
    weight: [N, Z]
    output: [B, 1, N]

    For B=1, this is original TEAL-style sparse GEMV.
    For B>1, this prototype uses a shared activation mask across batch rows,
    but still computes each batch row separately.
    """
    N, Z = weight.shape
    beam_width, seq_len, _ = x.shape

    assert seq_len == 1, "sparse GEMV path is decode-only; prefill should use matmul fallback"
    assert x.shape[2] == Z
    assert weight.stride(1) > 1, "weight should be column major"

    x = x.contiguous()

    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_N"]),
        triton.cdiv(Z, META["BLOCK_M"]),
        beam_width,
    )

    output = torch.empty(
        beam_width,
        seq_len,
        N,
        device=x.device,
        dtype=torch.float16,
    )

    kernel = splitk_sparse_gemv_kernel
    kernel[grid](
        output,
        weight,
        x,
        threshold,
        N,
        Z,
        N // 16,
        Z // 16,
        beam_width,
        sparsity_bin,
    )

    if x.dtype is not output.dtype:
        print(
            f"Warning: incurring dtype conversion overhead since input dtype is not torch.float16. "
            f"Detected dtype: {x.dtype}."
        )
        return output.to(dtype=x.dtype)

    return output


# fused implementation of qkv with three thresholds
# is unnecessary for uniform but is needed for block-wise greedy
@triton.autotune(
    configs=configs,
    key=["CACHE_KEY_M", "CACHE_KEY_N", "BATCHSIZE", "SPARSITY_BIN"],
)
@triton.jit
def qkv_kernel(
    Y,
    A,
    X,
    threshold_q,
    threshold_k,
    threshold_v,
    # Matrix dimensions
    N,
    N_q,
    N_kv,
    M,
    CACHE_KEY_N,
    CACHE_KEY_M,
    # Meta-parameters
    BATCHSIZE: tl.constexpr,
    SPARSITY_BIN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    start_n = tl.program_id(0)
    start_m = tl.program_id(1)
    batch_id = tl.program_id(2)

    is_q = start_n * BLOCK_N < N_q
    is_v = N_q + N_kv <= start_n * BLOCK_N

    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    A_ptr = A + rm[:, None] * N + rn[None, :]

    # x shape is [B, 1, M], contiguous.
    X_ptr = X + batch_id * M + rm

    # output shape is [B, 1, N], contiguous.
    Y_ptr = Y + batch_id * N + rn

    threshold = tl.where(is_q, threshold_q, tl.where(is_v, threshold_v, threshold_k))

    if BATCHSIZE == 1:
        x0 = tl.load(X_ptr, mask=rm < M, other=0.0, eviction_policy="evict_last")
        idx = tl.abs(x0) > threshold

        a = tl.load(A_ptr, mask=idx[:, None], other=0.0, eviction_policy="evict_first")
        acc = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)
    else:
        # Shared batch mask prototype:
        # Compute one mask for all batch rows using max(abs(x_b)) across batch.
        #
        # For QKV, threshold depends on whether this output block belongs to Q, K, or V.
        # All batch rows share that projection-specific mask for this block.
        b = tl.arange(0, BATCHSIZE)

        X_all_ptr = X + b[:, None] * M + rm[None, :]
        x_all = tl.load(X_all_ptr, mask=rm[None, :] < M, other=0.0)

        shared_score = tl.max(tl.abs(x_all), axis=0)
        idx = shared_score > threshold

        x0 = tl.load(X_ptr, mask=rm < M, other=0.0, eviction_policy="evict_last")
        a = tl.load(A_ptr, mask=idx[:, None], other=0.0, eviction_policy="evict_first")
        acc = tl.sum(a.to(tl.float32) * x0.to(tl.float32)[:, None], 0)

    rn = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rn < N

    tl.atomic_add(Y_ptr, acc, mask=mask_n)


def qkv_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    threshold_q: float,
    threshold_k: float,
    threshold_v: float,
    sparsity_bin: int,
    kv_size: int,
):
    """
    x:      [B, 1, Z]
    weight: [N, Z]
    output: [B, 1, N]

    For B=1, this is original TEAL-style sparse QKV GEMV.
    For B>1, this prototype uses one shared activation mask across batch rows.
    """
    N, Z = weight.shape
    beam_width, seq_len, _ = x.shape

    assert seq_len == 1, "sparse QKV GEMV path is decode-only; prefill should use matmul fallback"
    assert x.shape[2] == Z
    assert weight.stride(1) > 1, "weights should be column major"

    x = x.contiguous()

    N_q = N - 2 * kv_size
    N_k = kv_size

    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_N"]),
        triton.cdiv(Z, META["BLOCK_M"]),
        beam_width,
    )

    output = torch.empty(beam_width, seq_len, N, device=x.device, dtype=torch.float16)

    qkv_kernel[grid](
        output,
        weight,
        x,
        threshold_q,
        threshold_k,
        threshold_v,
        N,
        N_q,
        N_k,
        Z,
        N // 16,
        Z // 16,
        beam_width,
        sparsity_bin,
    )

    if x.dtype is not output.dtype:
        print(f"Warning: incurring dtype conversion overhead. Input dtype: {x.dtype}")
        return output.to(dtype=x.dtype)

    return output


import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "kernels"))

from compile_wrapper import BaseKernel


# wrappers for compatibility with torch.compile
class SparseGEMV(BaseKernel):
    def meta(
        self,
        # hidden_states -> [B, seq, D]
        hidden_states: torch.Tensor,
        # weights -> [I, D]
        weights: torch.Tensor,
        threshold: float,
        sparsity_bin: int,
    ) -> torch.Tensor:
        return hidden_states.new_empty((hidden_states.size(0), hidden_states.size(1), weights.size(0)))

    def forward(
        self,
        # hidden_states -> [B, seq, D]
        hidden_states: torch.Tensor,
        # weights -> [I, D]
        weights: torch.Tensor,
        threshold: float,
        sparsity_bin: int,
    ) -> torch.Tensor:
        if hidden_states.shape[1] == 1:
            return splitk_sparse_gemv(hidden_states, weights, threshold, sparsity_bin)

        # prefill fallback
        return torch.matmul(hidden_states, weights.T)


class SparseQKVGEMV(BaseKernel):
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
            return qkv_gemv(x, weight, threshold_q, threshold_k, threshold_v, sparsity_bin, kv_size)

        # prefill fallback
        return torch.matmul(x, weight.T)


# for testing purposes, to see if overhead at 0% is really due to strengthening torch.matmul
class DenseGEMV(BaseKernel):
    def meta(self, x: torch.Tensor, W: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x.new_empty(x.shape[0], x.shape[1], W.shape[0])

    def forward(self, x: torch.Tensor, W: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.matmul(x, W.T)