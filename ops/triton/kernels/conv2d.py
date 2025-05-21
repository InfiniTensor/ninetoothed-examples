import itertools

import triton
import triton.language as tl


@triton.autotune(
    configs=tuple(
        triton.Config(
            {
                "BLOCK_SIZE_M": block_size_m,
                "BLOCK_SIZE_N": block_size_n,
                "BLOCK_SIZE_K": block_size_k,
            },
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for block_size_m, block_size_n, block_size_k, num_stages, num_warps in itertools.product(
            (32, 64, 128), (32, 64, 128, 256), (32, 64), (3, 4, 5), (2, 4, 8)
        )
    ),
    key=["N", "C", "H", "W", "C", "R", "S"],
)
@triton.jit
def kernel(
    input_ptr,
    filter_ptr,
    output_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
    input_stride_n,
    input_stride_c,
    input_stride_h,
    input_stride_w,
    filter_stride_k,
    filter_stride_c,
    filter_stride_r,
    filter_stride_s,
    output_stride_n,
    output_stride_k,
    output_stride_p,
    output_stride_q,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    P: tl.constexpr = H - R + 1
    Q: tl.constexpr = W - S + 1

    GEMM_N: tl.constexpr = K
    GEMM_K: tl.constexpr = C * R * S

    pid = tl.program_id(0)
    num_pid_gemm_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)
    pid_gemm_m = pid // num_pid_gemm_n
    pid_gemm_n = pid % num_pid_gemm_n

    offs_gemm_i = pid_gemm_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_gemm_j = pid_gemm_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offs_n = offs_gemm_i // (P * Q)
    offs_k = offs_gemm_j
    npq_residual = offs_gemm_i % (P * Q)
    offs_p = npq_residual // Q
    offs_q = npq_residual % Q

    input_offs_gemm_m = (
        offs_n * input_stride_n + offs_p * input_stride_h + offs_q * input_stride_w
    )[:, None]
    filter_offs_gemm_n = (offs_k * filter_stride_k)[None, :]

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):
        offs_gemm_k = i * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        offs_c = offs_gemm_k // (R * S)
        crs_residual = offs_gemm_k % (R * S)
        offs_r = crs_residual // S
        offs_s = crs_residual % S

        input_offs_gemm_n = (
            offs_c * input_stride_c + offs_r * input_stride_h + offs_s * input_stride_w
        )[None, :]
        input_ptrs = input_ptr + input_offs_gemm_m + input_offs_gemm_n
        input_mask = ((offs_n < N) & (offs_p < P) & (offs_q < Q))[:, None] & (
            (offs_c < C) & (offs_r < R) & (offs_s < S)
        )[None, :]

        input = tl.load(input_ptrs, mask=input_mask)

        filter_offs_gemm_m = (
            offs_c * filter_stride_c
            + offs_r * filter_stride_r
            + offs_s * filter_stride_s
        )[:, None]
        filter_ptrs = filter_ptr + filter_offs_gemm_m + filter_offs_gemm_n
        filter_mask = (offs_k[None, :] < K) & (
            (offs_c < C) & (offs_r < R) & (offs_s < S)
        )[:, None]

        filter = tl.load(filter_ptrs, mask=filter_mask)

        accumulator = tl.dot(input, filter, accumulator)

    output = accumulator

    output_ptrs = (
        output_ptr
        + (
            offs_n * output_stride_n
            + offs_p * output_stride_p
            + offs_q * output_stride_q
        )[:, None]
        + (offs_k * output_stride_k)[None, :]
    )
    output_mask = (
        (offs_n[:, None] < N)
        & (offs_k[None, :] < K)
        & (offs_p[:, None] < P)
        & (offs_q[:, None] < Q)
    )

    tl.store(output_ptrs, output, mask=output_mask)
