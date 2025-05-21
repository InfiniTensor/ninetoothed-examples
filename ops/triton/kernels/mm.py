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
                "GROUP_SIZE_M": 8,
            },
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for block_size_m, block_size_n, block_size_k, num_stages, num_warps in itertools.product(
            (32, 64, 128), (32, 64, 128, 256), (32, 64), (3, 4, 5), (2, 4, 8)
        )
    ),
    key=["m", "n", "k"],
)
@triton.jit
def kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    m,
    n,
    k,
    input_stride_m,
    input_stride_k,
    other_stride_k,
    other_stride_n,
    output_stride_m,
    output_stride_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(m, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(n, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % m
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % n
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    input_ptrs = input_ptr + (
        offs_am[:, None] * input_stride_m + offs_k[None, :] * input_stride_k
    )
    other_ptrs = other_ptr + (
        offs_k[:, None] * other_stride_k + offs_bn[None, :] * other_stride_n
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
        input = tl.load(input_ptrs, mask=offs_k[None, :] < k - i * BLOCK_SIZE_K)
        other = tl.load(other_ptrs, mask=offs_k[:, None] < k - i * BLOCK_SIZE_K)

        accumulator = tl.dot(input, other, accumulator)

        input_ptrs += BLOCK_SIZE_K * input_stride_k
        other_ptrs += BLOCK_SIZE_K * other_stride_k

    output = accumulator

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = (
        output_ptr
        + output_stride_m * offs_cm[:, None]
        + output_stride_n * offs_cn[None, :]
    )

    tl.store(output_ptrs, output, mask=(offs_cm[:, None] < m) & (offs_cn[None, :] < n))
