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
    mat1_ptr,
    mat2_ptr,
    output_ptr,
    m,
    n,
    k,
    input_stride_m,
    input_stride_n,
    mat1_stride_m,
    mat1_stride_k,
    mat2_stride_k,
    mat2_stride_n,
    output_stride_m,
    output_stride_n,
    beta,
    alpha,
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
    mat1_ptrs = mat1_ptr + (
        offs_am[:, None] * mat1_stride_m + offs_k[None, :] * mat1_stride_k
    )
    mat2_ptrs = mat2_ptr + (
        offs_k[:, None] * mat2_stride_k + offs_bn[None, :] * mat2_stride_n
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
        mat1 = tl.load(mat1_ptrs, mask=offs_k[None, :] < k - i * BLOCK_SIZE_K)
        mat2 = tl.load(mat2_ptrs, mask=offs_k[:, None] < k - i * BLOCK_SIZE_K)

        accumulator = tl.dot(mat1, mat2, accumulator)

        mat1_ptrs += BLOCK_SIZE_K * mat1_stride_k
        mat2_ptrs += BLOCK_SIZE_K * mat2_stride_k

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_c = (offs_cm[:, None] < m) & (offs_cn[None, :] < n)

    input_ptrs = (
        input_ptr
        + input_stride_m * offs_cm[:, None]
        + input_stride_n * offs_cn[None, :]
    )

    input = tl.load(input_ptrs, mask=mask_c)

    output = beta * input + alpha * accumulator.to(tl.float16)

    output_ptrs = (
        output_ptr
        + output_stride_m * offs_cm[:, None]
        + output_stride_n * offs_cn[None, :]
    )

    tl.store(output_ptrs, output, mask=mask_c)
