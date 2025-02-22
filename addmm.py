import random

import ninetoothed
import torch
import triton
import triton.language as tl
from ninetoothed import Tensor

import matmul


def arrangement(input, mat1, mat2, beta, alpha, output):
    _, _, input_arranged = matmul.arrangement(mat1, mat2, input)

    mat1_arrange, mat2_arranged, output_arranged = matmul.arrangement(
        mat1, mat2, output
    )

    return input_arranged, mat1_arrange, mat2_arranged, beta, alpha, output_arranged


def application(input, mat1, mat2, beta, alpha, output):
    matmul.application(mat1, mat2, output)
    output = beta * input + alpha * output


addmm_kernel = ninetoothed.make(
    arrangement,
    application,
    (Tensor(2), Tensor(2), Tensor(2), Tensor(0), Tensor(0), Tensor(2)),
)


def addmm(input, mat1, mat2, beta=1, alpha=1):
    output = torch.empty(
        (mat1.shape[0], mat2.shape[1]), dtype=torch.float16, device=mat1.device
    )

    addmm_kernel(input, mat1, mat2, beta, alpha, output)

    return output


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
    ],
    key=["m", "n", "k"],
)
@triton.jit
def triton_addmm_kernel(
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
        mat1 = tl.load(
            mat1_ptrs, mask=offs_k[None, :] < k - i * BLOCK_SIZE_K, other=0.0
        )
        mat2 = tl.load(
            mat2_ptrs, mask=offs_k[:, None] < k - i * BLOCK_SIZE_K, other=0.0
        )
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


def triton_addmm(input, mat1, mat2, beta=1, alpha=1):
    output = torch.empty(
        (mat1.shape[0], mat2.shape[1]), dtype=torch.float16, device=mat1.device
    )

    def grid(meta):
        return (
            triton.cdiv(mat1.shape[0], meta["BLOCK_SIZE_M"])
            * triton.cdiv(mat2.shape[1], meta["BLOCK_SIZE_N"]),
        )

    triton_addmm_kernel[grid](
        input,
        mat1,
        mat2,
        output,
        mat1.shape[0],
        mat2.shape[1],
        mat1.shape[1],
        input.stride(0),
        input.stride(1),
        mat1.stride(0),
        mat1.stride(1),
        mat2.stride(0),
        mat2.stride(1),
        output.stride(0),
        output.stride(1),
        beta,
        alpha,
    )

    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    shape = (512, 512)
    input = torch.randn(shape, dtype=torch.float16, device="cuda")
    mat1 = torch.randn(shape, dtype=torch.float16, device="cuda")
    mat2 = torch.randn(shape, dtype=torch.float16, device="cuda")
    beta = random.uniform(0, 1)
    alpha = random.uniform(0, 1)
    ninetoothed_output = addmm(input, mat1, mat2, beta=beta, alpha=alpha)
    torch_output = torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha)
    triton_output = triton_addmm(input, mat1, mat2, beta=beta, alpha=alpha)
    print(ninetoothed_output)
    print(torch_output)
    print(triton_output)
    if torch.allclose(ninetoothed_output, torch_output, atol=0.01, rtol=0.01):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")
    if torch.allclose(ninetoothed_output, triton_output):
        print("✅ NineToothed and Triton match.")
    else:
        print("❌ NineToothed and Triton differ.")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "n", "k"],
            x_vals=[128 * i for i in range(2, 33)],
            line_arg="provider",
            line_vals=["ninetoothed", "torch", "triton"],
            line_names=["NineToothed", "PyTorch", "Triton"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="TFLOPS",
            plot_name="addmm-performance",
            args={},
        )
    )
    def benchmark(m, n, k, provider):
        input = torch.randn((m, n), dtype=torch.float16, device="cuda")
        mat1 = torch.randn((m, k), dtype=torch.float16, device="cuda")
        mat2 = torch.randn((k, n), dtype=torch.float16, device="cuda")
        beta = random.uniform(0, 1)
        alpha = random.uniform(0, 1)
        quantiles = [0.5, 0.2, 0.8]

        if provider == "ninetoothed":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: addmm(input, mat1, mat2, beta=beta, alpha=alpha),
                quantiles=quantiles,
            )
        elif provider == "torch":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha),
                quantiles=quantiles,
            )
        elif provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: triton_addmm(input, mat1, mat2, beta=beta, alpha=alpha),
                quantiles=quantiles,
            )

        def perf(ms):
            return (2 * m * n * k + 3 * m * n) * 1e-12 / (ms * 1e-3)

        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(show_plots=True, print_data=True, save_path=".")
