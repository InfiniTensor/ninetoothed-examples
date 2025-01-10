import ninetoothed
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from ninetoothed import Tensor

import matmul


def arrangement(input, filter, output):
    input_tiled = input.tile((1, *filter.shape[1:]), strides=(-1, -1, 1, 1))
    input_squeezed = input_tiled.squeeze(1)
    input_squeezed.dtype = input_squeezed.dtype.squeeze(0)
    input_raveled = input_squeezed.ravel()
    input_flattened = input_raveled.flatten(end_dim=3).flatten(start_dim=1)

    filter_flattened = filter.flatten(start_dim=1)
    filter_permuted = filter_flattened.permute((1, 0))

    output_flattened = output.permute((0, 2, 3, 1)).flatten(end_dim=3)

    return matmul.arrangement(input_flattened, filter_permuted, output_flattened)


conv2d_kernel = ninetoothed.make(
    arrangement,
    matmul.application,
    (Tensor(4), Tensor(4, constexpr_shape=True), Tensor(4)),
)


def conv2d(input, filter):
    n, _, h, w = input.shape
    k, _, r, s = filter.shape
    p = h - r + 1
    q = w - s + 1

    output = torch.empty((n, k, p, q), device=input.device, dtype=input.dtype)

    conv2d_kernel(
        input, filter, output, BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64
    )

    return output


@triton.jit
def triton_conv2d_kernel(
    input_ptr,
    filter_ptr,
    output_ptr,
    n,
    c,
    h,
    w,
    k,
    r,
    s,
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
    GROUP_SIZE_M: tl.constexpr,
):
    p = h - r + 1
    q = w - s + 1

    gemm_m = n * p * q
    gemm_n = k
    gemm_k = c * r * s

    pid = tl.program_id(0)
    num_pid_gemm_m = tl.cdiv(gemm_m, BLOCK_SIZE_M)
    num_pid_gemm_n = tl.cdiv(gemm_n, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_gemm_n
    group_id = pid // num_pid_in_group
    first_pid_gemm_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_gemm_m - first_pid_gemm_m, GROUP_SIZE_M)
    pid_gemm_m = first_pid_gemm_m + ((pid % num_pid_in_group) % group_size_m)
    pid_gemm_n = (pid % num_pid_in_group) // group_size_m

    offs_gemm_i = pid_gemm_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_gemm_j = pid_gemm_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offs_n = offs_gemm_i // (p * q)
    offs_k = offs_gemm_j
    npq_residual = offs_gemm_i % (p * q)
    offs_p = npq_residual // q
    offs_q = npq_residual % q

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, tl.cdiv(gemm_k, BLOCK_SIZE_K)):
        offs_gemm_k = i * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        offs_c = offs_gemm_k // (r * s)
        crs_residual = offs_gemm_k % (r * s)
        offs_r = crs_residual // s
        offs_s = crs_residual % s

        offs_h = offs_p[:, None] + offs_r[None, :]
        offs_w = offs_q[:, None] + offs_s[None, :]

        input_ptrs = (
            input_ptr
            + offs_n[:, None] * input_stride_n
            + offs_c[None, :] * input_stride_c
            + offs_h * input_stride_h
            + offs_w * input_stride_w
        )
        input_mask = (
            (offs_n[:, None] < n) & (offs_c[None, :] < c) & (offs_h < h) & (offs_w < w)
        )

        filter_ptrs = (
            filter_ptr
            + offs_k[None, :] * filter_stride_k
            + offs_c[:, None] * filter_stride_c
            + offs_r[:, None] * filter_stride_r
            + offs_s[:, None] * filter_stride_s
        )
        filter_mask = (offs_k[None, :] < k) & (
            (offs_c < c) & (offs_r < r) & (offs_s < s)
        )[:, None]

        input = tl.load(input_ptrs, mask=input_mask, other=0.0)
        filter = tl.load(filter_ptrs, mask=filter_mask, other=0.0)

        accumulator = tl.dot(input, filter, accumulator)

    output = accumulator.to(tl.float16)

    output_ptrs = (
        output_ptr
        + offs_n[:, None] * output_stride_n
        + offs_k[None, :] * output_stride_k
        + offs_p[:, None] * output_stride_p
        + offs_q[:, None] * output_stride_q
    )
    output_mask = (
        (offs_n[:, None] < n)
        & (offs_k[None, :] < k)
        & (offs_p[:, None] < p)
        & (offs_q[:, None] < q)
    )

    tl.store(output_ptrs, output, mask=output_mask)


def triton_conv2d(input, filter):
    n, c, h, w = input.shape
    k, _, r, s = filter.shape
    p = h - r + 1
    q = w - s + 1

    output = torch.empty((n, k, p, q), device=input.device, dtype=input.dtype)

    def grid(meta):
        return (
            triton.cdiv(n * p * q, meta["BLOCK_SIZE_M"])
            * triton.cdiv(k, meta["BLOCK_SIZE_N"]),
        )

    triton_conv2d_kernel[grid](
        input,
        filter,
        output,
        n,
        c,
        h,
        w,
        k,
        r,
        s,
        *input.stride(),
        *filter.stride(),
        *output.stride(),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
    )

    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    n, c, h, w = 4, 3, 224, 224
    k, _, r, s = 8, c, 3, 3
    input = torch.randn(n, c, h, w, device="cuda")
    filter = torch.randn(k, c, r, s, device="cuda")
    ninetoothed_output = conv2d(input, filter)
    torch_output = F.conv2d(input, filter)
    triton_output = triton_conv2d(input, filter)
    print(ninetoothed_output)
    print(torch_output)
    print(triton_output)
    if torch.allclose(ninetoothed_output, torch_output, atol=0.01, rtol=0.01):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")
    if torch.allclose(ninetoothed_output, triton_output, atol=0.01, rtol=0.01):
        print("✅ NineToothed and Triton match.")
    else:
        print("❌ NineToothed and Triton differ.")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["h", "w"],
            x_vals=[8 * i for i in range(2, 33)],
            line_arg="provider",
            line_vals=["ninetoothed", "torch", "triton"],
            line_names=["NineToothed", "PyTorch", "Triton"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="TFLOPS",
            plot_name="2d-convolution-performance",
            args={},
        )
    )
    def benchmark(h, w, provider):
        n, c, _, _ = 64, 3, h, w
        k, _, r, s = 64, c, 3, 3
        input = torch.randn((n, c, h, w), device="cuda")
        filter = torch.randn((k, c, r, s), device="cuda")

        if provider == "ninetoothed":
            ms = triton.testing.do_bench(lambda: conv2d(input, filter))
        elif provider == "torch":
            ms = triton.testing.do_bench(lambda: F.conv2d(input, filter))
        elif provider == "triton":
            ms = triton.testing.do_bench(lambda: triton_conv2d(input, filter))

        def perf(ms):
            p = h - r + 1
            q = w - s + 1

            return 2 * n * k * p * q * c * r * s * 1e-12 / (ms * 1e-3)

        return perf(ms)

    benchmark.run(show_plots=True, print_data=True, save_path=".")
