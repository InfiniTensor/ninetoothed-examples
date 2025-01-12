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


tensors = (Tensor(4, constexpr_shape=True) for _ in range(3))
conv2d_kernel = ninetoothed.make(arrangement, matmul.application, tensors)


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
    GROUP_SIZE_M: tl.constexpr,
):
    P: tl.constexpr = H - R + 1
    Q: tl.constexpr = W - S + 1

    GEMM_M: tl.constexpr = N * P * Q
    GEMM_N: tl.constexpr = K
    GEMM_K: tl.constexpr = C * R * S

    pid = tl.program_id(0)
    num_pid_gemm_m = tl.cdiv(GEMM_M, BLOCK_SIZE_M)
    num_pid_gemm_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_gemm_n
    group_id = pid // num_pid_in_group
    first_pid_gemm_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_gemm_m - first_pid_gemm_m, GROUP_SIZE_M)
    pid_gemm_m = first_pid_gemm_m + ((pid % num_pid_in_group) % group_size_m)
    pid_gemm_n = (pid % num_pid_in_group) // group_size_m

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

    output = accumulator.to(tl.float16)

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
    dtype = torch.float16
    input = torch.randn(n, c, h, w, dtype=dtype, device="cuda")
    filter = torch.randn(k, c, r, s, dtype=dtype, device="cuda")
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
    if torch.allclose(ninetoothed_output, triton_output, atol=0, rtol=0):
        print("✅ NineToothed and Triton match.")
    else:
        print("❌ NineToothed and Triton differ.")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["n"],
            x_vals=[2**i for i in range(1, 11)],
            x_log=True,
            line_arg="provider",
            line_vals=["ninetoothed", "torch", "triton"],
            line_names=["NineToothed", "PyTorch", "Triton"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="TFLOPS",
            plot_name="2d-convolution-performance",
            args={},
        )
    )
    def benchmark(n, provider):
        _, c, h, w = n, 512, 14, 14
        k, _, r, s = 512, c, 3, 3
        dtype = torch.float16
        input = torch.randn((n, c, h, w), dtype=dtype, device="cuda")
        filter = torch.randn((k, c, r, s), dtype=dtype, device="cuda")

        ninetoothed_output = conv2d(input, filter)
        torch_output = F.conv2d(input, filter)
        triton_output = triton_conv2d(input, filter)
        assert torch.allclose(ninetoothed_output, torch_output, atol=0.01, rtol=0.01)
        assert torch.allclose(ninetoothed_output, triton_output, atol=0, rtol=0)

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
