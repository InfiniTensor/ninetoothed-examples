import ninetoothed
import ninetoothed.language as ntl
import torch
import triton
import triton.language as tl
from ninetoothed import Symbol, Tensor


def arrangement(lhs, rhs, output):
    BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", constexpr=True)
    BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", constexpr=True)
    BLOCK_SIZE_K = Symbol("BLOCK_SIZE_K", constexpr=True)

    output_tiled = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

    lhs_tiled = (
        lhs.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
        .tile((1, -1))
        .expand((-1, output_tiled.shape[1]))
    )
    lhs_tiled.dtype = lhs_tiled.dtype.squeeze(0)

    rhs_tiled = (
        rhs.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
        .tile((-1, 1))
        .expand((output_tiled.shape[0], -1))
    )
    rhs_tiled.dtype = rhs_tiled.dtype.squeeze(1)

    return lhs_tiled, rhs_tiled, output_tiled


def application(lhs, rhs, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    for k in range(lhs.shape[0]):
        accumulator += ntl.dot(lhs[k], rhs[k])
    output = accumulator.to(ntl.float16)


matmul_kernel = ninetoothed.make(
    arrangement, application, (Tensor(2), Tensor(2), Tensor(2))
)


def matmul(lhs, rhs):
    output = torch.empty(
        (lhs.shape[0], rhs.shape[1]), device=lhs.device, dtype=torch.float16
    )

    matmul_kernel(lhs, rhs, output, BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64)

    return output


@triton.jit
def triton_matmul_kernel(
    lhs_ptr,
    rhs_ptr,
    output_ptr,
    m,
    n,
    k,
    lhs_stride_m,
    lhs_stride_k,
    rhs_stride_k,
    rhs_stride_n,
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
    lhs_ptrs = lhs_ptr + (
        offs_am[:, None] * lhs_stride_m + offs_k[None, :] * lhs_stride_k
    )
    rhs_ptrs = rhs_ptr + (
        offs_k[:, None] * rhs_stride_k + offs_bn[None, :] * rhs_stride_n
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
        lhs = tl.load(lhs_ptrs, mask=offs_k[None, :] < k - i * BLOCK_SIZE_K)
        rhs = tl.load(rhs_ptrs, mask=offs_k[:, None] < k - i * BLOCK_SIZE_K)
        accumulator = tl.dot(lhs, rhs, accumulator)
        lhs_ptrs += BLOCK_SIZE_K * lhs_stride_k
        rhs_ptrs += BLOCK_SIZE_K * rhs_stride_k
    output = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = (
        output_ptr
        + output_stride_m * offs_cm[:, None]
        + output_stride_n * offs_cn[None, :]
    )
    output_mask = (offs_cm[:, None] < m) & (offs_cn[None, :] < n)
    tl.store(output_ptrs, output, mask=output_mask)


def triton_matmul(lhs, rhs):
    output = torch.empty(
        (lhs.shape[0], rhs.shape[1]), device=lhs.device, dtype=torch.float16
    )

    def grid(meta):
        return (
            triton.cdiv(lhs.shape[0], meta["BLOCK_SIZE_M"])
            * triton.cdiv(rhs.shape[1], meta["BLOCK_SIZE_N"]),
        )

    triton_matmul_kernel[grid](
        lhs,
        rhs,
        output,
        lhs.shape[0],
        rhs.shape[1],
        lhs.shape[1],
        lhs.stride(0),
        lhs.stride(1),
        rhs.stride(0),
        rhs.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
    )

    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    shape = (512, 512)
    lhs = torch.randn(shape, device="cuda", dtype=torch.float16)
    rhs = torch.randn(shape, device="cuda", dtype=torch.float16)
    ninetoothed_output = matmul(lhs, rhs)
    torch_output = torch.matmul(lhs, rhs)
    triton_output = triton_matmul(lhs, rhs)
    print(ninetoothed_output)
    print(torch_output)
    print(triton_output)
    if torch.allclose(ninetoothed_output, torch_output):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")
    if torch.allclose(ninetoothed_output, triton_output, atol=0, rtol=0):
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
            plot_name="matrix-multiplication-performance",
            args={},
        )
    )
    def benchmark(m, n, k, provider):
        lhs = torch.randn((m, k), device="cuda", dtype=torch.float16)
        rhs = torch.randn((k, n), device="cuda", dtype=torch.float16)
        quantiles = [0.5, 0.2, 0.8]

        ninetoothed_output = matmul(lhs, rhs)
        torch_output = torch.matmul(lhs, rhs)
        triton_output = triton_matmul(lhs, rhs)
        assert torch.allclose(ninetoothed_output, torch_output, atol=0.025, rtol=0.025)
        assert torch.allclose(ninetoothed_output, triton_output, atol=0, rtol=0)

        if provider == "ninetoothed":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: matmul(lhs, rhs), quantiles=quantiles
            )
        elif provider == "torch":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.matmul(lhs, rhs), quantiles=quantiles
            )
        elif provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: triton_matmul(lhs, rhs), quantiles=quantiles
            )

        def perf(ms):
            return 2 * m * n * k * 1e-12 / (ms * 1e-3)

        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(show_plots=True, print_data=True, save_path=".")
