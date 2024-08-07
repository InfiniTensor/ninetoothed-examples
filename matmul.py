import ninetoothed
import ninetoothed.language as ntl
import torch
import triton
from ninetoothed import Symbol, Tensor

BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", meta=True)
BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", meta=True)
BLOCK_SIZE_K = Symbol("BLOCK_SIZE_K", meta=True)

output_tiled = Tensor(2).tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

lhs_tiled = (
    Tensor(2)
    .tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
    .tile((1, -1))
    .expand((-1, output_tiled.shape[1]))
)
rhs_tiled = (
    Tensor(2)
    .tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
    .tile((-1, 1))
    .expand((output_tiled.shape[0], -1))
)


@ninetoothed.jit
def matmul_kernel(lhs: lhs_tiled, rhs: rhs_tiled, output: output_tiled):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    for k in range(lhs.shape[1]):
        accumulator = ntl.dot(lhs[0, k], rhs[k, 0], accumulator)
    output = accumulator.to(ntl.float16)


def matmul(lhs, rhs):
    output = torch.empty(
        (lhs.shape[0], rhs.shape[1]), device=lhs.device, dtype=torch.float16
    )

    matmul_kernel(lhs, rhs, output)

    return output


torch.manual_seed(0)
shape = (512, 512)
lhs = torch.randn(shape, device="cuda", dtype=torch.float16)
rhs = torch.randn(shape, device="cuda", dtype=torch.float16)
torch_output = torch.matmul(lhs, rhs)
ninetoothed_output = matmul(lhs, rhs)
print(torch_output)
print(ninetoothed_output)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg="provider",
        line_vals=["ninetoothed", "cublas"],
        line_names=["NineToothed", "cuBLAS"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="TFLOPS",
        plot_name="matrix-multiplication-performance",
        args={},
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "cublas":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b), quantiles=quantiles
        )
    if provider == "ninetoothed":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul(a, b), quantiles=quantiles
        )

    def perf(ms):
        return 2 * M * N * K * 1e-12 / (ms * 1e-3)

    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True, save_path=".")
