import ninetoothed
import torch
import triton
from ninetoothed import Symbol, Tensor


def add(lhs, rhs):
    BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)

    @ninetoothed.jit
    def add_kernel(
        lhs: Tensor(1).tile((BLOCK_SIZE,)),
        rhs: Tensor(1).tile((BLOCK_SIZE,)),
        output: Tensor(1).tile((BLOCK_SIZE,)),
    ):
        output = lhs + rhs  # noqa: F841

    output = torch.empty_like(lhs)

    add_kernel(lhs, rhs, output)

    return output


torch.manual_seed(0)
size = 98432
lhs = torch.rand(size, device="cuda")
rhs = torch.rand(size, device="cuda")
torch_output = lhs + rhs
ninetoothed_output = add(lhs, rhs)
print(torch_output)
print(ninetoothed_output)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg="provider",
        line_vals=["ninetoothed", "torch"],
        line_names=["NineToothed", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="vector-addition-performance",
        args={},
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device="cuda", dtype=torch.float32)
    y = torch.rand(size, device="cuda", dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == "ninetoothed":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add(x, y), quantiles=quantiles
        )

    def gbps(ms):
        return 3 * x.numel() * x.element_size() / ms * 1e-6

    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, show_plots=True)
