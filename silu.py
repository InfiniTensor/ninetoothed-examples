from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

import ops.ninetoothed.torch
import ops.triton.torch


class SiLU(nn.Module):
    silu = None

    def __init__(self, other):
        super().__init__()

        self.__dict__ = other.__dict__

    def forward(self, input):
        return type(self).silu(input)


@contextmanager
def silu_backend(backend_name):
    _prev_impl = SiLU.silu

    if backend_name == "ninetoothed":
        impl = ops.ninetoothed.torch.silu
    elif backend_name == "triton":
        impl = ops.triton.torch.silu
    elif backend_name == "torch":
        impl = F.silu
    else:
        raise ValueError(f"unknown backend: `{backend_name}`")

    SiLU.silu = impl

    try:
        yield
    finally:
        SiLU.silu = _prev_impl


if __name__ == "__main__":
    torch.manual_seed(0)

    shape = (8, 256, 512)
    dtype = torch.float16
    device = "cuda"

    input = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ops.ninetoothed.torch.silu(input)
    torch_output = F.silu(input)
    triton_output = ops.triton.torch.silu(input)

    print(ninetoothed_output)
    print(torch_output)
    print(triton_output)

    if torch.allclose(ninetoothed_output, torch_output, atol=1e-3, rtol=1e-3):
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
            x_vals=[2**i for i in range(3, 10)],
            x_log=True,
            line_arg="provider",
            line_vals=["ninetoothed", "torch", "triton"],
            line_names=["NineToothed", "PyTorch", "Triton"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="ms",
            plot_name="silu-performance",
            args={},
        )
    )
    def benchmark(m, n, k, provider):
        input = torch.randn(m, n, k, dtype=dtype, device=device)

        ninetoothed_output = ops.ninetoothed.torch.silu(input)
        torch_output = F.silu(input)
        triton_output = ops.triton.torch.silu(input)

        assert torch.allclose(ninetoothed_output, torch_output, atol=0.001)
        assert torch.allclose(ninetoothed_output, triton_output, atol=0, rtol=0)

        if provider == "ninetoothed":
            ms = triton.testing.do_bench(lambda: ops.ninetoothed.torch.silu(input))
        elif provider == "torch":
            ms = triton.testing.do_bench(lambda: F.silu(input))
        elif provider == "triton":
            ms = triton.testing.do_bench(lambda: ops.triton.torch.silu(input))

        return ms

    benchmark.run(show_plots=True, print_data=True, save_path=".")
