from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

import ops.ninetoothed.torch
import ops.triton.torch
import ntops.torch

class AvgPool2d(nn.Module):
    avg_pool2d = None

    def __init__(self, other):
        super().__init__()

        self.__dict__ = other.__dict__

    def forward(self, input):
        def _pair(x):
            return (x, x) if isinstance(x, int) else x
        
        return type(self).avg_pool2d(
            input,
            kernel_size=_pair(self.kernel_size),
            stride=_pair(self.stride) if self.stride else _pair(self.kernel_size),
            padding=_pair(self.padding),
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override,
        )


@contextmanager
def avg_pool2d_backend(backend_name):
    _prev_impl = AvgPool2d.avg_pool2d

    if backend_name == "ninetoothed":
        impl = ntops.torch.avg_pool2d
    elif backend_name == "triton":
        impl = ops.triton.torch.avg_pool2d
    elif backend_name == "torch":
        impl = F.avg_pool2d
    else:
        raise ValueError(f"unknown backend: `{backend_name}`")

    AvgPool2d.avg_pool2d = impl

    try:
        yield
    finally:
        AvgPool2d.avg_pool2d = _prev_impl

if __name__ == "__main__":
    torch.manual_seed(0)

    input_shape = (32, 3, 64, 64)
    window_shape = (3, 3)

    input = torch.randn(input_shape, dtype=torch.float16, device="cuda")

    ninetoothed_output = ops.ninetoothed.torch.avg_pool2d(input, window_shape)
    torch_output = F.avg_pool2d(input, window_shape, ceil_mode=True)

    print(ninetoothed_output)
    print(torch_output)

    if torch.allclose(ninetoothed_output, torch_output):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["h", "w"],
            x_vals=[8 * i for i in range(2, 33)],
            line_arg="provider",
            line_vals=["ninetoothed", "torch"],
            line_names=["NineToothed", "PyTorch"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="ms",
            plot_name="avg-pool2d-performance",
            args={},
        )
    )
    def benchmark(h, w, provider):
        n, c, h, w = 64, 64, h, w
        r, s = 3, 3
        dtype = torch.float16
        device = "cuda"
        input = torch.randn((n, c, h, w), dtype=dtype, device=device)
        window_shape = (r, s)

        if provider == "ninetoothed":
            ms = triton.testing.do_bench(lambda: ops.ninetoothed.torch.avg_pool2d(input, window_shape))
        elif provider == "torch":
            ms = triton.testing.do_bench(lambda: F.avg_pool2d(input, window_shape))

        return ms

    benchmark.run(show_plots=True, print_data=True, save_path=".")
