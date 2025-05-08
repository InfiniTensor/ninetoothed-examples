import math

import ninetoothed
import ninetoothed.language as ntl
import torch
import torch.nn.functional as F
import triton
from ninetoothed import Symbol, Tensor


def arrangement(input, output):
    BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)

    WINDOW_HEIGHT = Symbol("WINDOW_HEIGHT", constexpr=True, upper_bound=16)
    WINDOW_WIDTH = Symbol("WINDOW_WIDTH", constexpr=True, upper_bound=16)

    input_arranged = input.tile((1, 1, WINDOW_HEIGHT, WINDOW_WIDTH))
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=4).flatten(start_dim=1)
    input_arranged = input_arranged.tile((BLOCK_SIZE, -1))

    output_arranged = output.tile((1, 1, 1, 1))
    output_arranged = output_arranged.ravel()
    output_arranged = output_arranged.flatten(end_dim=4).flatten(start_dim=1)
    output_arranged = output_arranged.tile((BLOCK_SIZE, -1))
    output_arranged.dtype = output_arranged.dtype.squeeze(1)

    return input_arranged, output_arranged


def application(input, output):
    output = ntl.max(input, axis=1)  # noqa: F841


max_pool2d_kernel = ninetoothed.make(
    arrangement, application, (Tensor(4, other=float("-inf")), Tensor(4))
)


def max_pool2d(input, window_shape):
    n, c, h, w = input.shape
    r, s = window_shape
    p = math.ceil((h - r) / r + 1)
    q = math.ceil((w - s) / s + 1)

    output = torch.empty(n, c, p, q, dtype=input.dtype, device=input.device)

    max_pool2d_kernel(input, output, WINDOW_HEIGHT=r, WINDOW_WIDTH=s)

    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    input_shape = (32, 3, 64, 64)
    window_shape = (3, 3)
    input = torch.randn(input_shape, dtype=torch.float16, device="cuda")
    ninetoothed_output = max_pool2d(input, window_shape)
    torch_output = F.max_pool2d(input, window_shape, ceil_mode=True)
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
            ylabel="GB/s",
            plot_name="2d-max-pooling-performance",
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
            ms = triton.testing.do_bench(lambda: max_pool2d(input, window_shape))
        elif provider == "torch":
            ms = triton.testing.do_bench(lambda: F.max_pool2d(input, window_shape))

        def gbps(ms):
            return 2 * input.numel() * input.element_size() / ms * 1e-6

        return gbps(ms)

    benchmark.run(show_plots=True, print_data=True, save_path=".")
