import torch
import torch.nn.functional as F
import triton

import ops.ninetoothed.torch
import ops.triton.torch


def torch_swiglu(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return a * F.silu(b)


if __name__ == "__main__":
    torch.manual_seed(0)

    shape = (13, 3)
    dtype = torch.float16
    device = "cuda"

    a = torch.rand(shape, dtype=dtype, device=device)
    b = torch.rand(shape, dtype=dtype, device=device)
    c = torch.rand(shape, dtype=dtype, device=device)

    ninetoothed_output = ops.ninetoothed.torch.swiglu(a, b)
    torch_output = torch_swiglu(a, b)
    triton_output = ops.triton.torch.swiglu(a, b)

    print(ninetoothed_output)
    print(torch_output)
    print(triton_output)

    if torch.allclose(ninetoothed_output, torch_output, atol=0, rtol=1e-3):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")
    if torch.allclose(ninetoothed_output, triton_output):
        print("✅ NineToothed and Triton match.")
    else:
        print("❌ NineToothed and Triton differ.")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "n"],
            x_vals=[128 * i for i in range(2, 50)],
            x_log=True,
            line_arg="provider",
            line_vals=["ninetoothed", "torch", "triton"],
            line_names=["NineToothed", "PyTorch", "Triton"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="ms",
            plot_name="swiglu-performance",
            args={},
        )
    )
    def benchmark(m, n, provider):
        shape = (m, n)

        a = torch.rand(shape, dtype=dtype, device=device)
        b = torch.rand(shape, dtype=dtype, device=device)

        if provider == "ninetoothed":
            ms = triton.testing.do_bench(lambda: ops.ninetoothed.torch.swiglu(a, b))
        elif provider == "torch":
            ms = triton.testing.do_bench(lambda: torch_swiglu(a, b))
        elif provider == "triton":
            ms = triton.testing.do_bench(lambda: ops.triton.torch.swiglu(a, b))

        return ms

    benchmark.run(print_data=True, show_plots=True, save_path=".")
