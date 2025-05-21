import torch
import triton

import ops.ninetoothed.torch
import ops.triton.torch

if __name__ == "__main__":
    torch.manual_seed(0)

    shape = (512, 512)
    dtype = torch.float16
    device = "cuda"

    input = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ops.ninetoothed.torch.mm(input, other)
    torch_output = torch.mm(input, other)
    triton_output = ops.triton.torch.mm(input, other)

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
            x_vals=[2**i for i in range(3, 13)],
            x_log=True,
            line_arg="provider",
            line_vals=["ninetoothed", "torch", "triton"],
            line_names=["NineToothed", "PyTorch", "Triton"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="ms",
            plot_name="mm-performance",
            args={},
        )
    )
    def benchmark(m, n, k, provider):
        input = torch.randn((m, k), dtype=dtype, device=device)
        other = torch.randn((k, n), dtype=dtype, device=device)

        ninetoothed_output = ops.ninetoothed.torch.mm(input, other)
        torch_output = torch.mm(input, other)
        triton_output = ops.triton.torch.mm(input, other)

        assert torch.allclose(ninetoothed_output, torch_output, atol=0.025, rtol=0.025)
        assert torch.allclose(ninetoothed_output, triton_output, atol=0, rtol=0)

        if provider == "ninetoothed":
            ms = triton.testing.do_bench(lambda: ops.ninetoothed.torch.mm(input, other))
        elif provider == "torch":
            ms = triton.testing.do_bench(lambda: torch.mm(input, other))
        elif provider == "triton":
            ms = triton.testing.do_bench(lambda: ops.triton.torch.mm(input, other))

        return ms

    benchmark.run(show_plots=True, print_data=True, save_path=".")
