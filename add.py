import torch
import triton

import ops.ninetoothed.torch
import ops.triton.torch

if __name__ == "__main__":
    torch.manual_seed(0)

    size = 98432
    dtype = torch.float16
    device = "cuda"

    input = torch.randn(size, dtype=dtype, device=device)
    other = torch.randn(size, dtype=dtype, device=device)

    ninetoothed_output = ops.ninetoothed.torch.add(input, other)
    torch_output = input + other
    triton_output = ops.triton.torch.add(input, other)

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
            x_names=["size"],
            x_vals=[2**i for i in range(18, 28)],
            x_log=True,
            line_arg="provider",
            line_vals=["ninetoothed", "torch", "triton"],
            line_names=["NineToothed", "PyTorch", "Triton"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="ms",
            plot_name="add-performance",
            args={},
        )
    )
    def benchmark(size, provider):
        input = torch.randn(size, dtype=dtype, device=device)
        other = torch.randn(size, dtype=dtype, device=device)

        ninetoothed_output = ops.ninetoothed.torch.add(input, other)
        torch_output = torch.add(input, other)
        triton_output = ops.triton.torch.add(input, other)

        assert torch.allclose(ninetoothed_output, torch_output)
        assert torch.allclose(ninetoothed_output, triton_output, atol=0, rtol=0)

        if provider == "ninetoothed":
            ms = triton.testing.do_bench(
                lambda: ops.ninetoothed.torch.add(input, other)
            )
        elif provider == "torch":
            ms = triton.testing.do_bench(lambda: torch.add(input, other))
        elif provider == "triton":
            ms = triton.testing.do_bench(lambda: ops.triton.torch.add(input, other))

        return ms

    benchmark.run(print_data=True, show_plots=True, save_path=".")
