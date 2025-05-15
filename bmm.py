import torch
import triton

import ops.ninetoothed.torch
import ops.triton.torch

if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size, m, n, k = 4, 512, 2028, 1024
    dtype = torch.float16
    device = "cuda"

    input = torch.randn(batch_size, m, k, dtype=dtype, device=device)
    other = torch.randn(batch_size, k, n, dtype=dtype, device=device)

    ninetoothed_output = ops.ninetoothed.torch.bmm(input, other)
    torch_output = torch.bmm(input, other)
    triton_output = ops.triton.torch.bmm(input, other)

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
            plot_name="batched-matrix-multiplication-performance",
            args={"b": 4},
        )
    )
    def benchmark(b, m, n, k, provider):
        input = torch.randn((b, m, k), dtype=dtype, device=device)
        other = torch.randn((b, k, n), dtype=dtype, device=device)

        ninetoothed_output = ops.ninetoothed.torch.bmm(input, other)
        torch_output = torch.bmm(input, other)
        triton_output = ops.triton.torch.bmm(input, other)

        assert torch.allclose(ninetoothed_output, torch_output)
        assert torch.allclose(ninetoothed_output, triton_output, atol=0, rtol=0)

        if provider == "ninetoothed":
            ms = triton.testing.do_bench(
                lambda: ops.ninetoothed.torch.bmm(input, other)
            )
        elif provider == "torch":
            ms = triton.testing.do_bench(lambda: torch.bmm(input, other))
        elif provider == "triton":
            ms = triton.testing.do_bench(lambda: ops.triton.torch.bmm(input, other))

        return ms

    benchmark.run(show_plots=True, print_data=True, save_path=".")
