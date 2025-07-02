import random

import torch
import triton

import ops.ninetoothed.torch
import ops.triton.torch

if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)

    shape = (512, 512)
    dtype = torch.float16
    device = "cuda"

    input = torch.randn(shape, dtype=dtype, device=device)
    mat1 = torch.randn(shape, dtype=dtype, device=device)
    mat2 = torch.randn(shape, dtype=dtype, device=device)
    beta = random.uniform(0, 1)
    alpha = random.uniform(0, 1)

    ninetoothed_output = ops.ninetoothed.torch.addmm(
        input, mat1, mat2, beta=beta, alpha=alpha
    )
    torch_output = torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha)
    triton_output = ops.triton.torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha)

    print(ninetoothed_output)
    print(torch_output)
    print(triton_output)

    if torch.allclose(ninetoothed_output, torch_output, atol=0.01, rtol=0.01):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")
    if torch.allclose(ninetoothed_output, triton_output):
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
            ylabel="ms",
            plot_name="addmm-performance",
            args={},
        )
    )
    def benchmark(m, n, k, provider):
        input = torch.randn((m, n), dtype=dtype, device=device)
        mat1 = torch.randn((m, k), dtype=dtype, device=device)
        mat2 = torch.randn((k, n), dtype=dtype, device=device)
        beta = random.uniform(0, 1)
        alpha = random.uniform(0, 1)

        if provider == "ninetoothed":
            ms = triton.testing.do_bench(
                lambda: ops.ninetoothed.torch.addmm(
                    input, mat1, mat2, beta=beta, alpha=alpha
                )
            )
        elif provider == "torch":
            ms = triton.testing.do_bench(
                lambda: torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha)
            )
        elif provider == "triton":
            ms = triton.testing.do_bench(
                lambda: ops.triton.torch.addmm(
                    input, mat1, mat2, beta=beta, alpha=alpha
                )
            )

        return ms

    benchmark.run(show_plots=True, print_data=True, save_path=".")
