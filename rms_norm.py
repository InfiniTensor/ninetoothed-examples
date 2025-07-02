import torch
import torch.nn.functional as F
import triton

import ops.ninetoothed.torch
import ops.triton.torch

if __name__ == "__main__":
    torch.manual_seed(0)

    dtype = torch.float16
    device = "cuda"

    input = torch.randn(1151, 8192, dtype=dtype, device=device)

    ninetoothed_output = ops.ninetoothed.torch.rms_norm(input)
    torch_output = F.rms_norm(input, input.shape[-1:])
    triton_output = ops.triton.torch.rms_norm(input)

    print(ninetoothed_output)
    print(torch_output)
    print(triton_output)

    if torch.allclose(ninetoothed_output, torch_output, atol=0.001, rtol=0.005):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")
    if torch.allclose(ninetoothed_output, triton_output, atol=0, rtol=0):
        print("✅ NineToothed and Triton match.")
    else:
        print("❌ NineToothed and Triton differ.")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["n"],
            x_vals=[2**i for i in range(5, 15)],
            x_log=True,
            line_arg="provider",
            line_vals=["ninetoothed", "torch", "triton"],
            line_names=["NineToothed", "PyTorch", "Triton"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="ms",
            plot_name="rms-norm-performance",
            args={"m": 4096},
        )
    )
    def benchmark(m, n, provider):
        input = torch.randn(m, n, dtype=dtype, device=device)

        ninetoothed_output = ops.ninetoothed.torch.rms_norm(input)
        torch_output = F.rms_norm(input, input.shape[-1:])
        triton_output = ops.triton.torch.rms_norm(input)

        assert torch.allclose(ninetoothed_output, torch_output, atol=0.001, rtol=0.005)
        assert torch.allclose(ninetoothed_output, triton_output, atol=0, rtol=0)

        if provider == "ninetoothed":
            ms = triton.testing.do_bench(lambda: ops.ninetoothed.torch.rms_norm(input))
        elif provider == "torch":
            ms = triton.testing.do_bench(
                lambda: torch.rms_norm(input, input.shape[-1:])
            )
        elif provider == "triton":
            ms = triton.testing.do_bench(lambda: ops.triton.torch.rms_norm(input))

        return ms

    benchmark.run(show_plots=True, print_data=True, save_path=".")
