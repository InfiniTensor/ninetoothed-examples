import torch
import torch.nn.functional as F
import triton

import ops.ninetoothed.torch
import ops.triton.torch

if __name__ == "__main__":
    torch.manual_seed(0)

    n, c, h, w = 4, 3, 224, 224
    k, _, r, s = 8, c, 3, 3
    dtype = torch.float16
    device = "cuda"

    input = torch.randn(n, c, h, w, dtype=dtype, device=device)
    filter = torch.randn(k, c, r, s, dtype=dtype, device=device)

    ninetoothed_output = ops.ninetoothed.torch.conv2d(input, filter)
    torch_output = F.conv2d(input, filter)
    triton_output = ops.triton.torch.triton_conv2d(input, filter)

    print(ninetoothed_output)
    print(torch_output)
    print(triton_output)

    if torch.allclose(ninetoothed_output, torch_output, atol=0.01, rtol=0.01):
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
            x_vals=[2**i for i in range(1, 11)],
            x_log=True,
            line_arg="provider",
            line_vals=["ninetoothed", "torch", "triton"],
            line_names=["NineToothed", "PyTorch", "Triton"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="ms",
            plot_name="2d-convolution-performance",
            args={},
        )
    )
    def benchmark(n, provider):
        _, c, h, w = n, 512, 14, 14
        k, _, r, s = 512, c, 3, 3

        input = torch.randn((n, c, h, w), dtype=dtype, device=device)
        filter = torch.randn((k, c, r, s), dtype=dtype, device=device)

        ninetoothed_output = ops.ninetoothed.torch.conv2d(input, filter)
        torch_output = F.conv2d(input, filter)
        triton_output = ops.triton.torch.triton_conv2d(input, filter)

        assert torch.allclose(ninetoothed_output, torch_output, atol=0.01, rtol=0.01)
        assert torch.allclose(ninetoothed_output, triton_output, atol=0, rtol=0)

        if provider == "ninetoothed":
            ms = triton.testing.do_bench(
                lambda: ops.ninetoothed.torch.conv2d(input, filter)
            )
        elif provider == "torch":
            ms = triton.testing.do_bench(lambda: F.conv2d(input, filter))
        elif provider == "triton":
            ms = triton.testing.do_bench(
                lambda: ops.triton.torch.triton_conv2d(input, filter)
            )

        return ms

    benchmark.run(show_plots=True, print_data=True, save_path=".")
