import torch
import triton

import ops.ninetoothed.torch
import ops.triton.torch
import ops.tilelang.torch

if __name__ == "__main__":
    torch.manual_seed(0)

    shape = (1024, 1024)
    dtype = torch.bfloat16
    device = "cuda"

    input = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ops.ninetoothed.torch.add(input, other)
    torch_output = input + other
    triton_output = ops.triton.torch.add(input, other)
    tilelang_output = ops.tilelang.torch.add(input, other)

    print(ninetoothed_output)
    print(torch_output)
    print(triton_output)
    print(tilelang_output)

    if torch.allclose(ninetoothed_output, torch_output):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")
    if torch.allclose(ninetoothed_output, triton_output, atol=0, rtol=0):
        print("✅ NineToothed and Triton match.")
    else:
        print("❌ NineToothed and Triton differ.")
    if torch.allclose(ninetoothed_output, tilelang_output, atol=0, rtol=0):
        print("✅ NineToothed and TileLang match.")
    else:
        print("❌ NineToothed and TileLang differ.")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "n"],
            x_vals=[2**i for i in range(5, 15)],
            x_log=True,
            line_arg="provider",
            line_vals=["ninetoothed", "torch", "triton", "tilelang"],
            line_names=["NineToothed", "PyTorch", "Triton", "TileLang"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-"), ("red", "--")],
            ylabel="ms",
            plot_name="add-performance",
            args={},
        )
    )
    def benchmark(m, n, provider):
        input = torch.randn((m, n), dtype=dtype, device=device)
        other = torch.randn((m, n), dtype=dtype, device=device)

        ninetoothed_output = ops.ninetoothed.torch.add(input, other)
        torch_output = torch.add(input, other)
        triton_output = ops.triton.torch.add(input, other)
        tilelang_output = ops.tilelang.torch.add(input, other)

        assert torch.allclose(ninetoothed_output, torch_output)
        assert torch.allclose(ninetoothed_output, triton_output, atol=0, rtol=0)
        assert torch.allclose(ninetoothed_output, tilelang_output, atol=0, rtol=0)

        if provider == "ninetoothed":
            ms = triton.testing.do_bench(
                lambda: ops.ninetoothed.torch.add(input, other)
            )
        elif provider == "torch":
            ms = triton.testing.do_bench(lambda: torch.add(input, other))
        elif provider == "triton":
            ms = triton.testing.do_bench(lambda: ops.triton.torch.add(input, other))
        elif provider == "tilelang":
            ms = triton.testing.do_bench(lambda: ops.tilelang.torch.add(input, other))

        return ms

    benchmark.run(print_data=True, show_plots=True, save_path="./add_bf16/")
