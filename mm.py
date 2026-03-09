import torch
import triton

import ops.ninetoothed.torch
import ops.triton.torch
import ops.tilelang.torch

if __name__ == "__main__":
    torch.manual_seed(0)

    shape = (512, 512)
    dtype = torch.float32
    device = "cuda"

    input = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)

    # TileLang: 先进行自动调优获取最优配置的 kernel
    tilelang_kernel = ops.tilelang.torch.tune_mm(
        m=shape[0], n=shape[1], k=shape[0], profile_backend="event", dtype=dtype
    )

    ninetoothed_output = ops.ninetoothed.torch.mm(input, other)
    torch_output = torch.mm(input, other)
    triton_output = ops.triton.torch.mm(input, other)
    tilelang_output = ops.tilelang.torch.mm(input, other, kernel=tilelang_kernel)

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

    # TileLang 正确性检查
    if torch.allclose(ninetoothed_output, tilelang_output, atol=1e-2, rtol=1e-2):
        print("✅ NineToothed and TileLang match.")
    else:
        print("❌ NineToothed and TileLang differ.")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "n", "k"],
            x_vals=[2**i for i in range(3, 13)],
            x_log=True,
            line_arg="provider",
            line_vals=["ninetoothed", "torch", "triton", "tilelang"],
            line_names=["NineToothed", "PyTorch", "Triton", "TileLang"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-"), ("red", "--")],
            ylabel="ms",
            plot_name="mm-performance",
            args={},
        )
    )
    def benchmark(m, n, k, provider):
        input = torch.randn((m, k), dtype=dtype, device=device)
        other = torch.randn((k, n), dtype=dtype, device=device)

        # TileLang: 先进行自动调优获取最优配置的 kernel
        tilelang_kernel = ops.tilelang.torch.tune_mm(
            m=m, n=n, k=k, profile_backend="event", dtype=dtype
        )

        ninetoothed_output = ops.ninetoothed.torch.mm(input, other)
        torch_output = torch.mm(input, other)
        triton_output = ops.triton.torch.mm(input, other)
        tilelang_output = ops.tilelang.torch.mm(input, other, kernel=tilelang_kernel)

        assert torch.allclose(ninetoothed_output, torch_output, atol=0.025, rtol=0.025)
        assert torch.allclose(ninetoothed_output, triton_output, atol=0, rtol=0)
        assert torch.allclose(ninetoothed_output, tilelang_output, atol=1e-2, rtol=1e-2)

        if provider == "ninetoothed":
            ms = triton.testing.do_bench(lambda: ops.ninetoothed.torch.mm(input, other))
        elif provider == "torch":
            ms = triton.testing.do_bench(lambda: torch.mm(input, other))
        elif provider == "triton":
            ms = triton.testing.do_bench(lambda: ops.triton.torch.mm(input, other))
        elif provider == "tilelang":
            ms = triton.testing.do_bench(lambda: ops.tilelang.torch.mm(input, other, kernel=tilelang_kernel))

        return ms

    benchmark.run(show_plots=True, print_data=True, save_path="./mm_fp32")
