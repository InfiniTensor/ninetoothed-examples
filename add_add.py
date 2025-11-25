import ninetoothed
import torch
import triton

import ops.ninetoothed.torch


def ninetoothed_add_add(a, b, c, use_fused=True):
    if use_fused:
        return ninetoothed_add_add_fused(a, b, c)

    return ninetoothed_add_add_unfused(a, b, c)


@torch.compile(backend=ninetoothed.fuse)
def ninetoothed_add_add_fused(a, b, c):
    return ninetoothed_add_add_unfused(a, b, c)


def ninetoothed_add_add_unfused(a, b, c):
    return ninetoothed_add(ninetoothed_add(a, b), c)


def ninetoothed_add(a, b):
    return ops.ninetoothed.torch.add(a, b, impl_id=1)


def torch_add_add(a, b, c, use_compiled=True):
    if use_compiled:
        return torch_add_add_compiled(a, b, c)

    return torch_add_add_uncompiled(a, b, c)


def torch_add_add_uncompiled(a, b, c):
    return a + b + c


@torch.compile
def torch_add_add_compiled(a, b, c):
    return torch_add_add_uncompiled(a, b, c)


if __name__ == "__main__":
    torch.manual_seed(0)

    size = 76045
    dtype = torch.float16
    device = "cuda"

    a = torch.randn(size, dtype=dtype, device=device)
    b = torch.randn(size, dtype=dtype, device=device)
    c = torch.randn(size, dtype=dtype, device=device)

    ninetoothed_output = ninetoothed_add_add(a, b, c)
    torch_output = torch_add_add(a, b, c)

    print(ninetoothed_output)
    print(torch_output)

    if torch.allclose(ninetoothed_output, torch_output, rtol=1e-2, atol=1e-2):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["size"],
            x_vals=[2**i for i in range(18, 28)],
            x_log=True,
            line_arg="provider",
            line_vals=[
                "ninetoothed_fused",
                "torch_compiled",
                "ninetoothed_unfused",
                "torch_uncompiled",
            ],
            line_names=[
                "NineToothed (Fused)",
                "PyTorch (Compiled)",
                "NineToothed (Unfused)",
                "PyTorch (Uncompiled)",
            ],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-"), ("purple", "-")],
            ylabel="ms",
            plot_name="add-add-performance",
            args={},
        )
    )
    def benchmark(size, provider):
        a = torch.randn(size, dtype=dtype, device=device)
        b = torch.randn(size, dtype=dtype, device=device)
        c = torch.randn(size, dtype=dtype, device=device)

        if provider == "ninetoothed_fused":
            ninetoothed_output = ninetoothed_add_add(a, b, c)
            torch_output = torch_add_add(a, b, c)

            assert torch.allclose(
                ninetoothed_output, torch_output, rtol=1e-2, atol=1e-2
            )
        elif provider == "ninetoothed_unfused":
            ninetoothed_output = ninetoothed_add_add(a, b, c, use_fused=False)
            torch_output = torch_add_add(a, b, c, use_compiled=False)

            assert torch.allclose(
                ninetoothed_output, torch_output, rtol=1e-2, atol=1e-2
            )

        if provider == "ninetoothed_fused":
            ms = triton.testing.do_bench(lambda: ninetoothed_add_add(a, b, c))
        elif provider == "torch_compiled":
            ms = triton.testing.do_bench(lambda: torch_add_add(a, b, c))
        elif provider == "ninetoothed_unfused":
            ms = triton.testing.do_bench(
                lambda: ninetoothed_add_add(a, b, c, use_fused=False)
            )
        elif provider == "torch_uncompiled":
            ms = triton.testing.do_bench(
                lambda: torch_add_add(a, b, c, use_compiled=False)
            )

        return ms

    benchmark.run(print_data=True, show_plots=True, save_path=".")
