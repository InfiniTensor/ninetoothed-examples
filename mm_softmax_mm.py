import ninetoothed
import torch
import triton

import ops.ninetoothed.torch


def ninetoothed_mm_softmax_mm(a, b, c, use_fused=True):
    if use_fused:
        return ninetoothed_mm_softmax_mm_fused(a, b, c)

    return ninetoothed_mm_softmax_mm_unfused(a, b, c)


@torch.compile(backend=ninetoothed.fuser)
def ninetoothed_mm_softmax_mm_fused(a, b, c):
    return ninetoothed_mm_softmax_mm_unfused(a, b, c)


def ninetoothed_mm_softmax_mm_unfused(a, b, c):
    return ninetoothed_mm(ninetoothed_softmax(ninetoothed_mm(a, b)), c)


def ninetoothed_softmax(a):
    return ops.ninetoothed.torch.softmax(a, impl_id=1)


def ninetoothed_mm(a, b):
    return ops.ninetoothed.torch.mm(a, b, impl_id=1)


def torch_mm_softmax_mm(a, b, c, use_compiled=True):
    if use_compiled:
        return torch_mm_softmax_mm_compiled(a, b, c)

    return torch_mm_softmax_mm_uncompiled(a, b, c)


def torch_mm_softmax_mm_uncompiled(a, b, c):
    return torch.mm(torch.softmax(torch.mm(a, b), dim=-1), c)


@torch.compile
def torch_mm_softmax_mm_compiled(a, b, c):
    return torch_mm_softmax_mm_uncompiled(a, b, c)


if __name__ == "__main__":
    torch.manual_seed(0)

    shape = (512, 512)
    dtype = torch.float16
    device = "cuda"

    a = torch.randn(shape, dtype=dtype, device=device)
    b = torch.randn(shape, dtype=dtype, device=device)
    c = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ninetoothed_mm_softmax_mm(a, b, c)
    torch_output = torch_mm_softmax_mm(a, b, c)

    print(ninetoothed_output)
    print(torch_output)

    if torch.allclose(ninetoothed_output, torch_output, rtol=1e-3, atol=1e-3):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m"],
            x_vals=[2**i for i in range(3, 13)],
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
            plot_name="mm-softmax-mm-performance",
            args={"n": 128},
        )
    )
    def benchmark(m, n, provider):
        a = torch.randn((m, n), dtype=dtype, device=device)
        b = torch.randn((n, m), dtype=dtype, device=device)
        c = torch.randn((m, n), dtype=dtype, device=device)

        if provider == "ninetoothed_fused":
            ninetoothed_output = ninetoothed_mm_softmax_mm(a, b, c)
            torch_output = torch_mm_softmax_mm(a, b, c)

            assert torch.allclose(
                ninetoothed_output, torch_output, rtol=1e-3, atol=1e-3
            )
        elif provider == "ninetoothed_unfused":
            ninetoothed_output = ninetoothed_mm_softmax_mm(a, b, c, use_fused=False)
            torch_output = torch_mm_softmax_mm(a, b, c, use_compiled=False)

            assert torch.allclose(
                ninetoothed_output, torch_output, rtol=1e-3, atol=1e-3
            )

        if provider == "ninetoothed_fused":
            ms = triton.testing.do_bench(lambda: ninetoothed_mm_softmax_mm(a, b, c))
        elif provider == "torch_compiled":
            ms = triton.testing.do_bench(lambda: torch_mm_softmax_mm(a, b, c))
        elif provider == "ninetoothed_unfused":
            ms = triton.testing.do_bench(
                lambda: ninetoothed_mm_softmax_mm(a, b, c, use_fused=False)
            )
        elif provider == "torch_uncompiled":
            ms = triton.testing.do_bench(
                lambda: torch_mm_softmax_mm(a, b, c, use_compiled=False)
            )

        return ms

    benchmark.run(show_plots=True, print_data=True, save_path=".")
