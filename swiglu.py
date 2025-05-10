import ninetoothed
import ninetoothed.language as ntl
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)


@ninetoothed.jit
def swiglu_kernel(
    a: Tensor(1).tile((BLOCK_SIZE,)),
    b: Tensor(1).tile((BLOCK_SIZE,)),
    c: Tensor(1).tile((BLOCK_SIZE,)),
):
    b_loaded = b
    gate = b_loaded * ntl.sigmoid(ntl.cast(b_loaded, ntl.float32))
    c = a * gate  # noqa: F841


def swiglu(a, b):
    a_1d = a.flatten()
    b_1d = b.flatten()

    c = torch.empty_like(a_1d)

    swiglu_kernel(a_1d, b_1d, c)

    return c.view_as(a)


@triton.jit
def triton_swiglu_kernel(
    a_ptr, b_ptr, c_ptr, num_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    silu_b = b * tl.sigmoid(tl.cast(b, tl.float32))
    c = a * silu_b

    tl.store(c_ptr + offsets, c, mask=mask)


def triton_swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Flatten the inputs so that the kernel always works on 1D tensors
    a_flat = a.flatten()
    b_flat = b.flatten()
    c_flat = torch.empty_like(a_flat)
    num_elements = a_flat.numel()

    def grid(meta):
        return (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

    triton_swiglu_kernel[grid](a_flat, b_flat, c_flat, num_elements, BLOCK_SIZE=1024)

    return c_flat.view_as(a)


def torch_swiglu(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return a * F.silu(b)


if __name__ == "__main__":
    torch.manual_seed(0)

    shape = (13, 3)
    dtype = torch.float16
    device = "cuda"
    a = torch.rand(shape, dtype=dtype, device=device)
    b = torch.rand(shape, dtype=dtype, device=device)
    c = torch.rand(shape, dtype=dtype, device=device)

    ninetoothed_output = swiglu(a, b)
    torch_output = torch_swiglu(a, b)
    triton_output = triton_swiglu(a, b)
    print(ninetoothed_output)
    print(torch_output)
    print(triton_output)

    if torch.allclose(ninetoothed_output, torch_output, atol=0, rtol=1e-3):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")
    if torch.allclose(ninetoothed_output, triton_output):
        print("✅ NineToothed and Triton match.")
    else:
        print("❌ NineToothed and Triton differ.")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "n"],
            x_vals=[128 * i for i in range(2, 50)],
            x_log=True,
            line_arg="provider",
            line_vals=["ninetoothed", "torch", "triton"],
            line_names=["NineToothed", "PyTorch", "Triton"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="GB/s",
            plot_name="swiglu-performance",
            args={},
        )
    )
    def benchmark(m, n, provider):
        shape = (m, n)
        dtype = torch.float16
        device = "cuda"

        a = torch.rand(shape, dtype=dtype, device=device)
        b = torch.rand(shape, dtype=dtype, device=device)
        quantiles = [0.5, 0.2, 0.8]

        if provider == "ninetoothed":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: swiglu(a, b), quantiles=quantiles
            )
        elif provider == "torch":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch_swiglu(a, b), quantiles=quantiles
            )
        elif provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: triton_swiglu(a, b), quantiles=quantiles
            )

        def gbps(ms):
            return 3 * a.numel() * a.element_size() / ms * 1e-6

        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(print_data=True, show_plots=True, save_path=".")
