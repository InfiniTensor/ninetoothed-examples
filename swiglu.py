import ninetoothed
import ninetoothed.language as ntl
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", meta=True)
BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", meta=True)


@ninetoothed.jit
def swiglu_kernel(
    a: Tensor(2).tile((BLOCK_SIZE_M, BLOCK_SIZE_N)),
    b: Tensor(2).tile((BLOCK_SIZE_M, BLOCK_SIZE_N)),
    c: Tensor(2).tile((BLOCK_SIZE_M, BLOCK_SIZE_N)),
):
    b_loaded = b
    gate = b_loaded * ntl.sigmoid(ntl.cast(b_loaded, ntl.float32))
    c = a * gate  # noqa: F841


def ninetoothed_swiglu(a, b):
    c = torch.empty_like(a)

    swiglu_kernel(a, b, c)

    return c


@triton.jit
def triton_swiglu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    a_stride_m,
    a_stride_n,
    b_stride_m,
    b_stride_n,
    c_stride_m,
    c_stride_n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    rows = offsets // n
    cols = offsets % n

    mask = (rows < m) & (cols < n)

    a_offsets = rows * a_stride_m + cols * a_stride_n
    b_offsets = rows * b_stride_m + cols * b_stride_n
    c_offsets = rows * c_stride_m + cols * c_stride_n

    a = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + b_offsets, mask=mask, other=0.0)

    silu_b = b * tl.sigmoid(tl.cast(b, tl.float32))
    c = a * silu_b

    tl.store(c_ptr + c_offsets, c, mask=mask)


def triton_swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n = a.shape
    c = torch.empty_like(a)

    def grid(meta):
        return (triton.cdiv(m * n, meta["BLOCK_SIZE"]),)

    triton_swiglu_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE=1024,
    )

    return c


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

    ninetoothed_output = ninetoothed_swiglu(a, b)
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
                lambda: ninetoothed_swiglu(a, b), quantiles=quantiles
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
