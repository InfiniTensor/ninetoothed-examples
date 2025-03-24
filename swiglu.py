import torch
import triton
import ninetoothed
import triton.language as tl
import ninetoothed.language as ntl
import torch.nn.functional as F
from ninetoothed import Symbol, Tensor

BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", meta=True)
BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", meta=True)


@ninetoothed.jit
def swiglu_kernel(
    a: Tensor(2).tile((BLOCK_SIZE_M, BLOCK_SIZE_N)),
    b: Tensor(2).tile((BLOCK_SIZE_M, BLOCK_SIZE_N)),
    c: Tensor(2).tile((BLOCK_SIZE_M, BLOCK_SIZE_N)),
):
    magic = b
    gate = magic * ntl.sigmoid(ntl.cast(magic, ntl.float32))
    c = a * gate


def ninetoothed_swiglu(a, b):
    c = torch.empty_like(a)

    swiglu_kernel(a, b, c)

    return c


@triton.jit
def triton_swiglu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    stride_am,
    stride_an,
    stride_bm,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    rows = offsets // N
    cols = offsets % N

    mask = (rows < M) & (cols < N)

    a_offsets = rows * stride_am + cols * stride_an
    b_offsets = rows * stride_bm + cols * stride_bn
    c_offsets = rows * stride_cm + cols * stride_cn

    a = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + b_offsets, mask=mask, other=0.0)

    silu_b = b * tl.sigmoid(tl.cast(b, tl.float32))
    c = a * silu_b

    tl.store(c_ptr + c_offsets, c, mask=mask)


def triton_swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, n = a.shape
    c = torch.empty_like(a)

    grid = lambda meta: (triton.cdiv(m * n, meta["BLOCK_SIZE"]),)

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
    a = torch.rand(shape, device="cuda", dtype=torch.float16)
    b = torch.rand(shape, device="cuda", dtype=torch.float16)
    c = torch.rand(shape, device="cuda", dtype=torch.float16)

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
            x_names=["size"],
            x_vals=[2**i for i in range(12, 28, 1)],
            x_log=True,
            line_arg="provider",
            line_vals=["ninetoothed", "torch", "triton"],
            line_names=["NineToothed", "PyTorch", "Triton"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="GB/s",
            plot_name="vector-addition-performance",
            args={},
        )
    )
    def benchmark(size, provider):
        a = torch.rand(size, device="cuda", dtype=torch.float16)
        b = torch.rand(size, device="cuda", dtype=torch.float16)
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
