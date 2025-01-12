import ninetoothed
import torch
import triton
import triton.language as tl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True)


@ninetoothed.jit
def add_kernel(
    lhs: Tensor(1).tile((BLOCK_SIZE,)),
    rhs: Tensor(1).tile((BLOCK_SIZE,)),
    output: Tensor(1).tile((BLOCK_SIZE,)),
):
    output = lhs + rhs  # noqa: F841


def add(lhs, rhs):
    output = torch.empty_like(lhs)

    add_kernel(lhs, rhs, output)

    return output


@triton.jit
def triton_add_kernel(
    lhs_ptr,
    rhs_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    lhs = tl.load(lhs_ptr + offsets, mask=mask)
    rhs = tl.load(rhs_ptr + offsets, mask=mask)
    output = lhs + rhs

    tl.store(output_ptr + offsets, output, mask=mask)


def triton_add(lhs, rhs):
    output = torch.empty_like(lhs)
    n_elements = output.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    triton_add_kernel[grid](lhs, rhs, output, n_elements, BLOCK_SIZE=1024)

    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    dtype = torch.float16
    lhs = torch.rand(size, dtype=dtype, device="cuda")
    rhs = torch.rand(size, dtype=dtype, device="cuda")
    ninetoothed_output = add(lhs, rhs)
    torch_output = lhs + rhs
    triton_output = triton_add(lhs, rhs)
    print(ninetoothed_output)
    print(torch_output)
    print(triton_output)
    if torch.allclose(ninetoothed_output, torch_output):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")
    if torch.allclose(ninetoothed_output, triton_output, atol=0, rtol=0):
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
        lhs = torch.rand(size, device="cuda", dtype=torch.float16)
        rhs = torch.rand(size, device="cuda", dtype=torch.float16)
        quantiles = [0.5, 0.2, 0.8]

        ninetoothed_output = add(lhs, rhs)
        torch_output = lhs + rhs
        triton_output = triton_add(lhs, rhs)
        assert torch.allclose(ninetoothed_output, torch_output)
        assert torch.allclose(ninetoothed_output, triton_output, atol=0, rtol=0)

        if provider == "ninetoothed":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: add(lhs, rhs), quantiles=quantiles
            )
        elif provider == "torch":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: lhs + rhs, quantiles=quantiles
            )
        elif provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: triton_add(lhs, rhs), quantiles=quantiles
            )

        def gbps(ms):
            return 3 * lhs.numel() * lhs.element_size() / ms * 1e-6

        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(print_data=True, show_plots=True, save_path=".")
