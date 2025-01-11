import ninetoothed
import ninetoothed.language as ntl
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


@ninetoothed.jit
def rms_norm_kernel(
    input: Tensor(2).tile((1, BLOCK_SIZE)),
    output: Tensor(2).tile((1, BLOCK_SIZE)),
    eps: Tensor(0),
):
    input_fp32 = ntl.cast(input, ntl.float32)
    output = input_fp32 * ntl.rsqrt(  # noqa: F841
        ntl.sum(input_fp32 * input_fp32) / input.shape[-1] + eps
    )


def rms_norm(input, eps=1e-5):
    output = torch.empty_like(input)

    rms_norm_kernel(input, output, eps, BLOCK_SIZE=input.shape[-1])

    return output


@triton.jit
def triton_rms_norm_kernel(
    input_ptr,
    output_ptr,
    num_cols,
    input_row_stride,
    output_row_stride,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
    mask = col_offsets < num_cols
    input = tl.load(input_ptrs, mask=mask)

    output = input * tl.rsqrt(tl.sum(input * input) / num_cols + eps)

    output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
    tl.store(output_ptrs, output, mask=mask)


def triton_rms_norm(input, eps=1e-5):
    output = torch.empty_like(input)

    triton_rms_norm_kernel[(input.shape[-2],)](
        input,
        output,
        input.shape[-1],
        input.stride(-2),
        output.stride(-2),
        eps,
        BLOCK_SIZE=triton.next_power_of_2(input.shape[-1]),
    )

    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    input = torch.randn(1151, 8192, dtype=torch.float16, device="cuda")
    ninetoothed_output = rms_norm(input)
    torch_output = F.rms_norm(input, input.shape[-1:])
    triton_output = triton_rms_norm(input)
    print(ninetoothed_output)
    print(torch_output)
    print(triton_output)
    if torch.allclose(ninetoothed_output, torch_output):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")
    if torch.allclose(ninetoothed_output, triton_output):
        print("✅ NineToothed and Triton match.")
    else:
        print("❌ NineToothed and Triton differ.")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["n"],
            x_vals=[512 * i for i in range(2, 32)],
            line_arg="provider",
            line_vals=["ninetoothed", "torch", "triton"],
            line_names=["NineToothed", "PyTorch", "Triton"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="GB/s",
            plot_name="rms-norm-performance",
            args={"m": 4096},
        )
    )
    def benchmark(m, n, provider):
        input = torch.randn(m, n, dtype=torch.float16, device="cuda")

        if provider == "ninetoothed":
            ms = triton.testing.do_bench(lambda: rms_norm(input))
        elif provider == "torch":
            ms = triton.testing.do_bench(
                lambda: torch.rms_norm(input, input.shape[-1:])
            )
        elif provider == "triton":
            ms = triton.testing.do_bench(lambda: triton_rms_norm(input))

        def gbps(ms):
            return 2 * input.numel() * input.element_size() * 1e-6 / ms

        return gbps(ms)

    benchmark.run(show_plots=True, print_data=True, save_path=".")
