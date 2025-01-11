import ninetoothed
import ninetoothed.language as ntl
import torch
import triton
import triton.language as tl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


@ninetoothed.jit
def softmax_kernel(
    input_row: Tensor(2, other=float("-inf")).tile((1, BLOCK_SIZE)),
    output_row: Tensor(2).tile((1, BLOCK_SIZE)),
):
    row_minus_max = input_row - ntl.max(input_row)
    numerator = ntl.exp(row_minus_max)
    denominator = ntl.sum(numerator)
    output_row = numerator / denominator  # noqa: F841


def softmax(input):
    output = torch.empty_like(input)

    softmax_kernel(input, output, BLOCK_SIZE=input.shape[-1])

    return output


@triton.jit
def triton_softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    row = tl.load(input_ptrs, mask=mask, other=float("-inf"))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def triton_softmax(input):
    output = torch.empty_like(input)

    triton_softmax_kernel[(input.shape[0],)](
        input,
        output,
        input.stride(0),
        output.stride(0),
        input.shape[0],
        input.shape[1],
        BLOCK_SIZE=triton.next_power_of_2(input.shape[-1]),
    )

    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    input = torch.randn(1823, 781, dtype=torch.float16, device="cuda")
    ninetoothed_output = softmax(input)
    torch_output = torch.softmax(input, axis=-1)
    triton_output = triton_softmax(input)
    print(ninetoothed_output)
    print(torch_output)
    print(triton_output)
    if torch.allclose(ninetoothed_output, torch_output, atol=1e-5):
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
            x_vals=[128 * i for i in range(2, 100)],
            line_arg="provider",
            line_vals=["ninetoothed", "torch", "triton"],
            line_names=["NineToothed", "PyTorch", "Triton"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="GB/s",
            plot_name="softmax-performance",
            args={"m": 4096},
        )
    )
    def benchmark(m, n, provider):
        input = torch.randn(m, n, device="cuda", dtype=torch.float16)
        stream = torch.cuda.Stream()
        torch.cuda.set_stream(stream)

        if provider == "ninetoothed":
            ms = triton.testing.do_bench(lambda: softmax(input))
        elif provider == "torch":
            ms = triton.testing.do_bench(lambda: torch.softmax(input, axis=-1))
        elif provider == "triton":
            ms = triton.testing.do_bench(lambda: triton_softmax(input))

        def gbps(ms):
            return 2 * input.numel() * input.element_size() * 1e-6 / ms

        return gbps(ms)

    benchmark.run(show_plots=True, print_data=True, save_path=".")
