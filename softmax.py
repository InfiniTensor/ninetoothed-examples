import ninetoothed
import ninetoothed.language as ntl
import torch
import triton
from ninetoothed import Tensor


def softmax(input):
    output = torch.empty_like(input)

    block_size = triton.next_power_of_2(input.shape[-1])

    @ninetoothed.jit
    def softmax_kernel(
        input_row: Tensor(2, other=float("-inf")).tile((1, block_size)),
        output_row: Tensor(2).tile((1, block_size)),
    ):
        row_minus_max = input_row - ntl.max(input_row)
        numerator = ntl.exp(row_minus_max)
        denominator = ntl.sum(numerator)
        output_row = numerator / denominator  # noqa: F841

    softmax_kernel(input, output)

    return output


torch.manual_seed(0)
input = torch.randn(1823, 781, device="cuda")
torch_output = torch.softmax(input, axis=-1)
ninetoothed_output = softmax(input)
print(torch_output)
print(ninetoothed_output)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg="provider",
        line_vals=["ninetoothed", "torch"],
        line_names=["NineToothed", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={"M": 4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == "ninetoothed":
        ms = triton.testing.do_bench(lambda: softmax(x))

    def gbps(ms):
        return 2 * x.numel() * x.element_size() * 1e-6 / ms

    return gbps(ms)


benchmark.run(show_plots=True, print_data=True, save_path=".")
