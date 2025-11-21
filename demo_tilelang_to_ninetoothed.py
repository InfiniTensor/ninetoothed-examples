import torch
import triton

import ops.ninetoothed.kernels.mm
import ops.tilelang.kernels.mm
import ops.triton.kernels.mm
import tilelang_to_ninetoothed

BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32

ninetoothed_mm_kernel = ops.ninetoothed.kernels.mm.kernel

triton_mm_kernel = ops.triton.kernels.mm.kernel

tilelang_mm_kernel = ops.tilelang.kernels.mm.mm(
    ops.tilelang.kernels.mm.M,
    ops.tilelang.kernels.mm.N,
    ops.tilelang.kernels.mm.K,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
)

ninetoothed_mm_kernel_from_tilelang = (
    tilelang_to_ninetoothed.transform_tilelang_to_ninetoothed(
        ops.tilelang.kernels.mm.mm
    )
)


def ninetoothed_mm(input, other):
    output_shape = (input.shape[0], other.shape[1])
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    ninetoothed_mm_kernel(input, other, output)

    return output


def triton_mm(input, other):
    output_shape = (input.shape[0], other.shape[1])
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    def grid(meta):
        return (
            triton.cdiv(input.shape[0], meta["BLOCK_SIZE_M"])
            * triton.cdiv(other.shape[1], meta["BLOCK_SIZE_N"]),
        )

    triton_mm_kernel[grid](
        input,
        other,
        output,
        input.shape[0],
        other.shape[1],
        input.shape[1],
        input.stride(0),
        input.stride(1),
        other.stride(0),
        other.stride(1),
        output.stride(0),
        output.stride(1),
    )

    return output


def tilelang_mm(input, other):
    output_shape = (input.shape[0], other.shape[1])
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    tilelang_mm_kernel(input, other, output)

    return output


def ninetoothed_from_tilelang_mm(input, other):
    m, k = input.shape
    _, n = other.shape

    output = torch.empty((m, n), dtype=input.dtype, device=input.device)

    ninetoothed_mm_kernel_from_tilelang(
        input,
        other,
        output,
        M=m,
        N=n,
        K=k,
        block_M=BLOCK_SIZE_M,
        block_N=BLOCK_SIZE_N,
        block_K=BLOCK_SIZE_K,
    )

    return output


def torch_mm(input, other):
    return torch.mm(input, other)


if __name__ == "__main__":
    torch.manual_seed(0)

    shape = (512, 512)
    dtype = torch.float16
    device = "cuda"

    input = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)

    ninetoothed_output = ninetoothed_mm(input, other)
    torch_output = torch_mm(input, other)
    triton_output = triton_mm(input, other)
    tilelang_output = tilelang_mm(input, other)
    ninetoothed_from_tilelang_output = ninetoothed_from_tilelang_mm(input, other)

    print(ninetoothed_output)
    print(torch_output)
    print(triton_output)
    print(tilelang_output)
    print(ninetoothed_from_tilelang_output)

    if torch.allclose(ninetoothed_output, torch_output):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")
    if torch.allclose(ninetoothed_output, triton_output, atol=0, rtol=0):
        print("✅ NineToothed and Triton match.")
    else:
        print("❌ NineToothed and Triton differ.")
    if torch.allclose(ninetoothed_output, tilelang_output):
        print("✅ NineToothed and TileLang match.")
    else:
        print("❌ NineToothed and TileLang differ.")
    if torch.allclose(ninetoothed_output, ninetoothed_from_tilelang_output):
        print("✅ NineToothed and NineToothed from TileLang match.")
    else:
        print("❌ NineToothed and NineToothed from TileLang differ.")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["m", "n", "k"],
            x_vals=[2**i for i in range(8, 13)],
            x_log=True,
            line_arg="provider",
            line_vals=[
                "ninetoothed",
                "torch",
                "triton",
                "tilelang",
                "ninetoothed_from_tilelang",
            ],
            line_names=[
                "NineToothed",
                "PyTorch",
                "Triton",
                "TileLang",
                "NineToothed from TileLang",
            ],
            styles=[
                ("blue", "-"),
                ("green", "-"),
                ("orange", "-"),
                ("cyan", "-"),
                ("purple", "-"),
            ],
            ylabel="ms",
            plot_name="mm-performance",
            args={},
        )
    )
    def benchmark(m, n, k, provider):
        input = torch.randn((m, k), dtype=dtype, device=device)
        other = torch.randn((k, n), dtype=dtype, device=device)

        if provider == "ninetoothed":
            ms = triton.testing.do_bench(lambda: ninetoothed_mm(input, other))
        elif provider == "torch":
            ms = triton.testing.do_bench(lambda: torch_mm(input, other))
        elif provider == "triton":
            ms = triton.testing.do_bench(lambda: triton_mm(input, other))
        elif provider == "tilelang":
            ms = triton.testing.do_bench(lambda: tilelang_mm(input, other))
        elif provider == "ninetoothed_from_tilelang":
            ms = triton.testing.do_bench(
                lambda: ninetoothed_from_tilelang_mm(input, other)
            )

        return ms

    benchmark.run(show_plots=True, print_data=True, save_path=".")
