import functools

import ninetoothed
import torch
import triton
import triton.language as tl
from ninetoothed import Tensor


def arrangement(tensor, sin_table, cos_table, interleaved=True):
    emb_dim = tensor.shape[-1]
    tile_shape = (1, 1, 1, emb_dim // 2)

    if interleaved:
        strides = (-1, -1, -1, 1)
        dilation = (1, 1, 1, 2)
    else:
        strides = None
        dilation = None

    tensor_arranged = tensor.tile(tile_shape, strides=strides, dilation=dilation)
    tensor_arranged = tensor_arranged.tile((1, 1, 1, 2))
    tensor_arranged.dtype = tensor_arranged.dtype.squeeze((0, 1, 2))
    tensor_arranged.dtype.dtype = tensor_arranged.dtype.dtype.squeeze((0, 1, 2))

    sin_table_arranged = sin_table.tile(tile_shape)
    sin_table_arranged.dtype = sin_table_arranged.dtype.squeeze((0, 1, 2))

    cos_table_arranged = cos_table.tile(tile_shape)
    cos_table_arranged.dtype = cos_table_arranged.dtype.squeeze((0, 1, 2))

    return tensor_arranged, sin_table_arranged, cos_table_arranged


def application(tensor, sin_table, cos_table):
    sin_table_loaded = sin_table
    cos_table_loaded = cos_table

    tensor_0 = tensor[0]
    tensor_1 = tensor[1]

    tensor[0] = tensor_0 * cos_table_loaded - tensor_1 * sin_table_loaded
    tensor[1] = tensor_0 * sin_table_loaded + tensor_1 * cos_table_loaded


tensors = tuple(Tensor(4, shape_options={"constexpr": True}) for _ in range(3))

interleaved_kernel = ninetoothed.make(
    functools.partial(arrangement, interleaved=True), application, tensors
)
non_interleaved_kernel = ninetoothed.make(
    functools.partial(arrangement, interleaved=False), application, tensors
)


def rope_kernel(tensor, sin_table, cos_table, interleaved=True):
    return (interleaved_kernel if interleaved else non_interleaved_kernel)(
        tensor, sin_table, cos_table
    )


def rope(tensor, sin_table, cos_table, interleaved=True):
    batch_size, _, num_heads, _ = tensor.shape

    sin_table = sin_table.unsqueeze(1).unsqueeze(0)
    sin_table = sin_table.expand(batch_size, -1, num_heads, -1)
    cos_table = cos_table.unsqueeze(1).unsqueeze(0)
    cos_table = cos_table.expand(batch_size, -1, num_heads, -1)

    tensor_cloned = tensor.clone()
    rope_kernel(tensor_cloned, sin_table, cos_table, interleaved)

    return tensor_cloned


@triton.jit
def triton_rope_kernel(
    tensor_ptr,
    sin_table_ptr,
    cos_table_ptr,
    tensor_stride_n,
    tensor_stride_l,
    tensor_stride_h,
    tensor_stride_e,
    sin_table_stride_l,
    cos_table_stride_l,
    emb_dim,
    INTERLEAVED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    off_n = tl.program_id(0)
    off_l = tl.program_id(1)
    off_h = tl.program_id(2)

    offs = tl.arange(0, BLOCK_SIZE)

    half_emb_dim = emb_dim // 2
    mask = offs < half_emb_dim

    sin_table = tl.load(sin_table_ptr + off_l * sin_table_stride_l + offs, mask=mask)
    cos_table = tl.load(cos_table_ptr + off_l * cos_table_stride_l + offs, mask=mask)

    even_offs = (
        off_n * tensor_stride_n + off_l * tensor_stride_l + off_h * tensor_stride_h
    )
    odd_offs = (
        off_n * tensor_stride_n + off_l * tensor_stride_l + off_h * tensor_stride_h
    )

    if INTERLEAVED:
        even_offs += (2 * offs) * tensor_stride_e
        odd_offs += (2 * offs + 1) * tensor_stride_e
    else:
        even_offs += offs * tensor_stride_e
        odd_offs += (offs + half_emb_dim) * tensor_stride_e

    even_ptrs = tensor_ptr + even_offs
    odd_ptrs = tensor_ptr + odd_offs

    even = tl.load(even_ptrs, mask=mask)
    odd = tl.load(odd_ptrs, mask=mask)

    tl.store(even_ptrs, even * cos_table - odd * sin_table, mask=mask)
    tl.store(odd_ptrs, even * sin_table + odd * cos_table, mask=mask)


def triton_rope(tensor, sin_table, cos_table, interleaved=True):
    batch_size, seq_len, num_heads, emb_dim = tensor.shape

    assert emb_dim % 2 == 0, "The embedding dimension must be even."

    BLOCK_SIZE = triton.next_power_of_2(emb_dim // 2)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024

    grid = (batch_size, seq_len, num_heads)

    tensor_cloned = tensor.clone()

    triton_rope_kernel[grid](
        tensor_cloned,
        sin_table,
        cos_table,
        *tensor.stride(),
        sin_table.stride(0),
        cos_table.stride(0),
        emb_dim,
        INTERLEAVED=interleaved,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return tensor_cloned


def torch_rope(input, sin_table, cos_table, interleaved=True):
    batch_size, seq_len, num_heads, emb_dim = input.shape

    assert emb_dim % 2 == 0, "The embedding dimension must be even."

    sin_table = sin_table[None, :, None, :]
    cos_table = cos_table[None, :, None, :]

    if interleaved:
        pair_wise_input = input.view(batch_size, seq_len, num_heads, emb_dim // 2, 2)
        input_0, input_1 = pair_wise_input[..., 0], pair_wise_input[..., 1]
        input_0_rotated = input_0 * cos_table - input_1 * sin_table
        input_1_rotated = input_0 * sin_table + input_1 * cos_table

        return torch.stack((input_0_rotated, input_1_rotated), dim=-1).view(input.shape)
    else:
        input_0 = x[..., : x.shape[-1] // 2]
        input_1 = x[..., x.shape[-1] // 2 :]
        input_0_rotated = input_0 * cos_table - input_1 * sin_table
        input_1_rotated = input_0 * sin_table + input_1 * cos_table

        return torch.cat((input_0_rotated, input_1_rotated), dim=-1)


def _generate_sin_and_cos_tables(
    seq_len, emb_dim, base=10000, dtype=torch.float32, device="cuda"
):
    assert emb_dim % 2 == 0, "The embedding dimension must be even."

    theta = base ** (
        -2 * (torch.arange(emb_dim // 2, dtype=dtype, device=device) / emb_dim)
    )

    positions = torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(1)
    sin_table = torch.sin(positions * theta)
    cos_table = torch.cos(positions * theta)

    return sin_table, cos_table


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size, seq_len, num_heads, emb_dim = 4, 128, 8, 64
    dtype = torch.float32
    device = "cuda"

    sin_table, cos_table = _generate_sin_and_cos_tables(seq_len, emb_dim)
    x = torch.randn(batch_size, seq_len, num_heads, emb_dim, dtype=dtype, device=device)

    ninetoothed_output = rope(x, sin_table, cos_table, interleaved=False)
    torch_output = torch_rope(x, sin_table, cos_table, interleaved=False)
    triton_output = triton_rope(x, sin_table, cos_table, interleaved=False)

    print(ninetoothed_output)
    print(torch_output)
    print(triton_output)

    if torch.allclose(ninetoothed_output, torch_output, atol=0.001):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")
    if torch.allclose(ninetoothed_output, triton_output, atol=0, rtol=0):
        print("✅ NineToothed and Triton match.")
    else:
        print("❌ NineToothed and Triton differ.")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=[2**i for i in range(5, 15)],
            line_arg="provider",
            line_vals=["ninetoothed", "torch", "triton"],
            line_names=["NineToothed", "PyTorch", "Triton"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="ms",
            plot_name="rope-performance",
            args={},
        )
    )
    def benchmark(seq_len, provider):
        batch_size, num_heads, emb_dim = 4, 32, 64
        shape = (batch_size, seq_len, num_heads, emb_dim)
        dtype = torch.float16
        device = "cuda"

        sin_table, cos_table = _generate_sin_and_cos_tables(seq_len, emb_dim)
        x = torch.randn(shape, dtype=dtype, device=device)

        if provider == "ninetoothed":
            ms = triton.testing.do_bench(lambda: rope(x, sin_table, cos_table))
        elif provider == "torch":
            ms = triton.testing.do_bench(lambda: torch_rope(x, sin_table, cos_table))
        elif provider == "triton":
            ms = triton.testing.do_bench(lambda: triton_rope(x, sin_table, cos_table))

        return ms

    benchmark.run(show_plots=True, print_data=True, save_path=".")
