import torch
import triton

import ops.ninetoothed.torch
import ops.triton.torch


def torch_rotary_position_embedding(input, sin_table, cos_table, interleaved=True):
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

    ninetoothed_output = ops.ninetoothed.torch.rotary_position_embedding(
        x, sin_table, cos_table, interleaved=False
    )
    torch_output = torch_rotary_position_embedding(
        x, sin_table, cos_table, interleaved=False
    )
    triton_output = ops.triton.torch.rotary_position_embedding(
        x, sin_table, cos_table, interleaved=False
    )

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
            plot_name="rotary-position-embedding-performance",
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
            ms = triton.testing.do_bench(
                lambda: ops.ninetoothed.torch.rotary_position_embedding(
                    x, sin_table, cos_table
                )
            )
        elif provider == "torch":
            ms = triton.testing.do_bench(
                lambda: torch_rotary_position_embedding(x, sin_table, cos_table)
            )
        elif provider == "triton":
            ms = triton.testing.do_bench(
                lambda: ops.triton.torch.rotary_position_embedding(
                    x, sin_table, cos_table
                )
            )

        return ms

    benchmark.run(show_plots=True, print_data=True, save_path=".")
