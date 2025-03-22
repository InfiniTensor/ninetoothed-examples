import ninetoothed
import torch
from ninetoothed import Tensor


def arrangement(tensor, sin_table, cos_table):
    emb_dim = tensor.shape[-1]
    half_emb_dim = emb_dim // 2

    tensor_arranged = tensor.tile(
        (1, 1, 1, half_emb_dim), strides=(-1, -1, -1, 1), dilation=(1, 1, 1, 2)
    )
    tensor_arranged = tensor_arranged.tile((1, 1, 1, 2))
    tensor_arranged.dtype = tensor_arranged.dtype.squeeze((0, 1, 2))
    tensor_arranged.dtype.dtype = tensor_arranged.dtype.dtype.squeeze((0, 1, 2))

    sin_table_arranged = sin_table.tile((1, 1, 1, half_emb_dim))
    sin_table_arranged.dtype = sin_table_arranged.dtype.squeeze((0, 1, 2))

    cos_table_arranged = cos_table.tile((1, 1, 1, half_emb_dim))
    cos_table_arranged.dtype = cos_table_arranged.dtype.squeeze((0, 1, 2))

    return tensor_arranged, sin_table_arranged, cos_table_arranged


def application(tensor, sin_table, cos_table):
    tensor_0 = tensor[0]
    tensor_1 = tensor[1]

    tensor[0] = tensor_0 * cos_table - tensor_1 * sin_table
    tensor[1] = tensor_0 * sin_table + tensor_1 * cos_table


tensors = tuple(Tensor(4, constexpr_shape=True) for _ in range(3))
rope_kernel = ninetoothed.make(arrangement, application, tensors)


def rope(tensor, sin_table, cos_table):
    batch_size, _, num_heads, _ = tensor.shape

    sin_table = sin_table.unsqueeze(1).unsqueeze(0)
    sin_table = sin_table.expand(batch_size, -1, num_heads, -1)
    cos_table = cos_table.unsqueeze(1).unsqueeze(0)
    cos_table = cos_table.expand(batch_size, -1, num_heads, -1)

    tensor_cloned = tensor.clone()
    rope_kernel(tensor_cloned, sin_table, cos_table)

    return tensor_cloned


def torch_rope(input, sin_table, cos_table):
    batch_size, seq_len, num_heads, emb_dim = input.shape

    assert emb_dim % 2 == 0, "The embedding dimension must be even."

    pair_wise_input = input.view(batch_size, seq_len, num_heads, emb_dim // 2, 2)
    sin_table = sin_table[None, :, None, :]
    cos_table = cos_table[None, :, None, :]

    pair_0, pair_1 = pair_wise_input[..., 0], pair_wise_input[..., 1]
    rotated_pair_0 = pair_0 * cos_table - pair_1 * sin_table
    rotated_pair_1 = pair_0 * sin_table + pair_1 * cos_table

    output = torch.stack((rotated_pair_0, rotated_pair_1), dim=-1).view(input.shape)

    return output


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
    sin_table, cos_table = _generate_sin_and_cos_tables(seq_len, emb_dim)
    dtype = torch.float32
    device = "cuda"
    x = torch.randn(batch_size, seq_len, num_heads, emb_dim, dtype=dtype, device=device)
    ninetoothed_output = rope(x, sin_table, cos_table)
    torch_output = torch_rope(x, sin_table, cos_table)
    print(ninetoothed_output)
    print(torch_output)
    if torch.allclose(ninetoothed_output, torch_output, atol=0.001):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")
