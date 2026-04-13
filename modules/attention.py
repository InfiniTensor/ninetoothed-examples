import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import repeat_kv

import ops.ninetoothed.torch
import ops.triton.torch
from modules._utils import _make_backend_manager


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
        input_0 = input[..., : input.shape[-1] // 2]
        input_1 = input[..., input.shape[-1] // 2 :]
        input_0_rotated = input_0 * cos_table - input_1 * sin_table
        input_1_rotated = input_0 * sin_table + input_1 * cos_table

        return torch.cat((input_0_rotated, input_1_rotated), dim=-1)


def generate_sin_and_cos_tables(
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


class Attention(nn.Module):
    scaled_dot_product_attention = None
    rotary_position_embedding = None

    def __init__(self, other):
        super().__init__()
        self.__dict__ = other.__dict__

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_value,
        cache_position,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        cos_table, sin_table = position_embeddings
        sin_table = sin_table[0, ..., sin_table.shape[-1] // 2 :]
        cos_table = cos_table[0, ..., cos_table.shape[-1] // 2 :]

        query_states = type(self).rotary_position_embedding(
            query_states, sin_table, cos_table, interleaved=False
        )
        key_states = type(self).rotary_position_embedding(
            key_states, sin_table, cos_table, interleaved=False
        )

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin_table,
                "cos": cos_table,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = type(self).scaled_dot_product_attention(
            query_states, key_states, value_states, scale=self.scaling
        )
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None


scaled_dot_product_attention_backend = _make_backend_manager(
    Attention,
    "scaled_dot_product_attention",
    {
        "ninetoothed": ops.ninetoothed.torch.scaled_dot_product_attention,
        "triton": ops.triton.torch.scaled_dot_product_attention,
        "torch": F.scaled_dot_product_attention,
    },
)

rotary_position_embedding_backend = _make_backend_manager(
    Attention,
    "rotary_position_embedding",
    {
        "ninetoothed": ops.ninetoothed.torch.rotary_position_embedding,
        "triton": ops.triton.torch.rotary_position_embedding,
        "torch": torch_rotary_position_embedding,
    },
)
