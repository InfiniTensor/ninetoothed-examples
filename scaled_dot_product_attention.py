from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
from transformers.models.llama.modeling_llama import repeat_kv

import ops.ninetoothed.torch
import ops.triton.torch
from rotary_position_embedding import torch_rotary_position_embedding


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
        sin_table = sin_table[0]
        cos_table = cos_table[0]

        query_states = type(self).rotary_position_embedding(
            query_states, sin_table, cos_table
        )
        key_states = type(self).rotary_position_embedding(
            key_states, sin_table, cos_table
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


@contextmanager
def scaled_dot_product_attention_backend(backend_name):
    _prev_impl = Attention.scaled_dot_product_attention

    if backend_name == "ninetoothed":
        impl = ops.ninetoothed.torch.scaled_dot_product_attention
    elif backend_name == "triton":
        impl = ops.triton.torch.scaled_dot_product_attention
    elif backend_name == "torch":
        impl = F.scaled_dot_product_attention
    else:
        raise ValueError(f"unknown backend: `{backend_name}`")

    Attention.scaled_dot_product_attention = impl

    try:
        yield
    finally:
        Attention.scaled_dot_product_attention = _prev_impl


@contextmanager
def rotary_position_embedding_backend(backend_name):
    _prev_impl = Attention.rotary_position_embedding

    if backend_name == "ninetoothed":
        impl = ops.ninetoothed.torch.rotary_position_embedding
    elif backend_name == "triton":
        impl = ops.triton.torch.rotary_position_embedding
    elif backend_name == "torch":
        impl = torch_rotary_position_embedding
    else:
        raise ValueError(f"unknown backend: `{backend_name}`")

    Attention.rotary_position_embedding = impl

    try:
        yield
    finally:
        Attention.rotary_position_embedding = _prev_impl


if __name__ == "__main__":
    torch.manual_seed(0)

    q_o_shape = (2, 8, 1024, 64)
    k_v_shape = (2, 8, 1024, 64)
    dtype = torch.float16
    device = "cuda"

    q = torch.randn(q_o_shape, dtype=dtype, device=device)
    k = torch.randn(k_v_shape, dtype=dtype, device=device)
    v = torch.randn(k_v_shape, dtype=dtype, device=device)

    ninetoothed_output = ops.ninetoothed.torch.scaled_dot_product_attention(q, k, v)
    torch_output = F.scaled_dot_product_attention(q, k, v)
    triton_output = ops.triton.torch.scaled_dot_product_attention(q, k, v)

    print(ninetoothed_output)
    print(torch_output)
    print(triton_output)

    if torch.allclose(ninetoothed_output, torch_output, atol=0.01):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")
    if torch.allclose(ninetoothed_output, triton_output, atol=1e-3, rtol=0):
        print("✅ NineToothed and Triton match.")
    else:
        print("❌ NineToothed and Triton differ.")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=[2**i for i in range(7, 17)],
            x_log=True,
            line_arg="provider",
            line_vals=["ninetoothed", "torch", "triton"],
            line_names=["NineToothed", "PyTorch", "Triton"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="ms",
            plot_name="scaled-dot-product-attention-performance",
            args={},
        )
    )
    def benchmark(seq_len, provider):
        batch_size, num_heads, emb_dim = 4, 32, 64
        shape = (batch_size, num_heads, seq_len, emb_dim)
        dtype = torch.float16
        device = "cuda"

        q = torch.randn(shape, dtype=dtype, device=device)
        k = torch.randn(shape, dtype=dtype, device=device)
        v = torch.randn(shape, dtype=dtype, device=device)

        ninetoothed_output = ops.ninetoothed.torch.scaled_dot_product_attention(q, k, v)
        torch_output = F.scaled_dot_product_attention(q, k, v)
        triton_output = ops.triton.torch.scaled_dot_product_attention(q, k, v)

        assert torch.allclose(ninetoothed_output, torch_output, atol=0.025, rtol=0.025)
        assert torch.allclose(ninetoothed_output, triton_output, atol=0.001, rtol=0.001)

        if provider == "ninetoothed":
            ms = triton.testing.do_bench(
                lambda: ops.ninetoothed.torch.scaled_dot_product_attention(q, k, v)
            )
        elif provider == "torch":
            ms = triton.testing.do_bench(
                lambda: F.scaled_dot_product_attention(q, k, v)
            )
        elif provider == "triton":
            ms = triton.testing.do_bench(
                lambda: ops.triton.torch.scaled_dot_product_attention(q, k, v)
            )

        return ms

    benchmark.run(show_plots=True, print_data=True, save_path=".")
