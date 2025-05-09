import functools
import math

import ninetoothed
import ninetoothed.language as ntl
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from ninetoothed import Symbol, Tensor
from transformers.models.llama.modeling_llama import repeat_kv

import rope


def arrangement(q, k, v, scale, o):
    BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", meta=True)
    BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", meta=True)

    def arrange_q_or_o(input):
        arranged = input.tile((1, 1, BLOCK_SIZE_M, -1))
        arranged.dtype = arranged.dtype.squeeze((0, 1))

        return arranged

    def arrange_k_or_v(input):
        arranged = (
            input.tile((1, 1, BLOCK_SIZE_N, -1))
            .tile((1, 1, -1, -1))
            .expand((-1, -1, q_arranged.shape[-2], -1))
        )
        arranged.dtype = arranged.dtype.squeeze((0, 1, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))

        return arranged

    q_arranged = arrange_q_or_o(q)

    return q_arranged, arrange_k_or_v(k), arrange_k_or_v(v), scale, arrange_q_or_o(o)


def application(q, k, v, scale, o):
    q_loaded = (q * scale * 1.44269504089).to(ntl.float16)

    acc = ntl.zeros((q.shape[-2], q.shape[-1]), dtype=ntl.float32)
    l_i = ntl.full((q.shape[-2],), 1, dtype=ntl.float32)
    m_i = ntl.full((q.shape[-2],), float("-inf"), dtype=ntl.float32)

    for i in range(k.shape[0]):
        qk = ntl.dot(q_loaded, ntl.trans(k[i]))

        m_ij = ntl.maximum(m_i, ntl.max(qk, 1))
        p = ntl.exp2(qk - m_ij[:, None])
        l_ij = ntl.sum(p, 1)

        alpha = ntl.exp2(m_i - m_ij)
        acc = acc * alpha[:, None] + ntl.dot(p.to(ntl.float16), v[i])
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    acc /= l_i[:, None]
    o = acc  # noqa: F841


q, k, v, o = (
    Tensor(4, shape_options=(None, None, None, {"constexpr": True, "upper_bound": 128}))
    for _ in range(4)
)
attention_kernel = ninetoothed.make(arrangement, application, (q, k, v, Tensor(0), o))


def attention(q, k, v, scale=None):
    if scale is None:
        scale = 1 / math.sqrt(q.shape[-1])

    o = torch.empty_like(q, dtype=v.dtype)

    attention_kernel(q, k, v, scale, o)

    return o


class Attention(nn.Module):
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

        _rope(query_states, sin_table, cos_table)
        _rope(key_states, sin_table, cos_table)

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

        attn_dtype = torch.float16
        attn_output = attention(
            query_states.to(attn_dtype),
            key_states.to(attn_dtype),
            value_states.to(attn_dtype),
            scale=self.scaling,
        ).to(query_states.dtype)
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128}, num_stages=4, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64}, num_stages=4, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64}, num_stages=4, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32}, num_stages=4, num_warps=8
        ),
    ],
    key=["EMB_DIM"],
)
@triton.jit
def triton_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    q_stride_z,
    q_stride_h,
    q_stride_m,
    q_stride_k,
    k_stride_z,
    k_stride_h,
    k_stride_n,
    k_stride_k,
    v_stride_z,
    v_stride_h,
    v_stride_k,
    v_stride_n,
    o_stride_z,
    o_stride_h,
    o_stride_m,
    o_stride_n,
    scale,
    SEQ_LEN: tl.constexpr,
    EMB_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    off_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    offs_m_start = off_m * BLOCK_SIZE_M

    q_off = off_z * q_stride_z + off_h * q_stride_h
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_off,
        shape=(SEQ_LEN, EMB_DIM),
        strides=(q_stride_m, q_stride_k),
        offsets=(offs_m_start, 0),
        block_shape=(BLOCK_SIZE_M, EMB_DIM),
        order=(1, 0),
    )
    k_off = off_z * k_stride_z + off_h * k_stride_h
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + k_off,
        shape=(EMB_DIM, SEQ_LEN),
        strides=(k_stride_k, k_stride_n),
        offsets=(0, 0),
        block_shape=(EMB_DIM, BLOCK_SIZE_N),
        order=(0, 1),
    )
    v_off = off_z * v_stride_z + off_h * v_stride_h
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + v_off,
        shape=(SEQ_LEN, EMB_DIM),
        strides=(v_stride_k, v_stride_n),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_N, EMB_DIM),
        order=(1, 0),
    )
    o_off = off_z * o_stride_z + off_h * o_stride_h
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + o_off,
        shape=(SEQ_LEN, EMB_DIM),
        strides=(o_stride_m, o_stride_n),
        offsets=(offs_m_start, 0),
        block_shape=(BLOCK_SIZE_M, EMB_DIM),
        order=(1, 0),
    )

    q = (tl.load(q_block_ptr) * scale * 1.44269504089).to(q_block_ptr.type.element_ty)

    acc = tl.zeros((BLOCK_SIZE_M, EMB_DIM), dtype=tl.float32)
    l_i = tl.full((BLOCK_SIZE_M,), 1, dtype=tl.float32)
    m_i = tl.full((BLOCK_SIZE_M,), float("-inf"), dtype=tl.float32)

    for _ in range(0, tl.cdiv(SEQ_LEN, BLOCK_SIZE_N)):
        k = tl.load(k_block_ptr)

        qk = tl.dot(q, k)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.exp2(qk)
        l_ij = tl.sum(p, 1)

        v = tl.load(v_block_ptr)
        alpha = tl.exp2(m_i - m_ij)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_block_ptr.type.element_ty), v)
        m_i = m_ij
        l_i = l_i * alpha + l_ij

        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_SIZE_N, 0))
        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_SIZE_N))

    acc /= l_i[:, None]

    tl.store(o_block_ptr, acc.to(o_ptr.type.element_ty))


def triton_attention(q, k, v, scale=None):
    o = torch.empty_like(q)

    batch_size, num_heads, seq_len, emb_dim = q.shape

    if scale is None:
        scale = 1 / math.sqrt(emb_dim)

    def grid(meta):
        return (
            triton.cdiv(seq_len, meta["BLOCK_SIZE_M"]),
            num_heads,
            batch_size,
        )

    triton_attention_kernel[grid](
        q,
        k,
        v,
        o,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *o.stride(),
        scale=scale,
        SEQ_LEN=seq_len,
        EMB_DIM=emb_dim,
    )

    return o


_rope_kernel = ninetoothed.make(
    functools.partial(rope.arrangement, interleaved=False),
    rope.application,
    rope.tensors,
)


def _rope(x, sin_table, cos_table):
    _, _, num_heads, _ = x.shape
    sin_table = sin_table.unsqueeze(2).expand(-1, -1, num_heads, -1)
    cos_table = cos_table.unsqueeze(2).expand(-1, -1, num_heads, -1)

    _rope_kernel(x, sin_table, cos_table)


if __name__ == "__main__":
    torch.manual_seed(0)
    shape = (2, 4, 1024, 64)
    dtype = torch.float16
    q = torch.randn(shape, dtype=dtype, device="cuda")
    k = torch.randn(shape, dtype=dtype, device="cuda")
    v = torch.randn(shape, dtype=dtype, device="cuda")

    ninetoothed_output = attention(q, k, v)
    torch_output = F.scaled_dot_product_attention(q, k, v)
    triton_output = triton_attention(q, k, v)
    print(ninetoothed_output)
    print(torch_output)
    print(triton_output)
    if torch.allclose(ninetoothed_output, torch_output, atol=0.01):
        print("✅ NineToothed and PyTorch match.")
    else:
        print("❌ NineToothed and PyTorch differ.")
    if torch.allclose(ninetoothed_output, triton_output, atol=0.01):
        print("✅ NineToothed and Triton match.")
    else:
        print("❌ NineToothed and Triton differ.")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=[2**i for i in range(10, 15)],
            line_arg="provider",
            line_vals=["ninetoothed", "torch", "triton"],
            line_names=["NineToothed", "PyTorch", "Triton"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="TFLOPS",
            plot_name="attention-performance",
            args={},
        )
    )
    def benchmark(seq_len, provider):
        batch_size, num_heads, emb_dim = 4, 32, 64
        shape = (batch_size, num_heads, seq_len, emb_dim)
        dtype = torch.float16
        q = torch.randn(shape, dtype=dtype, device="cuda")
        k = torch.randn(shape, dtype=dtype, device="cuda")
        v = torch.randn(shape, dtype=dtype, device="cuda")

        if provider == "ninetoothed":
            ms = triton.testing.do_bench(lambda: attention(q, k, v))
        elif provider == "torch":
            ms = triton.testing.do_bench(
                lambda: F.scaled_dot_product_attention(q, k, v, scale=1)
            )
        elif provider == "triton":
            ms = triton.testing.do_bench(lambda: triton_attention(q, k, v))

        def perf(ms):
            flops_per_matmul = 2 * batch_size * num_heads * seq_len * seq_len * emb_dim
            total_flops = 2 * flops_per_matmul

            return total_flops * 1e-12 / (ms * 1e-3)

        return perf(ms)

    benchmark.run(show_plots=True, print_data=True, save_path=".")
