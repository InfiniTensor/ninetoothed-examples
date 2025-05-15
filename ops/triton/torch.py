import math

import torch
import triton

import ops.triton.kernels.add
import ops.triton.kernels.addmm
import ops.triton.kernels.bmm
import ops.triton.kernels.conv2d
import ops.triton.kernels.fused_rms_norm
import ops.triton.kernels.mm
import ops.triton.kernels.rms_norm
import ops.triton.kernels.scaled_dot_product_attention
import ops.triton.kernels.softmax


def add(input, other):
    num_elements = input.numel()

    output = torch.empty_like(input)

    def grid(meta):
        return (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

    ops.triton.kernels.add.kernel[grid](
        input, other, output, num_elements, BLOCK_SIZE=1024
    )

    return output


def addmm(input, mat1, mat2, beta=1, alpha=1):
    output_shape = (mat1.shape[0], mat2.shape[1])
    output = torch.empty(output_shape, dtype=mat1.dtype, device=mat1.device)

    def grid(meta):
        return (
            triton.cdiv(mat1.shape[0], meta["BLOCK_SIZE_M"])
            * triton.cdiv(mat2.shape[1], meta["BLOCK_SIZE_N"]),
        )

    ops.triton.kernels.addmm.kernel[grid](
        input,
        mat1,
        mat2,
        output,
        mat1.shape[0],
        mat2.shape[1],
        mat1.shape[1],
        input.stride(0),
        input.stride(1),
        mat1.stride(0),
        mat1.stride(1),
        mat2.stride(0),
        mat2.stride(1),
        output.stride(0),
        output.stride(1),
        beta,
        alpha,
    )

    return output


def bmm(input, other):
    batch, m, k = input.shape
    _, _, n = other.shape
    output = torch.empty((batch, m, n), dtype=input.dtype, device=input.device)

    def grid(meta):
        return (
            batch
            * triton.cdiv(m, meta["BLOCK_SIZE_M"])
            * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
        )

    ops.triton.kernels.bmm.kernel[grid](
        input,
        other,
        output,
        m,
        n,
        k,
        *input.stride(),
        *other.stride(),
        *output.stride(),
    )

    return output


def conv2d(input, filter):
    n, c, h, w = input.shape
    k, _, r, s = filter.shape
    p = h - r + 1
    q = w - s + 1

    output = torch.empty((n, k, p, q), dtype=input.dtype, device=input.device)

    def grid(meta):
        return (
            triton.cdiv(n * p * q, meta["BLOCK_SIZE_M"])
            * triton.cdiv(k, meta["BLOCK_SIZE_N"]),
        )

    ops.triton.kernels.conv2d.kernel[grid](
        input,
        filter,
        output,
        n,
        c,
        h,
        w,
        k,
        r,
        s,
        *input.stride(),
        *filter.stride(),
        *output.stride(),
    )

    return output


def fused_rms_norm(x, w, eps=None):
    if eps is None:
        eps = torch.finfo(x.dtype).eps

    x_2d = x.view(-1, x.shape[-1])
    w_2d = w.expand_as(x_2d)
    y_2d = torch.empty_like(x_2d)

    ops.triton.kernels.fused_rms_norm.kernel[(x_2d.shape[-2],)](
        x_2d,
        w_2d,
        y_2d,
        x_2d.shape[-1],
        x_2d.stride(0),
        w_2d.stride(0),
        y_2d.stride(0),
        eps,
        BLOCK_SIZE=triton.next_power_of_2(x_2d.shape[-1]),
    )

    return y_2d.view(x.shape)


def mm(input, other):
    output_shape = (input.shape[0], other.shape[1])
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    def grid(meta):
        return (
            triton.cdiv(input.shape[0], meta["BLOCK_SIZE_M"])
            * triton.cdiv(other.shape[1], meta["BLOCK_SIZE_N"]),
        )

    ops.triton.kernels.mm.kernel[grid](
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


def rms_norm(input, eps=None):
    if eps is None:
        eps = torch.finfo(input.dtype).eps

    output = torch.empty_like(input)

    ops.triton.kernels.rms_norm.kernel[(input.shape[-2],)](
        input,
        output,
        input.shape[-1],
        input.stride(-2),
        output.stride(-2),
        eps,
        BLOCK_SIZE=triton.next_power_of_2(input.shape[-1]),
    )

    return output


def scaled_dot_product_attention(q, k, v, scale=None):
    batch_size, num_heads, seq_len, emb_dim = q.shape

    if scale is None:
        scale = 1 / math.sqrt(emb_dim)

    o = torch.empty_like(q)

    def grid(meta):
        return (
            triton.cdiv(seq_len, meta["BLOCK_SIZE_M"]),
            num_heads,
            batch_size,
        )

    ops.triton.kernels.scaled_dot_product_attention.kernel[grid](
        q,
        k,
        v,
        o,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *o.stride(),
        scale=scale,
        seq_len=seq_len,
        EMB_DIM=emb_dim,
    )

    return o


def softmax(input):
    output = torch.empty_like(input)

    ops.triton.kernels.softmax.kernel[(input.shape[0],)](
        input,
        output,
        input.stride(0),
        output.stride(0),
        input.shape[1],
        BLOCK_SIZE=triton.next_power_of_2(input.shape[-1]),
    )

    return output
