import math

import ninetoothed
import torch

import ops.ninetoothed.kernels.add
import ops.ninetoothed.kernels.addmm
import ops.ninetoothed.kernels.bmm
import ops.ninetoothed.kernels.conv2d
import ops.ninetoothed.kernels.fused_rms_norm
import ops.ninetoothed.kernels.max_pool2d
import ops.ninetoothed.kernels.mm
import ops.ninetoothed.kernels.rms_norm
import ops.ninetoothed.kernels.rotary_position_embedding
import ops.ninetoothed.kernels.scaled_dot_product_attention
import ops.ninetoothed.kernels.silu
import ops.ninetoothed.kernels.softmax
import ops.ninetoothed.kernels.swiglu

_DTYPE_MAPPING = {
    torch.float16: ninetoothed.float16,
    torch.bfloat16: ninetoothed.bfloat16,
    torch.float32: ninetoothed.float32,
}


def add(input, other):
    output = torch.empty_like(input)

    ops.ninetoothed.kernels.add.kernel(
        input, other, output, _DTYPE_MAPPING[input.dtype]
    )

    return output


def addmm(input, mat1, mat2, beta=1, alpha=1):
    output_shape = (mat1.shape[0], mat2.shape[1])
    output = torch.empty(output_shape, dtype=mat1.dtype, device=mat1.device)

    ops.ninetoothed.kernels.addmm.kernel(input, mat1, mat2, beta, alpha, output)

    return output


def bmm(lhs, rhs):
    output_shape = (lhs.shape[0], lhs.shape[-2], rhs.shape[-1])
    output = torch.empty(output_shape, dtype=lhs.dtype, device=lhs.device)

    ops.ninetoothed.kernels.bmm.kernel(lhs, rhs, output)

    return output


def conv2d(input, filter):
    n, _, h, w = input.shape
    k, _, r, s = filter.shape
    p = h - r + 1
    q = w - s + 1

    output = torch.empty((n, k, p, q), dtype=input.dtype, device=input.device)

    ops.ninetoothed.kernels.conv2d.kernel(input, filter, output)

    return output


def fused_rms_norm(x, w, eps=None):
    if eps is None:
        eps = torch.finfo(x.dtype).eps()

    x_2d = x.view(-1, x.shape[-1])
    w_2d = w.expand_as(x_2d)
    y_2d = torch.empty_like(x_2d)

    n = x.shape[-1]
    block_size = 1 << (n - 1).bit_length()

    ops.ninetoothed.kernels.fused_rms_norm.kernel(
        x_2d, w_2d, float(eps), y_2d, _DTYPE_MAPPING[x.dtype], block_size
    )

    return y_2d.view(x.shape)


def max_pool2d(input, window_shape):
    n, c, h, w = input.shape
    r, s = window_shape
    p = math.ceil((h - r) / r + 1)
    q = math.ceil((w - s) / s + 1)

    output = torch.empty(n, c, p, q, dtype=input.dtype, device=input.device)

    ops.ninetoothed.kernels.max_pool2d.kernel(
        input, output, WINDOW_HEIGHT=r, WINDOW_WIDTH=s
    )

    return output


def mm(input, other):
    m, k = input.shape
    _, n = other.shape
    output = torch.empty((m, n), dtype=input.dtype, device=input.device)

    ops.ninetoothed.kernels.mm.kernel(
        input, other, output, m, n, k, _DTYPE_MAPPING[input.dtype]
    )

    return output


def rms_norm(input, eps=None):
    if eps is None:
        eps = torch.finfo(input.dtype).eps

    output = torch.empty_like(input)

    n = input.shape[-1]
    block_size = 1 << (n - 1).bit_length()

    ops.ninetoothed.kernels.rms_norm.kernel(
        input, float(eps), output, _DTYPE_MAPPING[input.dtype], block_size
    )

    return output


def rotary_position_embedding(input, sin_table, cos_table, interleaved=True):
    batch_size, _, num_heads, _ = input.shape

    output = input.clone()
    sin_table = sin_table[None, :, None, :].expand(batch_size, -1, num_heads, -1)
    cos_table = cos_table[None, :, None, :].expand(batch_size, -1, num_heads, -1)

    ops.ninetoothed.kernels.rotary_position_embedding.kernel(
        output, sin_table, cos_table, interleaved
    )

    return output


def scaled_dot_product_attention(q, k, v, scale=None):
    if scale is None:
        scale = 1 / math.sqrt(q.shape[-1])

    o = torch.empty_like(q)

    ops.ninetoothed.kernels.scaled_dot_product_attention.kernel(q, k, v, scale, o)

    return o


def silu(input):
    input_flat = input.flatten()
    output_flat = torch.empty_like(input_flat)

    ops.ninetoothed.kernels.silu.kernel(
        input_flat, output_flat, _DTYPE_MAPPING[input.dtype]
    )

    return output_flat.view_as(input)


def softmax(input):
    output = torch.empty_like(input)

    n = input.shape[-1]
    block_size = 1 << (n - 1).bit_length()

    ops.ninetoothed.kernels.softmax.kernel(
        input, output, _DTYPE_MAPPING[input.dtype], block_size
    )

    return output


def swiglu(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()

    c = torch.empty_like(a_flat)

    ops.ninetoothed.kernels.swiglu.kernel(a_flat, b_flat, c, _DTYPE_MAPPING[a.dtype])

    return c.view_as(a)
