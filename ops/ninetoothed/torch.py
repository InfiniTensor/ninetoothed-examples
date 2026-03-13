import math

import torch

import ops.ninetoothed.kernels.add
import ops.ninetoothed.kernels.addmm
import ops.ninetoothed.kernels.bmm
import ops.ninetoothed.kernels.conv2d
import ops.ninetoothed.kernels.fused_rms_norm
import ops.ninetoothed.kernels.mm
import ops.ninetoothed.kernels.rms_norm
import ops.ninetoothed.kernels.rotary_position_embedding
import ops.ninetoothed.kernels.scaled_dot_product_attention
import ops.ninetoothed.kernels.silu
import ops.ninetoothed.kernels.softmax
import ops.ninetoothed.kernels.swiglu
import ops.ninetoothed.kernels.relu
import ops.ninetoothed.kernels.max_pool2d
import ops.ninetoothed.kernels.avg_pool2d

def add(input, other):
    output = torch.empty_like(input)

    ops.ninetoothed.kernels.add.kernel(input, other, output, BLOCK_SIZE=1024)

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

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if isinstance(stride, int):
        stride = (stride, stride)

    # TODO: Support `padding != 0`.
    assert padding == 0, "`padding != 0` is not supported yet."

    if isinstance(padding, str):
        if padding == "valid":
            padding = 0

    if isinstance(padding, int):
        padding = (padding, padding)

    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    # TODO: Support `groups != 1`.
    assert groups == 1, "`groups != 1` is not supported yet."

    n, _, h, w = input.shape
    k, _, r, s = weight.shape
    p = math.floor((h + 2 * padding[0] - dilation[0] * (r - 1) - 1) / stride[0] + 1)
    q = math.floor((w + 2 * padding[1] - dilation[1] * (s - 1) - 1) / stride[1] + 1)

    output = torch.empty((n, k, p, q), dtype=input.dtype, device=input.device)

    if bias is None:
        bias = torch.zeros((k,), dtype=output.dtype, device=output.device)

    bias = bias[None, :, None, None].expand_as(output)

    # kernel = _cached_make(ntops.kernels.conv2d.premake)

    ops.ninetoothed.kernels.conv2d.kernels[(stride[0],dilation[0])](
        input,
        weight,
        bias,
        output,
        # stride_h=stride[0],
        # stride_w=stride[1],
        # dilation_h=dilation[0],
        # dilation_w=dilation[1],
    )

    return output

def fused_rms_norm(x, w, eps=None):
    if eps is None:
        eps = torch.finfo(x.dtype).eps()

    x_2d = x.view(-1, x.shape[-1])
    w_2d = w.expand_as(x_2d)
    y_2d = torch.empty_like(x_2d)

    ops.ninetoothed.kernels.fused_rms_norm.kernel(
        x_2d, w_2d, eps, y_2d, BLOCK_SIZE=x.shape[-1]
    )

    return y_2d.view(x.shape)


def mm(input, other):
    output_shape = (input.shape[0], other.shape[1])
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    ops.ninetoothed.kernels.mm.kernel(input, other, output)

    return output


def rms_norm(input, eps=None):
    if eps is None:
        eps = torch.finfo(input.dtype).eps

    output = torch.empty_like(input)

    ops.ninetoothed.kernels.rms_norm.kernel(
        input, eps, output, BLOCK_SIZE=input.shape[-1]
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

    ops.ninetoothed.kernels.silu.kernel(input_flat, output_flat, BLOCK_SIZE=1024)

    return output_flat.view_as(input)

def relu(input):
    input_flat = input.flatten()
    output_flat = torch.empty_like(input_flat)

    ops.ninetoothed.kernels.relu.kernel(input_flat, output_flat, BLOCK_SIZE=1024)

    return output_flat.view_as(input)


def softmax(input):
    output = torch.empty_like(input)

    ops.ninetoothed.kernels.softmax.kernel(input, output, BLOCK_SIZE=input.shape[-1])

    return output


def swiglu(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()

    c = torch.empty_like(a_flat)

    ops.ninetoothed.kernels.swiglu.kernel(a_flat, b_flat, c, BLOCK_SIZE=1024)

    return c.view_as(a)

def max_pool2d(input, window_shape):
    n, c, h, w = input.shape
    r, s = window_shape
    p = math.ceil((h - r) / r + 1)
    q = math.ceil((w - s) / s + 1)

    output = torch.empty(n, c, p, q, dtype=input.dtype, device=input.device)

    ops.ninetoothed.kernels.max_pool2d.kernel(input, output, WINDOW_HEIGHT=r, WINDOW_WIDTH=s)
    
    return output

def avg_pool2d(input, window_shape):
    n, c, h, w = input.shape
    r, s = window_shape
    p = math.ceil((h - r) / r + 1)
    q = math.ceil((w - s) / s + 1)

    output = torch.empty(n, c, p, q, dtype=input.dtype, device=input.device)

    ops.ninetoothed.kernels.avg_pool2d.kernel(input, output, WINDOW_HEIGHT=r, WINDOW_WIDTH=s)

    return output