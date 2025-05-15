import math

import torch

import ops.ninetoothed.kernels.add
import ops.ninetoothed.kernels.addmm
import ops.ninetoothed.kernels.bmm
import ops.ninetoothed.kernels.conv2d
import ops.ninetoothed.kernels.fused_rms_norm
import ops.ninetoothed.kernels.mm
import ops.ninetoothed.kernels.rms_norm
import ops.ninetoothed.kernels.scaled_dot_product_attention
import ops.ninetoothed.kernels.silu
import ops.ninetoothed.kernels.softmax


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


def softmax(input):
    output = torch.empty_like(input)

    ops.ninetoothed.kernels.softmax.kernel(input, output, BLOCK_SIZE=input.shape[-1])

    return output
