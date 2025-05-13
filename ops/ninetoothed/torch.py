import torch

import ops.ninetoothed.kernels.add
import ops.ninetoothed.kernels.addmm
import ops.ninetoothed.kernels.conv2d
import ops.ninetoothed.kernels.mm
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


def conv2d(input, filter):
    n, _, h, w = input.shape
    k, _, r, s = filter.shape
    p = h - r + 1
    q = w - s + 1

    output = torch.empty((n, k, p, q), dtype=input.dtype, device=input.device)

    ops.ninetoothed.kernels.conv2d.kernel(input, filter, output)

    return output


def mm(input, other):
    output_shape = (input.shape[0], other.shape[1])
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    ops.ninetoothed.kernels.mm.kernel(input, other, output)

    return output


def softmax(input):
    output = torch.empty_like(input)

    ops.ninetoothed.kernels.softmax.kernel(input, output, BLOCK_SIZE=input.shape[-1])

    return output
