import torch

import ops.ninetoothed.kernels.add
import ops.ninetoothed.kernels.mm


def add(input, other):
    output = torch.empty_like(input)

    ops.ninetoothed.kernels.add.kernel(input, other, output, BLOCK_SIZE=1024)

    return output


def mm(input, other):
    output_shape = (input.shape[0], other.shape[1])
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    ops.ninetoothed.kernels.mm.kernel(input, other, output)

    return output
