import torch

import ops.tilelang.kernels.mm


def mm(input, other):
    output_shape = (input.shape[0], other.shape[1])
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    ops.tilelang.kernels.mm.kernel(input, other, output)

    return output
