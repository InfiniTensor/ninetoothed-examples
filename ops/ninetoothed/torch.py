import torch

import ops.ninetoothed.kernels.add


def add(input, other):
    output = torch.empty_like(input)

    ops.ninetoothed.kernels.add.kernel(input, other, output, BLOCK_SIZE=1024)

    return output
