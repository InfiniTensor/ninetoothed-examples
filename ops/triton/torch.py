import torch
import triton

import ops.triton.kernels.add


def add(input, other):
    num_elements = input.numel()

    output = torch.empty_like(input)

    def grid(meta):
        return (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

    ops.triton.kernels.add.kernel[grid](
        input, other, output, num_elements, BLOCK_SIZE=1024
    )

    return output
