import torch
import triton

import ops.triton.kernels.add
import ops.triton.kernels.mm


def add(input, other):
    num_elements = input.numel()

    output = torch.empty_like(input)

    def grid(meta):
        return (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

    ops.triton.kernels.add.kernel[grid](
        input, other, output, num_elements, BLOCK_SIZE=1024
    )

    return output


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
